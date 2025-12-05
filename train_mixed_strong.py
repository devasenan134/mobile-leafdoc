# train_mixed_strong.py
import argparse, json, random, math
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score


# ----------------- utils -----------------
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- image prepro -----------------
class GrayWorld:
    def __call__(self, img: Image.Image):
        x = np.asarray(img).astype(np.float32)
        m = x.reshape(-1,3).mean(axis=0) + 1e-6
        x = np.clip(x * (float(np.mean(m))/m), 0, 255).astype(np.uint8)
        return Image.fromarray(x)

class HSVLeafCrop:
    def __init__(self, min_frac=0.03, max_frac=0.97, pad=8): self.min_frac, self.max_frac, self.pad = min_frac, max_frac, pad
    def __call__(self, img: Image.Image):
        x = np.asarray(img.convert("RGB"))
        xf = x.astype(np.float32)/255.0
        mx, mn = xf.max(2), xf.min(2); diff = mx - mn + 1e-6
        h = np.zeros_like(mx)
        r,g,b = xf[...,0], xf[...,1], xf[...,2]
        h = np.where(mx==r, ((g-b)/diff)%6, h); h = np.where(mx==g, ((b-r)/diff)+2, h); h = np.where(mx==b, ((r-g)/diff)+4, h)
        h = h/6.0; s = np.where(mx==0, 0.0, diff/mx); v = mx
        mask = (((h>0.17)&(h<0.47)&(s>0.15)&(v>0.15)) | ((h>0.10)&(h<=0.17)&(s>0.10)&(v>0.15)))
        ys, xs = np.where(mask)
        if ys.size==0: return img
        y0,y1,x0,x1 = ys.min(), ys.max(), xs.min(), xs.max()
        H,W = mask.shape; frac = (y1-y0+1)*(x1-x0+1)/(H*W)
        if not (self.min_frac <= frac <= self.max_frac): return img
        y0=max(0,y0-self.pad); y1=min(H,y1+self.pad); x0=max(0,x0-self.pad); x1=min(W,x1+self.pad)
        return Image.fromarray(x[y0:y1, x0:x1, :])


# ----------------- dataset -----------------
class MixedDataset(Dataset):
    def __init__(self, paths, labels, img_size=288, normalize=False, train=True,
                 color_constancy=False, leaf_crop=False, strong_aug=True,
                 random_erasing_p=0.25):
        t=[]
        if leaf_crop:       t += [HSVLeafCrop()]
        if color_constancy: t += [GrayWorld()]
        if train:
            if strong_aug:
                t += [
                    transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(0.2,0.2,0.2,0.1),
                    transforms.RandomPerspective(0.2, p=0.2),
                    transforms.GaussianBlur(3),
                ]
            else:
                t += [transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
                      transforms.RandomHorizontalFlip(0.5)]
        else:
            t += [transforms.Resize((img_size, img_size))]
        t += [transforms.ToTensor()]
        if normalize: t += [transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        if train and random_erasing_p>0:
            t += [transforms.RandomErasing(p=random_erasing_p, scale=(0.02,0.2), ratio=(0.3,3.3))]
        self.tf = transforms.Compose(t)
        self.P = paths; self.L = labels

    def __len__(self): return len(self.P)
    def __getitem__(self, i):
        x = Image.open(self.P[i]).convert("RGB")
        x = self.tf(x)
        return x, int(self.L[i])


# ----------------- sampler (domain + class balancing) -----------------
def make_sampler(labels, sources, pd_weight=3.0, syn_weight=0.8, class_weight_from="pd"):
    """
    - Class weights computed from PD only (default) to reflect real PD imbalance.
    - Domain multipliers: pv=1.0, pd=pd_weight, syn=syn_weight.
    """
    sel = [i for i,s in enumerate(sources) if (class_weight_from=="all") or (s=="pd")]
    counts = defaultdict(int)
    for i in sel:
        counts[int(labels[i])] += 1
    # smooth
    for c in set(labels):
        counts.setdefault(int(c), 1)

    total = sum(counts.values()); C = len(counts)
    class_w = {c: total/(C*n) for c,n in counts.items()}  # inverse freq, normalized

    w = []
    for y,s in zip(labels, sources):
        base = class_w.get(int(y), 1.0)
        mult = 1.0 if s=="pv" else (pd_weight if s=="pd" else syn_weight)
        w.append(base * mult)
    weights = torch.tensor(w, dtype=torch.float)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


# ----------------- MixUp -----------------
def apply_mixup(x, y, alpha=0.2):
    if alpha <= 0: return x, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x = lam*x + (1-lam)*x[idx]
    y_a, y_b = y, y[idx]
    return x, (y_a, y_b, lam)

def mixup_criterion(criterion, logits, y_info, y_true):
    if y_info is None:
        return criterion(logits, y_true)
    y_a, y_b, lam = y_info
    return lam*criterion(logits, y_a) + (1-lam)*criterion(logits, y_b)


# ----------------- class weights (effective number) from PD only -----------------
def effective_class_weights(pd_labels_only, num_classes, beta=0.999):
    cnt = np.bincount(np.array(pd_labels_only, dtype=np.int64), minlength=num_classes)
    cnt[cnt==0] = 1
    eff = (1 - np.power(beta, cnt)) / (1 - beta)
    w = 1.0 / eff
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


# ----------------- model init -----------------
def build_student(unified_classes, init_from_pv, pv_to_unified, device):
    num_u = len(unified_classes)
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if init_from_pv is None else None)
    in_features = m.classifier[3].in_features

    if init_from_pv is None:
        m.classifier[3] = nn.Linear(in_features, num_u)
        m.to(device)
        return m

    # Load PV checkpoint and copy overlapping heads
    ckpt = torch.load(init_from_pv, map_location="cpu")
    sd = ckpt["model"]; pv_classes = ckpt.get("classes", None)
    if pv_classes and len(pv_classes)>0:
        pv_num = len(pv_classes)
    else:
        w = sd.get("classifier.3.weight", None) or sd.get("classifier.1.weight", None)
        pv_num = int(w.shape[0]); pv_classes = [str(i) for i in range(pv_num)]

    m.classifier[3] = nn.Linear(in_features, pv_num)
    m.load_state_dict(sd, strict=True)

    new_head = nn.Linear(in_features, num_u)
    nn.init.normal_(new_head.weight, std=0.01); nn.init.zeros_(new_head.bias)

    pv_name2idx = {c:i for i,c in enumerate(pv_classes)}
    uni2pv_idx = {}
    for pv_name, u_name in pv_to_unified.items():
        if u_name in unified_classes and pv_name in pv_name2idx:
            uni2pv_idx[u_name] = pv_name2idx[pv_name]
    with torch.no_grad():
        for u_idx, u_name in enumerate(unified_classes):
            if u_name in uni2pv_idx:
                src = int(uni2pv_idx[u_name])
                new_head.weight[u_idx].copy_(m.classifier[3].weight[src])
                new_head.bias[u_idx].copy_(m.classifier[3].bias[src])

    m.classifier[3] = new_head
    m.to(device)
    return m


# ----------------- eval / train -----------------
@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        if tta:
            logits = model(xb) + model(torch.flip(xb, dims=[-1]))
            logits *= 0.5
        else:
            logits = model(xb)
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(yb.tolist())
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m

@torch.no_grad()
def bn_recalibrate(model, loader, device, max_batches=200):
    model.train()
    n = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        _ = model(xb)
        n += 1
        if n >= max_batches: break
    model.eval()


def train_one_epoch(model, loader, device, optim, criterion, mixup_alpha=0.2, max_grad_norm=3.0):
    model.train()
    tot, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)

        xb, mix = apply_mixup(xb, yb, alpha=mixup_alpha)
        optim.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = mixup_criterion(criterion, logits, mix, yb)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optim.step()

        loss_sum += float(loss.item())*yb.size(0)
        tot += yb.size(0)
        correct += int((logits.argmax(1)==yb).sum().item())
    return loss_sum/tot, correct/tot


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="mixed_split_plus.json (with source & indices)")
    ap.add_argument("--classmap", required=True)
    ap.add_argument("--init-from-pv", default=None, help="PV checkpoint to warm-start (optional)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pd-only-last", type=int, default=6)
    ap.add_argument("--pd-weight", type=float, default=3.0)
    ap.add_argument("--syn-weight", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=288)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--color-constancy", action="store_true")
    ap.add_argument("--leaf-crop", action="store_true")
    ap.add_argument("--no-strong-aug", action="store_true")
    ap.add_argument("--random-erasing-p", type=float, default=0.25)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--bn-recalibrate", action="store_true")
    ap.add_argument("--out-dir", default="runs/mixed_strong")
    args = ap.parse_args()

    set_seed(args.seed)
    sp = json.load(open(args.split, "r"))
    cm = json.load(open(args.classmap, "r"))

    unified = sp.get("unified_classes", cm["unified_classes"])
    P, L, S = sp["paths"], sp["labels"], sp["source"]
    tr_idx, va_idx = sp["train_idx"], sp["val_idx"]
    pv_t_idx, pd_t_idx = sp["test_pv_idx"], sp["test_pd_idx"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets & loaders
    ds_tr = MixedDataset([P[i] for i in tr_idx], [L[i] for i in tr_idx],
                         img_size=args.img_size, normalize=args.normalize,
                         train=True, color_constancy=args.color_constancy, leaf_crop=args.leaf_crop,
                         strong_aug=not args.no_strong_aug, random_erasing_p=args.random_erasing_p)
    ds_va = MixedDataset([P[i] for i in va_idx], [L[i] for i in va_idx],
                         img_size=args.img_size, normalize=args.normalize, train=False)
    ds_pvte = MixedDataset([P[i] for i in pv_t_idx], [L[i] for i in pv_t_idx],
                           img_size=args.img_size, normalize=args.normalize, train=False)
    ds_pdte = MixedDataset([P[i] for i in pd_t_idx], [L[i] for i in pd_t_idx],
                           img_size=args.img_size, normalize=args.normalize, train=False)

    sampler_mixed = make_sampler([L[i] for i in tr_idx], [S[i] for i in tr_idx],
                                 pd_weight=args.pd_weight, syn_weight=args.syn_weight, class_weight_from="pd")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler_mixed, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_pv = DataLoader(ds_pvte, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_pd = DataLoader(ds_pdte, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # PD-only loader for tail
    pd_tr_idx = [i for i in tr_idx if S[i] == "pd"]
    ds_tr_pd = MixedDataset([P[i] for i in pd_tr_idx], [L[i] for i in pd_tr_idx],
                            img_size=args.img_size, normalize=args.normalize,
                            train=True, color_constancy=args.color_constancy, leaf_crop=args.leaf_crop,
                            strong_aug=not args.no_strong_aug, random_erasing_p=args.random_erasing_p)
    sampler_pd = make_sampler([L[i] for i in pd_tr_idx], ["pd"]*len(pd_tr_idx),
                              pd_weight=1.0, syn_weight=1.0, class_weight_from="pd")
    dl_tr_pd = DataLoader(ds_tr_pd, batch_size=args.batch_size, sampler=sampler_pd, num_workers=4, pin_memory=True)

    # model
    model = build_student(unified, args.init_from_pv, cm["plantvillage_to_unified"], device)
    head_params = list(model.classifier[3].parameters())
    backbone_params = [p for n,p in model.named_parameters() if not n.startswith("classifier.3")]

    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head}
    ], weight_decay=args.wd)

    # class-weighted CE (from PD train only) + label smoothing
    cw = effective_class_weights([L[i] for i in pd_tr_idx], len(unified), beta=0.999).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    # zero-shot baselines
    acc_pv0, f1_pv0 = evaluate(model, dl_pv, device, tta=args.tta)
    acc_pd0, f1_pd0 = evaluate(model, dl_pd, device, tta=args.tta)
    print(f"[zero-shot] PV acc={acc_pv0:.4f} F1={f1_pv0:.4f} | PD acc={acc_pd0:.4f} F1={f1_pd0:.4f}")

    best_f1 = -1.0
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "mobilenetv3small_mixed_best.pt"

    for ep in range(1, args.epochs+1):
        use_pd_only = (args.pd_only_last > 0 and ep > args.epochs - args.pd_only_last)
        tr_loader = dl_tr_pd if use_pd_only else dl_tr

        tr_loss, tr_acc = train_one_epoch(model, tr_loader, device, optim, criterion, mixup_alpha=args.mixup)
        va_acc, va_f1 = evaluate(model, dl_va, device, tta=False)
        phase = "PD-only" if use_pd_only else "mixed  "
        print(f"[{ep:02d}/{args.epochs}] {phase} loss={tr_loss:.4f} acc={tr_acc:.4f} | val acc={va_acc:.4f} f1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"model": model.state_dict(), "classes": unified}, best_path)
            print(f"[saved] {best_path}")

    # load best + optional BN recalibration on PD
    best = torch.load(best_path, map_location="cpu"); model.load_state_dict(best["model"])
    if args.bn_recalibrate:
        print("[info] BN recalibration on PD...")
        bn_recalibrate(model, dl_tr_pd, device, max_batches=200)

    acc_pv, f1_pv = evaluate(model, dl_pv, device, tta=args.tta)
    acc_pd, f1_pd = evaluate(model, dl_pd, device, tta=args.tta)
    print(f"[final] PV test  acc={acc_pv:.4f}  F1={f1_pv:.4f}")
    print(f"[final] PD test  acc={acc_pd:.4f}  F1={f1_pd:.4f}")
    print(f"[saved] best checkpoint → {best_path}")


if __name__ == "__main__":
    main()
