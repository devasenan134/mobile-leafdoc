import argparse, json, random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, f1_score


import numpy as np
from PIL import Image

class HSVLeafCrop:
    def __init__(self, min_frac=0.05, max_frac=0.95, pad=8):
        self.min_frac, self.max_frac, self.pad = min_frac, max_frac, pad

    def __call__(self, img: Image.Image):
        x = np.asarray(img.convert("RGB"))
        # RGB -> HSV
        x_f = x.astype(np.float32) / 255.0
        mx = x_f.max(axis=2); mn = x_f.min(axis=2)
        diff = mx - mn + 1e-6
        # Hue in [0,1]
        h = np.zeros_like(mx)
        r, g, b = x_f[...,0], x_f[...,1], x_f[...,2]
        h = np.where(mx==r, ((g-b)/diff) % 6, h)
        h = np.where(mx==g, ((b-r)/diff) + 2, h)
        h = np.where(mx==b, ((r-g)/diff) + 4, h)
        h = (h/6.0)
        s = np.where(mx==0, 0.0, diff/mx)
        v = mx

        # crude green/yellow mask (tweakable ranges)
        green = ((h > 0.17) & (h < 0.47)) & (s > 0.15) & (v > 0.15)
        yellow = ((h > 0.10) & (h <= 0.17)) & (s > 0.10) & (v > 0.15)
        mask = (green | yellow)

        ys, xs = np.where(mask)
        if ys.size == 0:
            return img  # no mask → keep original

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        # reject if mask too tiny or too huge
        H, W = mask.shape
        box_area = (y1 - y0 + 1) * (x1 - x0 + 1)
        frac = box_area / float(H*W)
        if not (self.min_frac <= frac <= self.max_frac):
            return img

        # pad and clip
        y0 = max(0, y0 - self.pad); y1 = min(H, y1 + self.pad)
        x0 = max(0, x0 - self.pad); x1 = min(W, x1 + self.pad)
        crop = x[y0:y1, x0:x1, :]
        return Image.fromarray(crop)


# ----------------- utils -----------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_split(p): return json.load(open(p, "r"))

# Simple Gray-World color constancy
class GrayWorld:
    def __call__(self, img: Image.Image):
        x = np.asarray(img).astype(np.float32)
        m = x.reshape(-1,3).mean(axis=0) + 1e-6
        scale = float(np.mean(m)) / m
        x = np.clip(x * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(x)


# ----------------- dataset -----------------
class SplitDataset(Dataset):
    def __init__(self, paths, labels, img_size=224, normalize=False, leaf_crop=False,
                 train=True, color_constancy=False, strong_aug=True,
                 random_erasing_p=0.25):
        t = []
        if leaf_crop:
            t += [HSVLeafCrop()]

        if color_constancy:
            t += [GrayWorld()]
        if train:
            if strong_aug:
                t += [
                    transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(0.2,0.2,0.2,0.1),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                    transforms.GaussianBlur(3),
                ]
            else:
                t += [transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                      transforms.RandomHorizontalFlip(0.5)]
        else:
            t += [transforms.Resize((img_size, img_size))]
        t += [transforms.ToTensor()]
        if normalize:
            t += [transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
        if train and random_erasing_p > 0:
            t += [transforms.RandomErasing(p=random_erasing_p, scale=(0.02, 0.2), ratio=(0.3, 3.3))]
        self.tf = transforms.Compose(t)
        self.paths = paths
        self.labels = labels

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(x)
        y = int(self.labels[i])
        return x, y


def make_domain_balanced_sampler(labels, sources, pd_weight=3.0, syn_weight=0.8, class_weight_from="pd"):
    """
    labels: list[int] global class ids
    sources: list[str] in {"pv","pd","syn"}
    - class weights computed on 'pd' only to avoid synth overpowering rare classes
    - domain multipliers: pv=1.0, pd=pd_weight, syn=syn_weight
    """
    # choose which subset to compute class weights from
    sel_idx = [i for i,s in enumerate(sources) if (class_weight_from=="all") or (s=="pd")]
    counts = defaultdict(int)
    for i in sel_idx:
        counts[int(labels[i])] += 1
    # smooth in case some class absent
    for c in set(labels):
        if c not in counts: counts[c] = 1

    total = sum(counts.values()); C = len(counts)
    class_w = {c: total/(C*n) for c,n in counts.items()}  # inverse freq, normalized

    w = []
    for y,s in zip(labels, sources):
        base = class_w.get(int(y), 1.0)
        mult = 1.0
        if s == "pd":  mult = pd_weight
        elif s == "syn": mult = syn_weight
        w.append(base * mult)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.float), num_samples=len(labels), replacement=True)


# ----------------- model -----------------
def load_pv_init(pt_path, unified_classes, pv_to_unified, device):
    ckpt = torch.load(pt_path, map_location="cpu")
    sd = ckpt["model"]; pv_classes = ckpt.get("classes", None)

    if pv_classes and len(pv_classes)>0:
        pv_num = len(pv_classes)
    else:
        w = sd.get("classifier.3.weight", None) or sd.get("classifier.1.weight", None)
        pv_num = int(w.shape[0]); pv_classes = [str(i) for i in range(pv_num)]

    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, pv_num)
    m.load_state_dict(sd, strict=True)

    num_u = len(unified_classes)
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
    m._unified_classes = unified_classes
    return m


# ----------------- train / eval -----------------
@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        if not tta:
            logits = model(xb)
        else:
            # simple TTA: average logits with horizontal flip
            logits = model(xb)
            logits += model(torch.flip(xb, dims=[-1]))
            logits *= 0.5
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        y_true.extend(yb.numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m

def train_one_epoch(model, loader, device, optim, criterion):
    model.train()
    tot, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward(); optim.step()
        loss_sum += float(loss.item())*yb.size(0)
        tot += yb.size(0); correct += int((logits.argmax(1)==yb).sum().item())
    return loss_sum/tot, correct/tot

def save_ckpt(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": getattr(model, "_unified_classes", None)}, path)
    print(f"[saved] {path}")


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed-split", required=True, help="mixed_split.json")
    ap.add_argument("--classmap", required=True)
    ap.add_argument("--init-from-pv", required=True, help="PV checkpoint (.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pd-only-last", type=int, default=3, help="Last N epochs sample PD-only")
    ap.add_argument("--pd-weight", type=float, default=2.0, help="Reweight PD in sampler during mixed phase")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--color-constancy", action="store_true")
    ap.add_argument("--leaf-crop", action="store_true")
    ap.add_argument("--no-strong-aug", action="store_true")
    ap.add_argument("--random-erasing-p", type=float, default=0.25)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out-dir", default="runs/mixed")
    args = ap.parse_args()

    set_seed(args.seed)
    sp = load_split(args.mixed_split)
    cm = json.load(open(args.classmap, "r"))
    unified = cm["unified_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    P, L, S = sp["paths"], sp["labels"], sp["source"]
    tr_idx, va_idx = sp["train_idx"], sp["val_idx"]
    pv_t_idx, pd_t_idx = sp["test_pv_idx"], sp["test_pd_idx"]

    # datasets
    ds_tr = SplitDataset([P[i] for i in tr_idx], [L[i] for i in tr_idx],
                         img_size=args.img_size, normalize=args.normalize,
                         train=True, color_constancy=args.color_constancy,
                         strong_aug=not args.no_strong_aug, random_erasing_p=args.random_erasing_p)
    ds_va = SplitDataset([P[i] for i in va_idx], [L[i] for i in va_idx],
                         img_size=args.img_size, normalize=args.normalize, train=False)
    ds_pvte = SplitDataset([P[i] for i in pv_t_idx], [L[i] for i in pv_t_idx],
                           img_size=args.img_size, normalize=args.normalize, train=False)
    ds_pdte = SplitDataset([P[i] for i in pd_t_idx], [L[i] for i in pd_t_idx],
                           img_size=args.img_size, normalize=args.normalize, train=False)

    # samplers
    sampler_mixed = make_domain_balanced_sampler([L[i] for i in tr_idx], [S[i] for i in tr_idx],
                                       pd_weight=3.0, syn_weight=0.8, class_weight_from="pd")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler_mixed, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_pvte = DataLoader(ds_pvte, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_pdte = DataLoader(ds_pdte, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # PD-only loader for tail phase
    pd_tr_idx = [i for i in tr_idx if S[i] == "pd"]
    ds_tr_pd = SplitDataset([P[i] for i in pd_tr_idx], [L[i] for i in pd_tr_idx],
                            img_size=args.img_size, normalize=args.normalize,
                            train=True, color_constancy=args.color_constancy,
                            strong_aug=not args.no_strong_aug, random_erasing_p=args.random_erasing_p)
    sampler_pd = make_domain_balanced_sampler([L[i] for i in pd_tr_idx], ["pd"]*len(pd_tr_idx), pd_weight=1.0)
    dl_tr_pd = DataLoader(ds_tr_pd, batch_size=args.batch_size, sampler=sampler_pd, num_workers=4, pin_memory=True)

    # model
    model = load_pv_init(args.init_from_pv, unified, cm["plantvillage_to_unified"], device)

    head_params = list(model.classifier[3].parameters())
    backbone_params = [p for n,p in model.named_parameters() if not n.startswith("classifier.3")]
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head}
    ], weight_decay=args.wd)

    # label smoothing helps OOD a bit
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # zero-shot baselines
    acc_pv0, f1_pv0 = evaluate(model, dl_pvte, device, tta=args.tta)
    acc_pd0, f1_pd0 = evaluate(model, dl_pdte, device, tta=args.tta)
    print(f"[zero-shot] PV acc={acc_pv0:.4f} F1={f1_pv0:.4f} | PD acc={acc_pd0:.4f} F1={f1_pd0:.4f}")

    best_f1 = -1.0
    best_path = Path(args.out_dir)/"mobilenetv3small_mixed_best.pt"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs+1):
        # last N epochs: PD-only; otherwise mixed
        use_pd_only = (args.pd_only_last > 0 and ep > args.epochs - args.pd_only_last)
        tr_loader = dl_tr_pd if use_pd_only else dl_tr

        tr_loss, tr_acc = train_one_epoch(model, tr_loader, device, optim, criterion)
        va_acc, va_f1 = evaluate(model, dl_va, device)  # val is mixed
        print(f"[{ep:02d}/{args.epochs}] {'PD-only' if use_pd_only else 'mixed ':6s} "
              f"loss={tr_loss:.4f} acc={tr_acc:.4f} | val acc={va_acc:.4f} f1={va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            save_ckpt(model, best_path)

    # Final test
    best = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best["model"])

    acc_pv, f1_pv = evaluate(model, dl_pvte, device, tta=args.tta)
    acc_pd, f1_pd = evaluate(model, dl_pdte, device, tta=args.tta)
    print(f"[final] PV test  acc={acc_pv:.4f}  F1={f1_pv:.4f}")
    print(f"[final] PD test  acc={acc_pd:.4f}  F1={f1_pd:.4f}")
    print(f"[saved] best checkpoint → {best_path}")


if __name__ == "__main__":
    main()
