# train_mixed.py
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

# --------- utils ----------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_split(p):
    sp = json.load(open(p, "r"))
    return sp

# Simple Gray-World color constancy (optional)
class GrayWorld:
    def __call__(self, img: Image.Image):
        x = np.asarray(img).astype(np.float32)
        eps = 1e-6
        m = x.reshape(-1,3).mean(axis=0) + eps
        gm = float(np.mean(m))
        scale = gm / m
        x = np.clip(x * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(x)

# --------- dataset ----------
class SplitDataset(Dataset):
    def __init__(self, paths, labels, img_size=224, normalize=False, train=True, color_constancy=False, strong_aug=True):
        t = []
        if color_constancy:
            t += [GrayWorld()]
        if train:
            if strong_aug:
                t += [
                    transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.1)], p=0.8),
                    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.2),
                    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
                ]
            else:
                t += [transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                      transforms.RandomHorizontalFlip(0.5)]
        else:
            t += [transforms.Resize((img_size, img_size))]
        t += [transforms.ToTensor()]
        if normalize:
            t += [transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
        self.tf = transforms.Compose(t)
        self.paths = paths
        self.labels = labels

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        x = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(x)
        y = int(self.labels[i])
        return x, y

def make_balanced_sampler(labels):
    counts = defaultdict(int)
    for y in labels: counts[int(y)] += 1
    total = sum(counts.values()); C = len(counts)
    class_w = {c: total/(C*cnt) for c, cnt in counts.items()}
    weights = torch.tensor([class_w[int(y)] for y in labels], dtype=torch.float)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

# --------- model ----------
def load_pv_init(pt_path, unified_classes, pv_to_unified, device):
    ckpt = torch.load(pt_path, map_location="cpu")
    sd = ckpt["model"]; pv_classes = ckpt.get("classes", None)

    # infer PV head size
    if pv_classes and len(pv_classes)>0:
        pv_num = len(pv_classes)
    else:
        w = sd.get("classifier.3.weight", None) or sd.get("classifier.1.weight", None)
        pv_num = int(w.shape[0])

    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, pv_num)
    m.load_state_dict(sd, strict=True)

    # new 28-unified head
    num_u = len(unified_classes)
    new_head = nn.Linear(in_features, num_u)
    nn.init.normal_(new_head.weight, std=0.01); nn.init.zeros_(new_head.bias)

    # copy matching rows PV->unified
    if pv_classes:
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

# --------- train/eval ----------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed-split", required=True, help="mixed_split.json from make_mixed_split.py")
    ap.add_argument("--classmap", required=True)
    ap.add_argument("--init-from-pv", required=True, help="PV checkpoint (.pt) to initialize from")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--normalize", action="store_true", help="Use ImageNet mean/std if baseline used it")
    ap.add_argument("--color-constancy", action="store_true")
    ap.add_argument("--no-strong-aug", action="store_true")
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="runs/mixed")
    args = ap.parse_args()

    set_seed(args.seed)
    sp = load_split(args.mixed_split)
    cm = json.load(open(args.classmap, "r"))
    unified = cm["unified_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets from split
    P = sp["paths"]; L = sp["labels"]
    tr_idx, va_idx = sp["train_idx"], sp["val_idx"]
    pv_t_idx, pd_t_idx = sp["test_pv_idx"], sp["test_pd_idx"]

    ds_tr = SplitDataset([P[i] for i in tr_idx], [L[i] for i in tr_idx],
                         img_size=args.img_size, normalize=args.normalize,
                         train=True, color_constancy=args.color_constancy,
                         strong_aug=not args.no_strong_aug)
    ds_va = SplitDataset([P[i] for i in va_idx], [L[i] for i in va_idx],
                         img_size=args.img_size, normalize=args.normalize, train=False)
    ds_pvte = SplitDataset([P[i] for i in pv_t_idx], [L[i] for i in pv_t_idx],
                           img_size=args.img_size, normalize=args.normalize, train=False)
    ds_pdte = SplitDataset([P[i] for i in pd_t_idx], [L[i] for i in pd_t_idx],
                           img_size=args.img_size, normalize=args.normalize, train=False)

    sampler = make_balanced_sampler([L[i] for i in tr_idx])
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_pvte = DataLoader(ds_pvte, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_pdte = DataLoader(ds_pdte, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model init from PV
    model = load_pv_init(args.init_from_pv, unified, cm["plantvillage_to_unified"], device)

    head_params = list(model.classifier[3].parameters())
    backbone_params = [p for n,p in model.named_parameters() if not n.startswith("classifier.3")]
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head}
    ], weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    # Baselines before training
    acc_pv0, f1_pv0 = evaluate(model, dl_pvte, device)
    acc_pd0, f1_pd0 = evaluate(model, dl_pdte, device)
    print(f"[zero-shot] PV test acc={acc_pv0:.4f} F1={f1_pv0:.4f} | PD test acc={acc_pd0:.4f} F1={f1_pd0:.4f}")

    best_f1 = -1.0
    best_path = Path(args.out_dir)/"mobilenetv3small_mixed_best.pt"

    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, device, optim, criterion)
        va_acc, va_f1 = evaluate(model, dl_va, device)
        print(f"[{ep:02d}/{args.epochs}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val acc={va_acc:.4f} f1={va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            save_ckpt(model, best_path)

    # Final test with best model
    best = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best["model"])

    acc_pv, f1_pv = evaluate(model, dl_pvte, device)
    acc_pd, f1_pd = evaluate(model, dl_pdte, device)
    print(f"[final] PV test  acc={acc_pv:.4f}  F1={f1_pv:.4f}")
    print(f"[final] PD test  acc={acc_pd:.4f}  F1={f1_pd:.4f}")
    print(f"[saved] best checkpoint → {best_path}")

if __name__ == "__main__":
    main()
