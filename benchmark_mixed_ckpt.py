# eval_mixed_ckpt.py
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score


# ---------- optional eval prepro to match training ----------
class GrayWorld:
    def __call__(self, img: Image.Image):
        x = np.asarray(img).astype(np.float32)
        m = x.reshape(-1,3).mean(axis=0) + 1e-6
        x = np.clip(x * (float(np.mean(m))/m), 0, 255).astype(np.uint8)
        return Image.fromarray(x)

class HSVLeafCrop:
    def __init__(self, min_frac=0.03, max_frac=0.97, pad=8):
        self.min_frac, self.max_frac, self.pad = min_frac, max_frac, pad
    def __call__(self, img: Image.Image):
        x = np.asarray(img.convert("RGB"))
        xf = x.astype(np.float32)/255.0
        mx, mn = xf.max(2), xf.min(2); diff = mx - mn + 1e-6
        h = np.zeros_like(mx)
        r,g,b = xf[...,0], xf[...,1], xf[...,2]
        h = np.where(mx==r, ((g-b)/diff)%6, h)
        h = np.where(mx==g, ((b-r)/diff)+2, h)
        h = np.where(mx==b, ((r-g)/diff)+4, h)
        h = h/6.0; s = np.where(mx==0, 0.0, diff/mx); v = mx
        mask = (((h>0.17)&(h<0.47)&(s>0.15)&(v>0.15)) | ((h>0.10)&(h<=0.17)&(s>0.10)&(v>0.15)))
        ys, xs = np.where(mask)
        if ys.size==0: return img
        y0,y1,x0,x1 = ys.min(), ys.max(), xs.min(), xs.max()
        H,W = mask.shape
        frac = (y1-y0+1)*(x1-x0+1)/(H*W)
        if not (self.min_frac <= frac <= self.max_frac): return img
        y0=max(0,y0-self.pad); y1=min(H,y1+self.pad)
        x0=max(0,x0-self.pad); x1=min(W,x1+self.pad)
        return Image.fromarray(x[y0:y1, x0:x1, :])


# ---------- dataset ----------
class EvalDataset(Dataset):
    def __init__(self, paths, labels, img_size=288, normalize=False, color_constancy=False, leaf_crop=False):
        t=[]
        if leaf_crop:       t += [HSVLeafCrop()]
        if color_constancy: t += [GrayWorld()]
        t += [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        if normalize:
            t += [transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        self.tf = transforms.Compose(t)
        self.P = paths; self.L = labels
    def __len__(self): return len(self.P)
    def __getitem__(self, i):
        x = Image.open(self.P[i]).convert("RGB")
        return self.tf(x), int(self.L[i])


# ---------- model loader ----------
def load_model(pt_path, device):
    ckpt = torch.load(pt_path, map_location="cpu")
    classes = ckpt.get("classes", None)
    sd = ckpt["model"]
    if classes is None:
        # infer num classes from head
        w = sd.get("classifier.3.weight", None) or sd.get("classifier.1.weight", None)
        if w is None: raise RuntimeError("Cannot infer head size.")
        classes = [str(i) for i in range(int(w.shape[0]))]
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, len(classes))
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()
    return m, classes


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="checkpoint .pt")
    ap.add_argument("--split", required=True, help="mixed_split_plus.json")
    ap.add_argument("--img-size", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--color-constancy", action="store_true")
    ap.add_argument("--leaf-crop", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out-json", default="", help="optional: save metrics to JSON")
    args = ap.parse_args()

    sp = json.load(open(args.split, "r"))
    P, L = sp["paths"], sp["labels"]
    tr_idx, va_idx = sp["train_idx"], sp["val_idx"]
    pv_t_idx, pd_t_idx = sp["test_pv_idx"], sp["test_pd_idx"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.pt, device)

    # datasets/loaders (eval-time transforms only)
    def mk_loader(indices):
        ds = EvalDataset([P[i] for i in indices],
                         [L[i] for i in indices],
                         img_size=args.img_size,
                         normalize=args.normalize,
                         color_constancy=args.color_constancy,
                         leaf_crop=args.leaf_crop)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    dl_tr  = mk_loader(tr_idx)
    dl_val = mk_loader(va_idx)
    dl_pv  = mk_loader(pv_t_idx)
    dl_pd  = mk_loader(pd_t_idx)
    dl_all = mk_loader(pv_t_idx + pd_t_idx)

    # eval
    train_acc, train_f1 = evaluate(model, dl_tr, device, tta=args.tta)
    val_acc,   val_f1   = evaluate(model, dl_val, device, tta=args.tta)
    pv_acc,    pv_f1    = evaluate(model, dl_pv,  device, tta=args.tta)
    pd_acc,    pd_f1    = evaluate(model, dl_pd,  device, tta=args.tta)
    all_acc,   all_f1   = evaluate(model, dl_all, device, tta=args.tta)

    print(f"TRAIN   acc={train_acc:.4f}  macro-F1={train_f1:.4f}")
    print(f"VAL     acc={val_acc:.4f}    macro-F1={val_f1:.4f}")
    print(f"TEST PV acc={pv_acc:.4f}    macro-F1={pv_f1:.4f}")
    print(f"TEST PD acc={pd_acc:.4f}    macro-F1={pd_f1:.4f}")
    print(f"TEST ALL acc={all_acc:.4f}  macro-F1={all_f1:.4f}")

    if args.out_json:
        out = {
            "train": {"acc": train_acc, "macro_f1": train_f1},
            "val":   {"acc": val_acc,   "macro_f1": val_f1},
            "test_pv": {"acc": pv_acc,  "macro_f1": pv_f1},
            "test_pd": {"acc": pd_acc,  "macro_f1": pd_f1},
            "test_all": {"acc": all_acc,"macro_f1": all_f1},
            "img_size": args.img_size,
            "normalize": args.normalize,
            "leaf_crop": args.leaf_crop,
            "color_constancy": args.color_constancy,
            "tta": args.tta,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out_json, "w"), indent=2)
        print(f"[saved] {args.out_json}")

if __name__ == "__main__":
    main()
