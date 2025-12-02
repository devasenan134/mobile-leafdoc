import argparse, json, os, math
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- data ----------
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

class PDTestDataset(Dataset):
    def __init__(self, root_dir, classmap, img_size=224, normalize=False):
        self.paths, self.labels, self.u_classes = [], [], classmap["unified_classes"]
        u2i = {u:i for i,u in enumerate(self.u_classes)}
        pd_map = classmap["plantdoc_to_unified"]
        for d in sorted([p for p in Path(root_dir).iterdir() if p.is_dir()]):
            pd_folder = d.name
            if pd_folder not in pd_map: 
                continue
            u = pd_map[pd_folder]
            if u not in u2i: 
                continue
            y = u2i[u]
            for img in d.rglob("*"):
                if img.suffix.lower() in IMG_EXTS:
                    self.paths.append(str(img)); self.labels.append(y)

        tf = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        if normalize:
            tf += [transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        self.tf = transforms.Compose(tf)

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(x)
        return x, int(self.labels[i]), self.paths[i]


# ---------- model ----------
def load_model_ckpt(pt_path, device):
    ckpt = torch.load(pt_path, map_location="cpu")
    sd = ckpt["model"]
    classes = ckpt.get("classes", None)

    # infer num classes if needed
    if isinstance(classes, list) and len(classes) > 0:
        num_classes = len(classes)
    else:
        w = sd.get("classifier.3.weight", None) or sd.get("classifier.1.weight", None)
        if w is None:
            raise RuntimeError("Cannot infer head size from checkpoint.")
        num_classes = int(w.shape[0])
        classes = [str(i) for i in range(num_classes)]

    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, num_classes)
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()
    return m, classes


# ---------- utils ----------
@torch.no_grad()
def predict_logits(model, loader, device, idx_map=None, topk=5):
    all_logits, all_y, all_paths = [], [], []
    for xb, yb, pb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)  # [B, C_ckpt]
        if idx_map is not None:
            logits = logits[:, idx_map]  # reorder to unified order
        all_logits.append(logits.cpu())
        all_y.append(yb)
        all_paths += list(pb)
    logits = torch.cat(all_logits, 0).float()
    y_true = torch.cat(all_y, 0).numpy()
    probs = torch.softmax(logits, 1).numpy()
    y_pred = probs.argmax(1)
    # top-k hit
    topk_idx = np.argsort(-probs, axis=1)[:, :topk]
    topk_hit = np.array([t in row for t, row in zip(y_true, topk_idx)], dtype=bool)
    return y_true, y_pred, probs, topk_idx, topk_hit, all_paths


def plot_confusion(cm_norm, class_names, out_png, max_classes=28):
    fig = plt.figure(figsize=(max(8, max_classes*0.35), max(8, max_classes*0.35)))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (row=GT, col=Pred)')
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    plt.tight_layout()
    plt.ylabel('GT'); plt.xlabel('Pred')
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def top_confusions(cm, class_names, k=3):
    out = []
    for i, row in enumerate(cm):
        # zero out diagonal & get top confusions
        r = row.copy(); r[i] = 0
        if r.sum() == 0:
            out.append((class_names[i], [])); continue
        j_sorted = np.argsort(-r)[:k]
        pairs = [(class_names[j], int(r[j])) for j in j_sorted if r[j] > 0]
        out.append((class_names[i], pairs))
    return out


def save_errors_grid(paths, preds, gts, class_names, out_png, cols=5, max_imgs=25):
    sel = list(range(min(len(paths), max_imgs)))
    n = len(sel); rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2))
    axes = axes.ravel()
    for k in range(rows*cols):
        ax = axes[k]
        ax.axis("off")
        if k >= n: continue
        pth = paths[sel[k]]
        try:
            img = Image.open(pth).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"{class_names[gts[sel[k]]]} → {class_names[preds[sel[k]]]}", fontsize=7)
        except Exception as e:
            ax.text(0.1, 0.5, f"err: {e}", fontsize=6)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="checkpoint to evaluate")
    ap.add_argument("--classmap", required=True, help="overlap-only classmap.json")
    ap.add_argument("--pd-test-dir", required=True, help="PlantDoc test root")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="optional: evaluate only first N images")
    ap.add_argument("--out-dir", default="analysis_pd")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm = json.load(open(args.classmap, "r"))
    u_classes = cm["unified_classes"]

    # data
    dset = PDTestDataset(args.pd_test_dir, cm, img_size=args.img_size, normalize=args.normalize)
    if args.limit and args.limit > 0:
        dset.paths = dset.paths[:args.limit]
        dset.labels = dset.labels[:args.limit]
    dl = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    model, ckpt_classes = load_model_ckpt(args.pt, device)

    # map checkpoint class order -> unified order (columns)
    u2i = {u:i for i,u in enumerate(u_classes)}
    c2i = {c:i for i,c in enumerate(ckpt_classes)}
    idx_map = []
    missing = []
    for u in u_classes:
        if u in c2i:
            idx_map.append(c2i[u])
        else:
            missing.append(u)
    if missing:
        raise SystemExit(f"Checkpoint class list missing unified classes: {missing}")
    idx_map = torch.tensor(idx_map, dtype=torch.long)

    # run
    y_true, y_pred, probs, topk_idx, topk_hit, paths = predict_logits(model, dl, device, idx_map=idx_map, topk=args.topk)

    # ---- metrics (use full label list) ----
    labels_all = list(range(len(u_classes)))  # 0..27 for your 28 unified classes
    present = set(np.unique(y_true))
    absent  = [u_classes[i] for i in labels_all if i not in present]
    if absent:
        print("Absent classes in PD test set:", absent)


    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0)
    topk_acc = float(topk_hit.mean())

    # per-class report using fixed labels/target_names
    report = classification_report(
        y_true, y_pred,
        labels=labels_all,
        target_names=u_classes,
        output_dict=True,
        zero_division=0
    )

    # per-class accuracy (guard when class absent)
    per_class_acc = {}
    y_true_arr = np.asarray(y_true)
    for c in labels_all:
        mask = (y_true_arr == c)
        per_class_acc[u_classes[c]] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")

    # confusion matrix (use same labels list)
    cm_raw  = confusion_matrix(y_true, y_pred, labels=labels_all)
    cm_norm = cm_raw.astype(np.float32) / np.maximum(cm_raw.sum(1, keepdims=True), 1)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm_norm, u_classes, out_dir / "confusion_pd_norm.png")

    # top confusions per class
    conf_pairs = top_confusions(cm_raw, u_classes, k=3)

    # save CSV
    import csv
    csv_path = out_dir / "predictions_pd.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","gt","pred","correct","topk_hit","p_pred","topk_labels","topk_probs"])
        for i, (p, yt, yp, hit) in enumerate(zip(paths, y_true, y_pred, topk_hit)):
            tk_idx = topk_idx[i]
            w.writerow([
                p,
                u_classes[yt],
                u_classes[yp],
                int(yp == yt),
                int(hit),
                f"{probs[i, yp]:.6f}",
                "|".join(u_classes[j] for j in tk_idx),
                "|".join(f"{probs[i, j]:.4f}" for j in tk_idx),
            ])


    # hardest classes (by F1 then acc)
    per_class = []
    for cname in u_classes:
        r = report.get(cname, {})
        f1c = float(r.get("f1-score", 0.0))
        prec = float(r.get("precision", 0.0))
        rec = float(r.get("recall", 0.0))
        acc_c = per_class_acc[cname]
        per_class.append((cname, f1c, acc_c, prec, rec))
    per_class.sort(key=lambda x: (x[1], x[2]))  # lowest F1 then acc

    # save summary JSON
    summary = {
        "overall": {"top1_acc": acc, "macro_f1": f1m, "topk_acc": topk_acc},
        "per_class": [
            {"class": n, "f1": f1, "acc": accc, "precision": p, "recall": r}
            for (n, f1, accc, p, r) in per_class
        ],
        "top_confusions": [
            {"gt": gt, "confused_with": [{"pred": p, "count": c} for (p,c) in pairs]}
            for (gt, pairs) in conf_pairs
        ]
    }
    json.dump(summary, open(out_dir/"summary_pd.json","w"), indent=2)

    # error grids for the 5 hardest classes
    hardest = [n for (n, *_rest) in per_class[:5]]
    name2i = {u:i for i,u in enumerate(u_classes)}
    for cname in hardest:
        ci = name2i[cname]
        err_idx = np.where((y_true == ci) & (y_pred != ci))[0]
        if err_idx.size == 0: 
            continue
        sel_paths = [paths[i] for i in err_idx[:25]]
        sel_preds = [y_pred[i] for i in err_idx[:25]]
        sel_gts = [y_true[i] for i in err_idx[:25]]
        save_errors_grid(sel_paths, sel_preds, sel_gts, u_classes, out_dir/f"errors_{cname}.png")

    # console report
    print(f"PD test: top1_acc={acc:.4f}  macroF1={f1m:.4f}  top{args.topk}_acc={topk_acc:.4f}")
    print(f"[saved] CSV: {csv_path}")
    print(f"[saved] confusion: {out_dir/'confusion_pd_norm.png'}")
    print(f"[saved] summary: {out_dir/'summary_pd.json'}")
    print("Hardest classes (by F1 then acc):")
    for (n,f1,ac,pr,re) in per_class[:10]:
        print(f"  - {n:30s}  F1={f1:.3f}  Acc={ac:.3f}  Prec={pr:.3f}  Rec={re:.3f}")

if __name__ == "__main__":
    main()
