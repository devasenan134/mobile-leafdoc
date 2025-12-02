# make_mixed_split.py
import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def load_classmap(p):
    cm = json.load(open(p, "r"))
    for k in ["unified_classes","plantvillage_to_unified","plantdoc_to_unified"]:
        if k not in cm: raise SystemExit(f"classmap missing key: {k}")
    return cm

def scan_dir(root: Path, folder2unified: dict, unified2idx: dict):
    files, labels = [], []
    for cls_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        key = cls_dir.name
        if key not in folder2unified: 
            continue
        u = folder2unified[key]
        if u not in unified2idx:
            continue
        y = unified2idx[u]
        for img in cls_dir.rglob("*"):
            if img.suffix.lower() in IMG_EXTS:
                files.append(str(img))
                labels.append(y)
    return files, labels

def stratify(files, labels, test_ratio, seed=42):
    y = np.array(labels)
    idx = np.arange(len(files))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    tr_idx, te_idx = next(sss.split(idx, y))
    return (tr_idx.tolist(), te_idx.tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classmap", required=True)
    ap.add_argument("--pv-dir", required=True, help="PlantVillage root (e.g., PlantVillage-Dataset/raw/color)")
    ap.add_argument("--pd-train-dir", required=True, help="PlantDoc train root")
    ap.add_argument("--pd-test-dir", required=True, help="PlantDoc test root")
    ap.add_argument("--pv-test-ratio", type=float, default=0.1, help="If you don't already have a PV split, we’ll carve 10% as PV test")
    ap.add_argument("--pv-val-ratio", type=float, default=0.1, help="From PV train portion, carve 10% for val")
    ap.add_argument("--pd-val-ratio", type=float, default=0.1, help="From PD train, carve 10% for val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="mixed_split.json")
    args = ap.parse_args()

    cm = load_classmap(args.classmap)
    unified = cm["unified_classes"]; unified2idx = {u:i for i,u in enumerate(unified)}

    pv_root = Path(args.pv-dir if hasattr(args, "pv-dir") else args.pv_dir)  # in case shell hyphen fix
    pd_tr = Path(args.pd_train_dir); pd_te = Path(args.pd_test_dir)

    # Scan PV and PD
    pv_files, pv_labels = scan_dir(pv_root, cm["plantvillage_to_unified"], unified2idx)
    pdtr_files, pdtr_labels = scan_dir(pd_tr, cm["plantdoc_to_unified"], unified2idx)
    pdte_files, pdte_labels = scan_dir(pd_te, cm["plantdoc_to_unified"], unified2idx)

    if not pv_files: raise SystemExit("No PlantVillage files found (overlap-only).")
    if not pdtr_files or not pdte_files: raise SystemExit("No PlantDoc files found (train/test).")

    # Split PV into train/val/test
    pv_tr_idx, pv_te_idx = stratify(pv_files, pv_labels, args.pv_test_ratio, seed=args.seed)
    # Now split pv_tr_idx further into train/val
    pv_tr_files = [pv_files[i] for i in pv_tr_idx]; pv_tr_labels = [pv_labels[i] for i in pv_tr_idx]
    sub_tr, sub_va = stratify(pv_tr_files, pv_tr_labels, args.pv_val_ratio, seed=args.seed)
    pv_va_idx = [pv_tr_idx[i] for i in sub_va]
    pv_tr_idx = [pv_tr_idx[i] for i in sub_tr]

    # PD: split only its train dir into train/val, test stays as given
    pd_tr_idx, pd_va_idx = stratify(pdtr_files, pdtr_labels, args.pd_val_ratio, seed=args.seed)
    # PD test is fixed set
    pd_te_idx = list(range(len(pdte_files)))

    # Build unified arrays
    paths = []
    labels = []
    source = []

    # PV
    base_pv = 0
    paths += pv_files; labels += pv_labels; source += ["pv"]*len(pv_files)
    # PD train
    base_pdtr = len(paths)
    paths += pdtr_files; labels += pdtr_labels; source += ["pd"]*len(pdtr_files)
    # PD test
    base_pdte = len(paths)
    paths += pdte_files; labels += pdte_labels; source += ["pd"]*len(pdte_files)

    # Indices in the global arrays
    def remap(idxs, base): return [base+i for i in idxs]

    split = {
        "unified_classes": unified,
        "paths": paths,
        "labels": labels,
        "source": source,
        "train_idx": remap(pv_tr_idx, base_pv) + remap(pd_tr_idx, base_pv + len(pv_files)),
        "val_idx": remap(pv_va_idx, base_pv) + remap(pd_va_idx, base_pv + len(pv_files)),
        "test_pv_idx": remap(pv_te_idx, base_pv),
        "test_pd_idx": remap(pd_te_idx, base_pdte),
        "img_size": 224,
        "normalized": False
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(split, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}")
    print(f"Counts  PV train/val/test: {len(pv_tr_idx)}/{len(pv_va_idx)}/{len(pv_te_idx)} | PD train/val/test: {len(pd_tr_idx)}/{len(pd_va_idx)}/{len(pd_te_idx)}")

if __name__ == "__main__":
    main()
