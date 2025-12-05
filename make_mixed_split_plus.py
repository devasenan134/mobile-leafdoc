import argparse, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

EXTS={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def load_cm(p):
    cm=json.load(open(p))
    for k in ["unified_classes","plantvillage_to_unified","plantdoc_to_unified"]:
        if k not in cm: raise SystemExit("classmap missing keys")
    return cm

def scan_dir(root, folder2unified, u2i):
    P,L=[],[]
    for d in sorted([x for x in Path(root).iterdir() if x.is_dir()]):
        key=d.name
        if key not in folder2unified: continue
        u=folder2unified[key]
        if u not in u2i: continue
        y=u2i[u]
        for im in d.rglob("*"):
            if im.suffix.lower() in EXTS: P.append(str(im)); L.append(y)
    return P,L

def scan_unified_folder(root, u2i):
    P,L=[],[]
    for d in sorted([x for x in Path(root).iterdir() if x.is_dir()]):
        u=d.name
        if u not in u2i: continue
        y=u2i[u]
        for im in d.rglob("*"):
            if im.suffix.lower() in EXTS: P.append(str(im)); L.append(y)
    return P,L

def stratify(P,L,ratio,seed):
    y=np.array(L); idx=np.arange(len(P))
    sss=StratifiedShuffleSplit(n_splits=1,test_size=ratio,random_state=seed)
    tr,te=next(sss.split(idx,y)); return tr.tolist(), te.tolist()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--classmap",required=True)
    ap.add_argument("--pv-dir",required=True)
    ap.add_argument("--pd-train-dir",required=True)
    ap.add_argument("--pd-test-dir",required=True)
    ap.add_argument("--extra-train-dir", default=None, help="e.g., data_fda_pd")
    ap.add_argument("--pv-test-ratio",type=float,default=0.1)
    ap.add_argument("--pv-val-ratio",type=float,default=0.1)
    ap.add_argument("--pd-val-ratio",type=float,default=0.1)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="mixed_split_plus.json")
    args=ap.parse_args()

    cm=load_cm(args.classmap)
    unified=cm["unified_classes"]; u2i={u:i for i,u in enumerate(unified)}

    pvP,pvL = scan_dir(args.pv_dir, cm["plantvillage_to_unified"], u2i)
    pdtrP,pdtrL = scan_dir(args.pd_train_dir, cm["plantdoc_to_unified"], u2i)
    pdteP,pdteL = scan_dir(args.pd_test_dir,  cm["plantdoc_to_unified"], u2i)

    if not pvP or not pdtrP or not pdteP: raise SystemExit("missing data; check roots/classmap")

    pv_tr, pv_te = stratify(pvP, pvL, args.pv_test_ratio, args.seed)
    pv_trP=[pvP[i] for i in pv_tr]; pv_trL=[pvL[i] for i in pv_tr]
    pv_subtr, pv_va = stratify(pv_trP, pv_trL, args.pv_val_ratio, args.seed)
    pv_va = [pv_tr[i] for i in pv_va]; pv_tr = [pv_tr[i] for i in pv_subtr]

    pd_tr, pd_va = stratify(pdtrP, pdtrL, args.pd_val_ratio, args.seed)
    pd_te = list(range(len(pdteP)))

    P = pvP + pdtrP + pdteP
    L = pvL + pdtrL + pdteL
    S = (["pv"]*len(pvP)) + (["pd"]*len(pdtrP)) + (["pd"]*len(pdteP))

    # extra synthetic (unified folder structure)
    if args.extra_train_dir:
        exP, exL = scan_unified_folder(args.extra_train_dir, u2i)
        P += exP; L += exL; S += ["syn"]*len(exP)
        ex_base = len(pvP) + len(pdtrP) + len(pdteP)
        ex_idx = list(range(ex_base, ex_base+len(exP)))
    else:
        ex_idx = []

    base_pv = 0
    base_pdtr = len(pvP)
    base_pdte = base_pdtr + len(pdtrP)

    def remap(idxs, base): return [base+i for i in idxs]

    split = {
        "unified_classes": unified,
        "paths": P, "labels": L, "source": S,
        "train_idx": remap(pv_tr, base_pv) + remap(pd_tr, base_pdtr) + ex_idx,
        "val_idx": remap(pv_va, base_pv) + remap(pd_va, base_pdtr),
        "test_pv_idx": remap(pv_te, base_pv),
        "test_pd_idx": remap(pd_te, base_pdte),
        "img_size": 224, "normalized": False
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(split, open(args.out,"w"), indent=2)
    print(f"[saved] {args.out}")
    print(f"train: {len(split['train_idx'])} (pv {len(pv_tr)} | pd {len(pd_tr)} | syn {len(ex_idx)})")
    print(f"val:   {len(split['val_idx'])}   pv {len(pv_va)} | pd {len(pd_va)}")
    print(f"test:  pv {len(split['test_pv_idx'])} | pd {len(split['test_pd_idx'])}")
if __name__ == "__main__":
    main()
