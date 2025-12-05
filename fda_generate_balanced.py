# fda_generate_balanced.py
import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image

EXTS={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def imread(p): return np.asarray(Image.open(p).convert("RGB"))
def imsave(arr, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(p, quality=95)

def fda_source_to_target(src, tgt, beta=0.03):
    """
    Robust FDA:
    - resize tgt to src size,
    - clamp b >= 1 and within image bounds,
    - swap low-frequency amplitudes in the center square.
    """
    # ensure same HxW
    Hs, Ws = src.shape[:2]
    if tgt.shape[:2] != (Hs, Ws):
        tgt = np.array(Image.fromarray(tgt).resize((Ws, Hs), Image.BILINEAR))

    src = src.astype(np.float32)
    tgt = tgt.astype(np.float32)

    sf = np.fft.fft2(src, axes=(0, 1))
    tf = np.fft.fft2(tgt, axes=(0, 1))
    sa, sp = np.abs(sf), np.angle(sf)
    ta = np.abs(tf)

    # pick box radius
    b = int(round(min(Hs, Ws) * float(beta)))
    b = max(1, min(b, Hs // 2 - 1, Ws // 2 - 1))  # keep inside image and >=1
    if b < 1:  # super tiny images fallback
        return src.clip(0, 255).astype(np.uint8)

    ch, cw = Hs // 2, Ws // 2
    h0, h1 = ch - b, ch + b
    w0, w1 = cw - b, cw + b

    # swap the low-frequency square
    sa[h0:h1, w0:w1, :] = ta[h0:h1, w0:w1, :]

    out = np.fft.ifft2(sa * np.exp(1j * sp), axes=(0, 1)).real
    return np.clip(out, 0, 255).astype(np.uint8)

def scan_mapped(root, folder2u, allowed_u):
    by_u = {u:[] for u in allowed_u}
    root = Path(root)
    for d in root.iterdir():
        if not d.is_dir(): continue
        key = d.name
        if key not in folder2u: continue
        u = folder2u[key]
        if u not in allowed_u: continue
        for im in d.rglob("*"):
            if im.suffix.lower() in EXTS:
                by_u[u].append(str(im))
    return by_u

def count_pd_train(pd_train_dir, pd2u, unified):
    counts = {u:0 for u in unified}
    for d in Path(pd_train_dir).iterdir():
        if not d.is_dir(): continue
        key = d.name
        if key not in pd2u: continue
        u = pd2u[key]
        if u not in counts: continue
        for im in d.rglob("*"):
            if im.suffix.lower() in EXTS:
                counts[u] += 1
    return counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classmap", required=True)
    ap.add_argument("--pv-dir", required=True, help="PlantVillage root (raw/color)")
    ap.add_argument("--pd-train-dir", required=True, help="PlantDoc train root")
    ap.add_argument("--out-dir", required=True, help="Output root (unified folders)")
    # balancing options (choose one)
    ap.add_argument("--balance-to", type=int, default=0, help="target per-class count (e.g., 150). If 0, use quantile.")
    ap.add_argument("--balance-quantile", type=float, default=0.75, help="quantile of PD counts to balance up to (e.g., 0.75~Q3).")
    ap.add_argument("--beta", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    cm = json.load(open(args.classmap))
    unified = cm["unified_classes"]
    pv2u, pd2u = cm["plantvillage_to_unified"], cm["plantdoc_to_unified"]

    # PD counts and target
    pd_counts = count_pd_train(args.pd_train_dir, pd2u, unified)
    counts_arr = np.array([pd_counts[u] for u in unified], dtype=np.int32)
    if args.balance_to > 0:
        target = args.balance_to
    else:
        q = np.quantile(counts_arr, args.balance_quantile)
        target = int(round(q))
    print(f"[info] PD counts quantiles: min={counts_arr.min()} Q50={np.quantile(counts_arr,0.5):.1f} "
          f"Q75={np.quantile(counts_arr,0.75):.1f} max={counts_arr.max()}  -> target={target}")

    # collect donors/sources
    pv_by_u = scan_mapped(args.pv_dir, pv2u, unified)
    pd_by_u = scan_mapped(args.pd_train_dir, pd2u, unified)

    out_root = Path(args.out_dir)
    total = 0
    for u in unified:
        need = max(0, target - pd_counts[u])
        if need == 0:
            print(f"{u:35s} need=0 (skip)")
            continue
        pv_list, pd_list = pv_by_u[u], pd_by_u[u]
        if not pv_list or not pd_list:
            print(f"{u:35s} no donors or sources (skip)"); continue

        random.shuffle(pv_list); random.shuffle(pd_list)
        made = 0; out_dir = out_root / u
        # simple round-robin over PV images, pair each with random PD style until 'need' reached
        i = 0
        while made < need and i < len(pv_list)*10:  # safety loop
            s = pv_list[i % len(pv_list)]
            t = random.choice(pd_list)
            out = fda_source_to_target(imread(s), imread(t), beta=args.beta)
            name = f"{Path(s).stem}_fda{made:05d}.jpg"
            imsave(out, out_dir / name)
            made += 1; total += 1; i += 1
        print(f"{u:35s} PD={pd_counts[u]:4d}  -> +SYN={made:4d}  (to target {target})")
    print(f"[done] wrote {total} FDA images under {out_root}")

if __name__ == "__main__":
    main()
