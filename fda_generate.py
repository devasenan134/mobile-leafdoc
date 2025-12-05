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
    # src, tgt: HxWx3 uint8
    src = src.astype(np.float32); tgt = tgt.astype(np.float32)
    src_fft = np.fft.fft2(src, axes=(0,1))
    tgt_fft = np.fft.fft2(tgt, axes=(0,1))
    src_amp, src_pha = np.abs(src_fft), np.angle(src_fft)
    tgt_amp = np.abs(tgt_fft)
    H,W,_ = src.shape
    b = int(np.floor(min(H,W)*beta))
    c_h, c_w = H//2, W//2
    h0,h1 = c_h-b, c_h+b
    w0,w1 = c_w-b, c_w+b
    src_amp[h0:h1, w0:w1, :] = tgt_amp[h0:h1, w0:w1, :]
    fft = src_amp * np.exp(1j*src_pha)
    out = np.fft.ifft2(fft, axes=(0,1)).real
    return np.clip(out, 0, 255).astype(np.uint8)

def collect(root, folder2u, allowed_u):
    files_by_u = {u:[] for u in allowed_u}
    for d in Path(root).iterdir():
        if not d.is_dir(): continue
        key = d.name
        if key not in folder2u: continue
        u = folder2u[key]
        if u not in allowed_u: continue
        for im in d.rglob("*"):
            if im.suffix.lower() in EXTS:
                files_by_u[u].append(str(im))
    return files_by_u

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--classmap",required=True)
    ap.add_argument("--pv-dir",required=True, help="PlantVillage root (raw/color)")
    ap.add_argument("--pd-train-dir",required=True, help="PlantDoc train root")
    ap.add_argument("--out-dir",required=True, help="Output root for FDA images")
    ap.add_argument("--focus-from-summary", default=None, help="summary_pd.json to auto-focus hardest classes")
    ap.add_argument("--focus-topk", type=int, default=8)
    ap.add_argument("--per-src", type=int, default=1, help="FDA variants per PV image")
    ap.add_argument("--beta", type=float, default=0.03, help="Fourier swap ratio (0.02–0.06 typical)")
    ap.add_argument("--max-per-class", type=int, default=500, help="cap outputs per class")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    random.seed(args.seed)
    cm=json.load(open(args.classmap))
    unified=cm["unified_classes"]
    pv2u=cm["plantvillage_to_unified"]; pd2u=cm["plantdoc_to_unified"]

    # focus classes (optional) from analysis summary
    focus=set(unified)
    if args.focus_from_summary:
        summ=json.load(open(args.focus_from_summary))
        # take K hardest by F1 (ascending)
        pcs=summ["per_class"][:args.focus_topk]
        focus=set([p["class"] for p in pcs])
        print("Focusing classes:", sorted(focus))

    pv = collect(args.pv_dir, pv2u, focus)
    pd = collect(args.pd_train_dir, pd2u, focus)

    out_root=Path(args.out_dir)
    total=0
    for u in unified:
        if u not in focus: continue
        pv_list, pd_list = pv.get(u, []), pd.get(u, [])
        if not pv_list or not pd_list: continue
        random.shuffle(pv_list); random.shuffle(pd_list)
        out_cls = out_root / u
        count=0
        for s in pv_list:
            if count >= args.max_per_class: break
            src = imread(s)
            for _ in range(args.per_src):
                t = imread(random.choice(pd_list))
                out = fda_source_to_target(src, t, beta=args.beta)
                name = f"{Path(s).stem}_fda{_}.jpg"
                imsave(out, out_cls / name)
                count+=1; total+=1
                if count >= args.max_per_class: break
        print(f"{u:35s} -> {count} new")
    print(f"Total FDA images: {total}. Saved under {out_root}")

if __name__ == "__main__":
    main()
