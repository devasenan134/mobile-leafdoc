# eval_onnx.py
import argparse, json
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import onnxruntime as ort

def load_split(split_json: str, subset: str):
    sp = json.load(open(split_json, "r"))
    paths = sp["paths"]
    labels = np.array(sp["labels"], dtype=np.int64)
    idx = np.array(sp[f"{subset}_idx"], dtype=np.int64)
    img_size = sp.get("img_size", 224)
    normalized = bool(sp.get("normalized", True))
    return [paths[i] for i in idx], labels[idx], img_size, normalized

def preprocess(path, img_size=224, normalize=True):
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    x = np.array(img).astype("float32") / 255.0   # HWC
    x = np.transpose(x, (2,0,1))                  # CHW
    if normalize:
        mean = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(3,1,1)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(3,1,1)
        x = (x - mean) / std
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", choices=["train","val","test"], default="test")
    ap.add_argument("--batch-size", type=int, default=1)   # keep 1 for static-batch models
    ap.add_argument("--gpu", action="store_true", help="Use CUDA if available in ORT")
    ap.add_argument("--normalize", action="store_true", help="Force ImageNet normalization (else use split.json flag)")
    ap.add_argument("--img-size", type=int, default=None, help="Override image size")
    args = ap.parse_args()

    # dataset
    paths, labels, sp_img, sp_norm = load_split(args.split, args.subset)
    img_size = args.img_size or sp_img
    use_norm = bool(args.normalize or sp_norm)

    # ORT providers
    prov = ort.get_available_providers()
    use_gpu = args.gpu and ("CUDAExecutionProvider" in prov)
    providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    if args.gpu and not use_gpu:
        print("[WARN] CUDAExecutionProvider not available. Using CPU.")

    sess = ort.InferenceSession(args.onnx, providers=providers)
    inp = sess.get_inputs()[0]
    inp_name = inp.name
    out_name = sess.get_outputs()[0].name

    # detect static-batch=1 models
    static_b1 = False
    try:
        static_b1 = (inp.shape is not None and len(inp.shape) > 0 and inp.shape[0] == 1)
    except Exception:
        pass
    if static_b1 and args.batch_size != 1:
        print("[INFO] Model expects batch=1. Forcing batch-size=1 to avoid reshape errors.")
        args.batch_size = 1

    y_true, y_pred = [], []
    if args.batch_size == 1:
        for p, y in zip(paths, labels):
            xb = preprocess(p, img_size, use_norm)[None, ...]  # 1x3xHxW
            logits = sess.run([out_name], {inp_name: xb})[0]
            y_true.append(int(y))
            y_pred.append(int(np.argmax(logits, axis=1)[0]))
    else:
        # simple batched path
        for i in range(0, len(paths), args.batch_size):
            batch_paths = paths[i:i+args.batch_size]
            xb = np.stack([preprocess(p, img_size, use_norm) for p in batch_paths], axis=0)
            logits = sess.run([out_name], {inp_name: xb})[0]
            y_true.extend(labels[i:i+len(batch_paths)].tolist())
            y_pred.extend(np.argmax(logits, axis=1).tolist())

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"{args.subset.upper()}  acc={acc:.4f}  macro-F1={f1:.4f}")

if __name__ == "__main__":
    main()
