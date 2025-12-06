# export_quantize.py
import argparse, os, json, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

import onnx
from onnxruntime.quantization import (
    quantize_dynamic, quantize_static,
    QuantType, QuantFormat, CalibrationMethod, CalibrationDataReader
)
from onnxruntime.quantization.preprocess import quant_pre_process


# -----------------------------
# Load PyTorch model from .pt
# -----------------------------
def load_model(pt_path: str, num_classes: int):
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[3].in_features  # 1024
    m.classifier[3] = nn.Linear(in_features, num_classes)
    ckpt = torch.load(pt_path, map_location="cpu")
    m.load_state_dict(ckpt["model"])
    m.eval()
    classes = ckpt.get("classes", None)
    return m, classes


# -----------------------------
# Export ONNX (static by default)
# -----------------------------
def export_onnx(model, onnx_path: str, img_size=224, opset=19, dynamic_batch=False):
    dummy = torch.randn(1, 3, img_size, img_size)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset)

    # NOTE: MobileNetV3 ONNX graphs often contain internal reshape constants like [1,576].
    # Static batch (1) avoids those runtime reshape errors. Enable dynamic only if you know you need it.
    if dynamic_batch:
        kwargs["dynamic_shapes"] = [{0: "batch"}]  # inputs only

    torch.onnx.export(model, dummy, onnx_path, **kwargs)
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"[OK] Exported FP32 ONNX → {onnx_path}")


# -----------------------------
# ORT pre-process (shape infer / cleanups)
# -----------------------------
def preprocess_onnx(in_path: str) -> str:
    out_path = str(Path(in_path).with_suffix("").as_posix()) + "_pre.onnx"
    quant_pre_process(in_path, out_path)
    onnx.checker.check_model(onnx.load(out_path))
    print(f"[OK] Preprocessed ONNX → {out_path}")
    return out_path


# -----------------------------
# Calibration data reader (static INT8)
# -----------------------------
class ImageListCalibrationReader(CalibrationDataReader):
    def __init__(self, model_path: str, image_paths, img_size=224,
                 normalize=True, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
                 batch_size=1):
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = sess.get_inputs()[0].name

        self.img_size = img_size
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32).reshape(3,1,1)
        self.std  = np.array(std, dtype=np.float32).reshape(3,1,1)
        self.batch_size = batch_size

        self._batches = []
        self._index = 0

        batch = []
        for p in image_paths:
            x = self._load_preprocess(p)
            batch.append(x)
            if len(batch) == batch_size:
                self._batches.append({self.input_name: np.stack(batch, axis=0)})
                batch = []
        if batch:
            self._batches.append({self.input_name: np.stack(batch, axis=0)})

    def _load_preprocess(self, path):
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        x = np.array(img).astype("float32") / 255.0  # HWC
        x = np.transpose(x, (2,0,1))  # CHW
        if self.normalize:
            x = (x - self.mean) / self.std
        return x

    def get_next(self):
        if self._index >= len(self._batches):
            return None
        b = self._batches[self._index]
        self._index += 1
        return b


# -----------------------------
# Quantization wrappers
# -----------------------------
def do_dynamic_quant(fp32_path: str, out_int8_path: str):
    # Dynamic quantization mostly targets linear ops; keep it to Gemm/MatMul
    quantize_dynamic(
        model_input=fp32_path,
        model_output=out_int8_path,
        weight_type=QuantType.QInt8,
        per_channel=False,
        op_types_to_quantize=["Gemm", "MatMul"],
    )
    onnx.checker.check_model(onnx.load(out_int8_path))
    print(f"[OK] Quantized (dynamic INT8) → {out_int8_path}")

def do_static_quant(fp32_path: str, out_int8_path: str, reader: CalibrationDataReader):
    # Static INT8 (QDQ) quantizes Conv + MatMul with calibration; best for conv-heavy nets.
    quantize_static(
        model_input=fp32_path,
        model_output=out_int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "MatMul"],
        # optimize_model=False,  # we preprocessed already
    )
    onnx.checker.check_model(onnx.load(out_int8_path))
    print(f"[OK] Quantized (static INT8, QDQ) → {out_int8_path}")


# -----------------------------
# Split helper for calibration
# -----------------------------
def pick_paths_from_split(split_json: str, subset: str, n_samples: int, seed=42):
    split = json.load(open(split_json, "r"))
    all_paths = split["paths"]
    idx = list(split[f"{subset}_idx"])
    random.Random(seed).shuffle(idx)
    idx = idx[:min(n_samples, len(idx))]
    chosen = [all_paths[i] for i in idx]
    img_size = split.get("img_size", 224)
    normalized = bool(split.get("normalized", True))
    return chosen, img_size, normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to mobilenetv3small_best.pt")
    ap.add_argument("--num-classes", type=int, default=38)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--opset", type=int, default=19)
    ap.add_argument("--dynamic-batch", action="store_true", help="Try dynamic batch export (default: static 1)")
    ap.add_argument("--quant", choices=["none","dynamic","static"], default="static")
    ap.add_argument("--split", type=str, help="split.json (required for --quant static)")
    ap.add_argument("--subset", choices=["train","val","test"], default="train")
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--calib-batch", type=int, default=1)  # keep 1 to avoid reshape bugs
    ap.add_argument("--normalize", action="store_true", help="Apply ImageNet mean/std during calibration")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) Load torch model
    model, classes = load_model(args.pt, args.num_classes)

    # 2) Export ONNX (FP32)
    fp32 = out / "mobilenetv3small_fp32.onnx"
    export_onnx(model, str(fp32), img_size=args.img_size, opset=args.opset, dynamic_batch=args.dynamic_batch)

    # 3) Pre-process ONNX
    pre = preprocess_onnx(str(fp32))

    # 4) Quantize
    if args.quant == "dynamic":
        int8 = out / "mobilenetv3small_int8_dynamic.onnx"
        do_dynamic_quant(pre, str(int8))
    elif args.quant == "static":
        if not args.split:
            raise SystemExit("--split is required for --quant static")
        paths, s_img, s_norm = pick_paths_from_split(args.split, args.subset, args.calib_samples)
        use_norm = bool(args.normalize or s_norm)
        reader = ImageListCalibrationReader(
            model_path=pre, image_paths=paths,
            img_size=args.img_size or s_img,
            normalize=use_norm,
            batch_size=args.calib_batch  # keep 1 unless you’ve patched reshapes
        )
        int8 = out / "mobilenetv3small_int8_static_qdq.onnx"
        do_static_quant(pre, str(int8), reader)
    else:
        print("[INFO] Skipping quantization.")

    # 5) Size report
    def mb(p): return os.path.getsize(p)/(1024*1024)
    print(f"[SIZE] FP32: {mb(fp32):.2f} MB")
    if args.quant != "none":
        print(f"[SIZE] INT8 : {mb(int8):.2f} MB")


if __name__ == "__main__":
    main()
