# export_quantize.py (robust)
import argparse, torch, torch.nn as nn
from torchvision import models
from pathlib import Path
import onnx
from onnxsim import simplify

from onnxruntime.quantization import quantize_dynamic, QuantType

def load_model(pt_path, num_classes):
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[3].in_features  # 1024
    m.classifier[3] = nn.Linear(in_features, num_classes)
    ckpt = torch.load(pt_path, map_location="cpu")
    m.load_state_dict(ckpt["model"])
    m.eval()
    classes = ckpt.get("classes", None)
    return m, classes

def export_onnx(model, onnx_path, img_size=224, opset=19, dynamic_batch=True):
    dummy = torch.randn(1, 3, img_size, img_size)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset)
    if dynamic_batch:
        kwargs["dynamic_shapes"] = [{0: "batch"}]  # input dim-0 is dynamic
    torch.onnx.export(model, dummy, onnx_path, **kwargs)
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"[OK] Exported ONNX → {onnx_path}")

def preprocess_onnx(in_path: str) -> str:
    """
    Try ORT's quantization pre-process (shape inference, cleanups).
    If unavailable, just return input path.
    """
    out_path = str(Path(in_path).with_suffix("").as_posix()) + "_pre.onnx"
    try:
        from onnxruntime.quantization.preprocess import quant_pre_process
        quant_pre_process(
            in_path, out_path,
            # You can add: blocklist_ops=['Conv'] etc. if ever needed
        )
        onnx.checker.check_model(onnx.load(out_path))
        print(f"[OK] Preprocessed ONNX → {out_path}")
        return out_path
    except Exception as e:
        print(f"[WARN] Preprocess skipped ({e}). Using original graph.")
        return in_path

def simplify_onnx(in_path: str) -> str:
    """
    Run onnx-simplifier; helps resolve shape mismatches in some graphs.
    """
    out_path = str(Path(in_path).with_suffix("").as_posix()) + "_simp.onnx"
    try:
        model = onnx.load(in_path)
        simp_model, check = simplify(model)
        if check:
            onnx.save(simp_model, out_path)
            onnx.checker.check_model(onnx.load(out_path))
            print(f"[OK] Simplified ONNX → {out_path}")
            return out_path
        else:
            print("[WARN] onnx-simplifier check failed; using original.")
            return in_path
    except Exception as e:
        print(f"[WARN] Simplify skipped ({e}). Using previous graph.")
        return in_path

def quantize_onnx(fp32_path, int8_path):
    # Quantize only linear ops; this avoids some conv-related shape-infer quirks.
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        per_channel=False,
        op_types_to_quantize=["Gemm", "MatMul"],
    )
    print(f"[OK] Quantized (dynamic) → {int8_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--num-classes", type=int, default=38)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--opset", type=int, default=19)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model, classes = load_model(args.pt, args.num_classes)

    fp32 = out / "mobilenetv3small_fp32.onnx"
    export_onnx(model, str(fp32), img_size=args.img_size, opset=args.opset)

    # 1) ORT preprocess (shape infer / cleanups)
    pre = preprocess_onnx(str(fp32))
    # 2) Simplify (often fixes Gemm/MatMul inferred dims like 576 vs 1024)
    simp = simplify_onnx(pre)

    # 3) Quantize (dynamic) on the cleaned graph
    int8 = out / "mobilenetv3small_int8_dynamic.onnx"
    quantize_onnx(simp, str(int8))
