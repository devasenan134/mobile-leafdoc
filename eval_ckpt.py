import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

class ListDataset(Dataset):
    def __init__(self, paths, labels, img_size=224, normalize=False):
        t = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        if normalize:
            t += [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]
        self.tf = transforms.Compose(t)
        self.paths = paths
        self.labels = labels

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.tf(img)
        y = int(self.labels[i])
        return x, y

def load_model_from_ckpt(pt_path: str, device: torch.device):
    ckpt = torch.load(pt_path, map_location="cpu")
    # infer num_classes from checkpoint
    classes = ckpt.get("classes", None)
    if isinstance(classes, list) and len(classes) > 0:
        num_classes = len(classes)
    else:
        # fall back to classifier weight shape
        w = ckpt["model"].get("classifier.3.weight", None)
        if w is None:
            # older keys sometimes use "classifier.1"
            w = ckpt["model"].get("classifier.1.weight", None)
        num_classes = int(w.shape[0]) if w is not None else 38

    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, num_classes)
    m.load_state_dict(ckpt["model"])
    m.to(device).eval()
    return m, classes, num_classes

@torch.no_grad()
def evaluate(model, loader, device):
    all_logits, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.cpu())
        all_labels.append(yb)
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = logits.argmax(1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    macro_auc = None
    try:
        num_classes = logits.shape[1]
        y_bin = label_binarize(labels, classes=np.arange(num_classes))
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        macro_auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        pass

    return acc, macro_f1, macro_auc, preds, labels


def load_split(split_path: str, subset: str, normalize_mode: str, img_size_override: int | None):
    sp = json.load(open(split_path, "r"))
    paths_all = sp["paths"]
    labels_all = sp["labels"]
    idx = sp[f"{subset}_idx"]
    paths = [paths_all[i] for i in idx]
    labels = [labels_all[i] for i in idx]

    split_norm = bool(sp.get("normalized", False))
    if normalize_mode == "auto":
        normalize = split_norm
    elif normalize_mode == "on":
        normalize = True
    else:
        normalize = False

    img_size = img_size_override if img_size_override else int(sp.get("img_size", 224))
    return paths, labels, img_size, normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to checkpoint .pt")
    ap.add_argument("--split", required=True, help="Path to split.json saved during training")
    ap.add_argument("--subset", choices=["train","val","test"], default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--img-size", type=int, default=None, help="Override image size")
    ap.add_argument("--normalize", choices=["auto","on","off"], default="auto",
                    help="ImageNet mean/std: auto=use split.json flag, on=force, off=disable")
    ap.add_argument("--save-cm-csv", type=str, default=None)
    ap.add_argument("--print-report", action="store_true")
    args = ap.parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # data split
    paths, labels, img_size, normalize = load_split(args.split, args.subset, args.normalize, args.img_size)

    # model (head size inferred from ckpt)
    model, classes, num_classes = load_model_from_ckpt(args.pt, device)

    # data loader
    ds = ListDataset(paths, labels, img_size=img_size, normalize=normalize)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # eval
    acc, macro_f1, macro_auc, preds, y_true = evaluate(model, dl, device)
    print(f"{args.subset.upper()}  Acc={acc:.4f}  Macro-F1={macro_f1:.4f}" +
          (f"  Macro-AUC={macro_auc:.4f}" if macro_auc is not None else ""))

    # optional outputs
    if args.save_cm_csv or args.print_report:
        cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
        if args.save_cm_csv:
            outp = Path(args.save_cm_csv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(outp, cm, fmt="%d", delimiter=",")
            print(f"[saved] confusion matrix CSV → {outp}")
        if args.print_report:
            target_names = classes if classes and len(classes) == num_classes else None
            print("\nPer-class report:")
            print(classification_report(y_true, preds, target_names=target_names, digits=4))


if __name__ == "__main__":
    main()
