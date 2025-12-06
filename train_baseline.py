import argparse, os, random, numpy as np, time, json
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def make_loaders(data_dir, img_size=224, batch_size=64, seed=42, normalize=False):
    # it is mentioned in the paper that there is no preprocessing applied to the PlantVillage dataset, so we are skipping the normalization by default
    t = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    if normalize:
        t += [transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
    tfm = transforms.Compose(t)

    ds = datasets.ImageFolder(data_dir, transform=tfm)
    y = np.array(ds.targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, tmp_idx = next(sss.split(np.zeros(len(y)), y))
    y_tmp = y[tmp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)  # split 20% into 10/10
    val_rel_idx, test_rel_idx = next(sss2.split(np.zeros(len(y_tmp)), y_tmp))
    val_idx, test_idx = tmp_idx[val_rel_idx], tmp_idx[test_rel_idx]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_paths = [os.path.abspath(p) for p, _ in ds.samples]
    split = {
        "paths": all_paths,                    
        "labels": ds.targets,                    
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "img_size": args.img_size,
        "normalized": bool(args.normalize),
    }
    with open(out_dir / "split.json", "w") as f:
        json.dump(split, f)
    print(f"dataset split lists saved to {out_dir/'split.json'}")

    dl_train = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_test  = DataLoader(Subset(ds, test_idx),  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return ds, dl_train, dl_val, dl_test

def build_model(num_classes):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, num_classes)
    return m

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_logits, all_y = [], []
    for x,y in dataloader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu()); all_y.append(y)
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    y_pred = logits.argmax(1).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro')
    return acc, f1

def train(args):
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds, dl_train, dl_val, dl_test = make_loaders(args.data_dir, args.img_size, args.batch_size, args.seed, args.normalize)
    num_classes = len(ds.classes)

    model = build_model(num_classes).to(device)

    # The hyperparameters as of given in the paper: 
    # Adam, betas=(0.5, 0.99), lr=1e-4; epochs=200; bs=64
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.99))
    ce = nn.CrossEntropyLoss()

    best_val = -1; best_path = Path(args.out_dir)/'mobilenetv3small_best.pt'
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for x,y in tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()*x.size(0)

        # validation
        val_acc, val_f1 = evaluate(model, dl_val, device)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model': model.state_dict(), 'classes': ds.classes}, best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch}] train_loss={(total_loss/len(dl_train.dataset)):.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

    # load best
    ckpt = torch.load(best_path, map_location='cpu')
    model.load_state_dict(ckpt['model']); model.to(device)

    test_acc, test_f1 = evaluate(model, dl_test, device)
    print(f"TEST acc={test_acc:.4f} | TEST macro-F1={test_f1:.4f}")

    final_path = Path(args.out_dir)/'mobilenetv3small_final.pt'
    torch.save({'model': model.state_dict(), 'classes': ds.classes}, final_path)
    print(f"Saved best to: {best_path}\nSaved final to: {final_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="PlantVillage-Dataset/raw/color/", help="Path to PlantVillage root (class folders)")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", action="store_true", help="Use ImageNet mean/std normalization (paper: none)")
    args = ap.parse_args()
    train(args)
