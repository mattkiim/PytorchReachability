# imports
import os
import sys
import math
import yaml
import pathlib
import collections

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))

import tools

# (224, 224, 3) || 32 || SiLU || True || 4 || 7 

# implement the same CNN that is used in the Dreamer architecture

class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


# instantiate
cnn = ConvEncoder(
    input_shape=(224, 224, 1),
    depth=32,
    act="SiLU",
    norm=True,
    kernel_size=4,
    minres=7
)

expert_eps = collections.OrderedDict()
print("Expert Eps", expert_eps)
config_path = "/home/matthew/PytorchReachability/configs/configs_hw.yaml"
dataset_path = '/data/mattkiim/consolidated_hz1.h5'

import h5py

dataset_path = "/data/mattkiim/consolidated_hz1.h5"

threshold_fraction = 0.4

all_heat_frames, all_labels, all_traj_ids = [], [], []

with h5py.File(dataset_path, "r") as f:
    traj_keys = sorted(f.keys())
    for traj_i, traj_key in enumerate(traj_keys):
        heat = f[traj_key]["hot_inner"][:, :, :, :1]  # (T,H,W,1)
        for frame in heat:
            frame_wax = frame > 0.0
            active = np.sum(frame_wax)
            frac_super_hot = np.sum(frame[frame_wax] > 0.6) / active if active > 0 else 0.0
            label = 1 if frac_super_hot > threshold_fraction else 0
            all_heat_frames.append(frame.astype(np.float32))
            all_labels.append(label)
            all_traj_ids.append(traj_i)

            
all_labels = np.array(all_labels)
num_fails = np.sum(all_labels == 1)
num_safe = np.sum(all_labels == 0)

print(f"Safe frames: {num_safe}")
print(f"Fail frames: {num_fails}")
print(f"Total frames: {len(all_labels)}")
print(f"Fail percentage: {num_fails / len(all_labels) * 100:.2f}%")
# quit()
            
all_heat_frames = np.array(all_heat_frames, dtype=np.float32)
print(all_heat_frames[0].max()); quit()
all_labels = np.array(all_labels, dtype=np.int64)

print("Frames:", all_heat_frames.shape, "Labels:", all_labels.shape)


from torch.utils.data import Dataset, DataLoader

class HeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)        # (N, H, W, 1)
        self.y = torch.from_numpy(y)        # (N,)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = self.X[idx].permute(2, 0, 1)  # (1, H, W)
        return img, self.y[idx]

from torch.utils.data import Subset

all_labels   = np.array(all_labels, dtype=np.int64)
all_traj_ids = np.array(all_traj_ids, dtype=np.int64)

dataset = HeatDataset(np.array(all_heat_frames), all_labels)

uniq_traj = np.unique(all_traj_ids)
rng = np.random.default_rng(0)
rng.shuffle(uniq_traj)

n_train_traj = int(0.8 * len(uniq_traj))
train_traj = set(uniq_traj[:n_train_traj])
test_traj  = set(uniq_traj[n_train_traj:])

train_idx = np.nonzero(np.isin(all_traj_ids, list(train_traj)))[0]
test_idx  = np.nonzero(np.isin(all_traj_ids, list(test_traj)))[0]

train_ds = Subset(dataset, train_idx.tolist())
test_ds  = Subset(dataset, test_idx.tolist())

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)


class HeatClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder((224, 224, 1), 32, "SiLU", True, 4, 7)
        self.fc = nn.Linear(self.encoder.outdim, 2)  # binary classification
    def forward(self, x):
        # x: (B, 1, H, W)
        # Adjust to match ConvEncoder expected shape: (B, H, W, C)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        feat = self.encoder(x)     # (B, outdim)
        return self.fc(feat)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HeatClassifier().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

classes = np.array([0, 1])
weights = compute_class_weight('balanced', classes=classes, y=all_labels)
weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=weights)

for epoch in range(5):  # increase as needed
    model.train()
    total_loss, total_correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
    train_acc = total_correct / len(train_ds)

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_acc = val_correct / len(test_ds)

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_ds):.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
    
import os, shutil, csv
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ---------- evaluate once & collect everything ----------
model.eval()
all_preds, all_true, all_probs = [], [], []
cached_imgs = []  # (N,1,H,W)
with torch.no_grad():
    for imgs, labels in test_loader:
        logits = model(imgs.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(fail)
        preds = logits.argmax(1).cpu().numpy()
        labs  = labels.cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_true.extend(labs)
        cached_imgs.append(imgs.cpu().numpy())

all_preds = np.array(all_preds, dtype=int)
all_true  = np.array(all_true, dtype=int)
all_probs = np.array(all_probs, dtype=float)
cached_imgs = np.concatenate(cached_imgs, axis=0)  # (N,1,H,W)
N = cached_imgs.shape[0]
print(f"[eval] collected {N} eval frames")

# ---------- confusion matrix ----------
cm = confusion_matrix(all_true, all_preds, labels=[0,1])
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
print("Confusion [[TN,FP],[FN,TP]]:\n", cm)

# save cm plot
os.makedirs("eval_plots", exist_ok=True)
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.xticks([0,1], ["Safe(0)", "Fail(1)"])
plt.yticks([0,1], ["Safe(0)", "Fail(1)"])
plt.tight_layout()
plt.savefig("eval_plots/confusion_matrix.png", dpi=200)
plt.close()

# ---------- prepare output dirs ----------
root_all = "eval_plots/all"
root_bycase = "eval_plots/by_case"
for p in [root_all, root_bycase]:
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)

case_dirs = {
    "TP": os.path.join(root_bycase, "TP"),
    "FP": os.path.join(root_bycase, "FP"),
    "FN": os.path.join(root_bycase, "FN"),
    "TN": os.path.join(root_bycase, "TN"),
}
for d in case_dirs.values():
    os.makedirs(d, exist_ok=True)

# ---------- save all frames + categorized copies + manifest ----------
threshold_val = 0.6  # just for visualization overlay
manifest_path = os.path.join("eval_plots", "manifest.csv")
with open(manifest_path, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["idx", "true_label", "pred_label", "prob_fail"])  # header

    for idx in range(N):
        heat_img = cached_imgs[idx, 0]  # (H,W)
        # normalize to [0,1] for display only
        heat_norm = (heat_img - heat_img.min()) / (heat_img.ptp() + 1e-8)
        gray = (heat_norm * 255).astype(np.uint8)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)

        # red overlay for "super hot"
        mask = heat_norm > threshold_val
        overlay = np.zeros_like(gray_rgb)
        overlay[mask] = [255, 0, 0]
        blended = (0.7 * gray_rgb + 0.3 * overlay).astype(np.uint8)

        tl, pl, pf = int(all_true[idx]), int(all_preds[idx]), float(all_probs[idx])

        # save in 'all'
        fname = f"idx_{idx:06d}_label_{tl}_pred_{pl}_p{pf:.3f}.png"
        Image.fromarray(blended).save(os.path.join(root_all, fname))

        # categorized copy
        if tl == 1 and pl == 1:
            case = "TP"
        elif tl == 0 and pl == 1:
            case = "FP"
        elif tl == 1 and pl == 0:
            case = "FN"
        else:
            case = "TN"
        Image.fromarray(blended).save(os.path.join(case_dirs[case], fname))

        # manifest row
        writer.writerow([idx, tl, pl, pf])

print(f"[eval] saved all plots to '{root_all}', categorized to '{root_bycase}', and manifest at '{manifest_path}'")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
