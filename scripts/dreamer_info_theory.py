import os, sys, pathlib, pickle as pkl
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader
import gymnasium as gym
import ruamel.yaml as yaml
import argparse

# --- import Dreamer ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
import models, tools

# -------------------
# Dataset utilities
# -------------------
def load_dataset(path):
    with open(path, "rb") as f:
        data = pkl.load(f)

    trajectories = []
    for run in data:
        rgb_seq  = np.array(run['obs']['image'])   # (T,H,W,3)
        ir_seq   = np.array(run['obs']['heat'])   # (T,H,W,1)
        heat     = np.array(run['obs']['priv_heat'])       # (T,)

        # make binary labels: unsafe if priv_heat ≥ 0.8
        labels   = (heat >= 0.8).astype(np.int64)

        trajectories.append((rgb_seq, ir_seq, labels))

    return trajectories



def split_dataset(all_rgb, all_ir, all_labels):
    rgb    = torch.from_numpy(all_rgb).float() / 255.0  # (N,3,H,W)
    ir     = torch.from_numpy(all_ir).float() / 255.0   # (N,1,H,W)
    labels = torch.from_numpy(all_labels).long()

    dataset = TensorDataset(rgb, ir, labels)

    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_calib = int(0.15 * n_total)
    n_eval  = n_total - n_train - n_calib

    return random_split(dataset, [n_train, n_calib, n_eval])


def split_dataset_traj(trajectories, seed=42):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(trajectories))
    rng.shuffle(idxs)

    n_total = len(idxs)
    n_train = int(0.8 * n_total)
    n_calib = int(0.1 * n_total)
    n_eval  = n_total - n_train - n_calib

    train = [trajectories[i] for i in idxs[:n_train]]
    calib = [trajectories[i] for i in idxs[n_train:n_train+n_calib]]
    eval  = [trajectories[i] for i in idxs[n_train+n_calib:]]

    return train, calib, eval


# -------------------
# Config loader
# -------------------
def load_config(section_names=None, config_path="configs.yaml"):
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    config_path = pathlib.Path(config_path)
    configs = yaml_loader.load(config_path.read_text())

    if section_names is None:
        name_list = ["defaults"]
    else:
        name_list = ["defaults", *section_names]

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    cfg = parser.parse_args([])

    return cfg

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, trajectories):
        self.rgb    = np.concatenate([traj[0] for traj in trajectories], axis=0)
        self.ir     = np.concatenate([traj[1] for traj in trajectories], axis=0)
        self.labels = np.concatenate([traj[2] for traj in trajectories], axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb = torch.from_numpy(self.rgb[idx]).float() / 255.0
        ir  = torch.from_numpy(self.ir[idx]).float() / 255.0
        label = torch.tensor(self.labels[idx]).long()
        return rgb, ir, label

# -------------------
# Classifier
# -------------------
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)


# -------------------
# Temperature Scaling
# -------------------
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def set_temperature(self, valid_loader, device):
        """Tune temperature on calibration set by minimizing NLL."""
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.log_temperature], lr=0.01, max_iter=50)

        def eval():
            loss = 0
            with torch.no_grad():
                for feats, labels in valid_loader:
                    feats, labels = feats.to(device), labels.to(device)
                    logits = self.model(feats) / self.temperature
                    loss += nll_criterion(logits, labels).item()
            return loss / len(valid_loader)

        def closure():
            optimizer.zero_grad()
            loss = 0
            for feats, labels in valid_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = self.model(feats) / self.temperature
                loss += nll_criterion(logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Optimal temperature: {self.temperature.item():.3f}, Val NLL={eval():.4f}")
        return self



# -------------------
# Training loop
# -------------------
def train_and_eval(encoder_fn, enc_dim, train_loader, calib_loader, eval_loader, device, tag):
    print(f"\n==== Training {tag} classifier ====")

    clf = MLPClassifier(enc_dim).to(device)
    opt = optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # --- Train ---
    for epoch in range(5):
        clf.train()
        total_loss = 0
        for rgb, ir, labels in train_loader:
            feats = encoder_fn(rgb.to(device), ir.to(device))  # get embeddings
            labels = labels.to(device)

            logits = clf(feats)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: train loss {total_loss/len(train_loader):.4f}")

    # Freeze MLP after training
    for p in clf.parameters():
        p.requires_grad = False

    # --- Calibrate ---
    calib_feats, calib_labels = [], []
    with torch.no_grad():
        for rgb, ir, labels in calib_loader:
            calib_feats.append(encoder_fn(rgb.to(device), ir.to(device)))
            calib_labels.append(labels)
    calib_feats = torch.cat(calib_feats)
    calib_labels = torch.cat(calib_labels)

    calib_dataset = torch.utils.data.TensorDataset(calib_feats, calib_labels)
    calib_loader_feats = DataLoader(calib_dataset, batch_size=64, shuffle=False)

    clf_ts = ModelWithTemperature(clf)
    clf_ts.set_temperature(calib_loader_feats, device)

    # --- Eval ---
    nll_criterion = nn.CrossEntropyLoss(reduction="sum")
    eval_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for rgb, ir, labels in eval_loader:
            feats = encoder_fn(rgb.to(device), ir.to(device))
            logits = clf_ts(feats)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += len(labels)
            eval_loss += nll_criterion(logits, labels.to(device)).item()

    eval_acc = correct / total
    eval_nll = eval_loss / total
    print(f"Eval results ({tag}): accuracy={eval_acc:.3f}, NLL={eval_nll:.4f}")

    return clf_ts

@torch.no_grad()
def confusion_for_classifier(
    clf, 
    loader, 
    device, 
    encoder_fn=None,          # required if loader yields (rgb, ir, labels, _)
    positive_label=1, 
    threshold=None            # if None -> argmax; else threshold on P(positive)
):
    clf.eval()
    tp = tn = fp = fn = 0
    n = 0

    for batch in loader:
        if len(batch) == 2:        # (feats, labels) from make_eval_loader(...)
            feats, labels = batch
        else:                      # (rgb, ir, labels, _) from raw DataLoader
            assert encoder_fn is not None, "Pass encoder_fn when using raw (rgb, ir, labels, _)"
            rgb, ir, labels, _ = batch
            feats = encoder_fn(rgb.to(device), ir.to(device))
        labels = labels.to(device)

        logits = clf(feats.to(device))
        if threshold is None:
            preds = logits.argmax(dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, positive_label] >= threshold).long()

        # counts
        tp += torch.sum((preds == positive_label) & (labels == positive_label)).item()
        tn += torch.sum((preds != positive_label) & (labels != positive_label)).item()
        fp += torch.sum((preds == positive_label) & (labels != positive_label)).item()
        fn += torch.sum((preds != positive_label) & (labels == positive_label)).item()
        n  += labels.numel()

    # metrics
    acc = (tp + tn) / max(n, 1)
    tpr = tp / max(tp + fn, 1)           # recall / sensitivity
    tnr = tn / max(tn + fp, 1)           # specificity
    ppv = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0  # precision
    npv = tn / max(tn + fn, 1) if (tn + fn) > 0 else 0.0
    f1  = (2 * ppv * tpr) / max(ppv + tpr, 1e-12) if (ppv + tpr) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)

    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn), 
        "total": int(n), "f1": f1, "balanced_accuracy": bal_acc,
        "matrix": [[tp/n, tn/n], [fp/n, fn/n]]
    }

# -------------------
# Compute entropy
# -------------------
def compute_entropy_and_mi(clf, val_loader, device):
    """
    clf: model returning logits
    val_loader: DataLoader over validation set (feats, labels)
    """
    # --- Step 1: empirical label distribution ---
    all_labels = []
    with torch.no_grad():
        for feats, labels in val_loader:   # only 2 items now
            all_labels.append(labels)
    all_labels = torch.cat(all_labels).cpu().numpy()
    classes, counts = np.unique(all_labels, return_counts=True)
    probs = counts / counts.sum()
    H_labels = -(probs * np.log(probs + 1e-12)).sum()

    # --- Step 2: NLL on validation set ---
    nll_loss = 0.0
    total = 0
    clf.eval()
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = clf(feats)
            log_probs = F.log_softmax(logits, dim=1)
            nll_loss += F.nll_loss(log_probs, labels, reduction="sum").item()
            total += labels.size(0)
    avg_nll = nll_loss / total

    # --- Step 3: MI bound ---
    mi_bound = H_labels - avg_nll

    print(f"H(Label) = {H_labels:.4f}, "
          f"NLL = {avg_nll:.4f}, "
          f"MI bound = {mi_bound:.4f}, "
          f"MI/H = {(mi_bound/H_labels):.4f}"
          )

    return H_labels, avg_nll, mi_bound


# -------------------
# Make eval loader
# -------------------
def make_eval_loader(encoder_fn, eval_loader, device):
    feats_list, labels_list = [], []
    with torch.no_grad():
        for rgb, ir, labels in eval_loader:
            feats = encoder_fn(rgb.to(device), ir.to(device))
            feats_list.append(feats.cpu())
            labels_list.append(labels)
    feats_all = torch.cat(feats_list)
    labels_all = torch.cat(labels_list)
    dataset = torch.utils.data.TensorDataset(feats_all, labels_all)
    return DataLoader(dataset, batch_size=64, shuffle=False)



# -------------------
# Main
# -------------------
def main():
    config = load_config(config_path="configs/configs_po.yaml")
    tools.set_seed_everywhere(config.seed)
    image_size = 128
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8),
        "heat": gym.spaces.Box(low=0, high=255, shape=(image_size, image_size, 1), dtype=np.uint8),
        "obs_state": gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
    })
    config.num_actions = act_space.shape[0]

    # Load pretrained WM
    wm = models.WorldModel(obs_space, act_space, 0, config)
    ckpt = torch.load(config.rssm_ckpt_path, map_location=config.device, weights_only=True)
    state_dict = {k[14:]: v for k,v in ckpt['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict, strict=False)
    wm.eval().to(config.device)

    for p in wm.encoder.parameters():
        p.requires_grad = False

    # --- dataset ---
    trajectories = load_dataset("train_data/wm_demos128_multimodal_v2plus3_info_theory_6.pkl")
    train_trajs, calib_trajs, eval_trajs = split_dataset_traj(trajectories)

    train_set = FrameDataset(train_trajs)
    calib_set = FrameDataset(calib_trajs)
    eval_set  = FrameDataset(eval_trajs)
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    calib_loader = DataLoader(calib_set, batch_size=64, shuffle=False)
    eval_loader  = DataLoader(eval_set, batch_size=64, shuffle=False)

    # --- encoders ---
    def enc_rgb(rgb, ir): 
        return wm.encoder._cnn(rgb)       # only RGB

    def enc_ir(rgb, ir):  
        return wm.encoder._heat_cnn(ir)   # only IR

    def enc_both(rgb, ir):
        return torch.cat([wm.encoder._cnn(rgb),
                          wm.encoder._heat_cnn(ir)], dim=-1)

    # --- embedding dims ---
    enc_dim_rgb  = wm.encoder._cnn.outdim
    enc_dim_ir   = wm.encoder._heat_cnn.outdim
    enc_dim_both = enc_dim_rgb + enc_dim_ir

    clf_rgb  = train_and_eval(enc_rgb, enc_dim_rgb, train_loader, calib_loader, eval_loader, config.device, "RGB")
    eval_loader_rgb = make_eval_loader(enc_rgb, eval_loader, config.device)
    compute_entropy_and_mi(clf_rgb, eval_loader_rgb, config.device)

    clf_ir   = train_and_eval(enc_ir, enc_dim_ir, train_loader, calib_loader, eval_loader, config.device, "IR")
    eval_loader_ir = make_eval_loader(enc_ir, eval_loader, config.device)
    compute_entropy_and_mi(clf_ir, eval_loader_ir, config.device)

    clf_both = train_and_eval(enc_both, enc_dim_both, train_loader, calib_loader, eval_loader, config.device, "RGB+IR")
    eval_loader_both = make_eval_loader(enc_both, eval_loader, config.device)
    compute_entropy_and_mi(clf_both, eval_loader_both, config.device)
    
    conf_rgb  = confusion_for_classifier(clf_rgb,  eval_loader_rgb,  config.device)
    conf_ir   = confusion_for_classifier(clf_ir,   eval_loader_ir,   config.device)
    conf_both = confusion_for_classifier(clf_both, eval_loader_both, config.device)

    print("RGB:",  conf_rgb)
    print("IR:",   conf_ir)
    print("Both:", conf_both)


if __name__ == "__main__":
    main()
