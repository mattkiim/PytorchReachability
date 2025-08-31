import os, sys, pathlib, pickle as pkl
import numpy as np
import torch
from torch import nn, optim
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

    all_rgb, all_ir, all_labels = [], [], []
    for run in data:
        rgb_seq  = run['obs']['image']      # (T,H,W,3)
        ir_seq   = run['obs']['heat']       # (T,H,W,1)
        heat     = np.array(run['obs']['priv_heat'])
        labels   = (heat > 0.8 - 1e-6).astype(np.int64)  # unsafe if priv_heat ≥ 0.8

        all_rgb.append(rgb_seq)
        all_ir.append(ir_seq)
        all_labels.append(labels)

    all_rgb    = np.concatenate(all_rgb, axis=0)
    all_ir     = np.concatenate(all_ir, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_rgb, all_ir, all_labels


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
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def set_temperature(self, valid_loader, device):
        """Tune temperature on calibration set by minimizing NLL."""
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

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
    # First, precompute embeddings for calibration set
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
    correct, total = 0, 0
    with torch.no_grad():
        for rgb, ir, labels in eval_loader:
            feats = encoder_fn(rgb.to(device), ir.to(device))
            logits = clf_ts(feats)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += len(labels)
    print(f"Eval accuracy ({tag}): {correct/total:.3f}")
    return clf_ts


# -------------------
# Main
# -------------------
def main():
    config = load_config(config_path="configs/configs_v2.yaml")
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
    all_rgb, all_ir, all_labels = load_dataset("/data/mattkiim/wm_demos128_multimodal_v2plus_12.pkl")
    train_set, calib_set, eval_set = split_dataset(all_rgb, all_ir, all_labels)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    calib_loader = DataLoader(calib_set, batch_size=64, shuffle=False)
    eval_loader  = DataLoader(eval_set, batch_size=64)

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

    # --- run three experiments ---
    clf_rgb  = train_and_eval(enc_rgb,  enc_dim_rgb,  train_loader, calib_loader, eval_loader, config.device, "RGB")
    clf_ir   = train_and_eval(enc_ir,   enc_dim_ir,   train_loader, calib_loader, eval_loader, config.device, "IR")
    clf_both = train_and_eval(enc_both, enc_dim_both, train_loader, calib_loader, eval_loader, config.device, "RGB+IR")


if __name__ == "__main__":
    main()
