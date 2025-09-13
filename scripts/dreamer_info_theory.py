import os, sys, pathlib, pickle as pkl
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader
import gymnasium as gym
import ruamel.yaml as yaml
import argparse
import h5py
from tqdm import tqdm


# import Dreamer
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
import models, tools

# -------------------
# Dataset utilities
# -------------------
def load_dataset(
    h5_path,
    max_trajs=None,
    normalize_actions=True,
    eps=1e-8,
    use_avg_rule=False,
    hot_mask_threshold=0.0,
    too_hot_threshold=0.6,
    frac_too_hot_threshold=0.5,
    avg_heat_threshold=0.1,
):
    # normalize actions
    action_mins, action_maxs = None, None
    if normalize_actions:
        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted(f.keys())
            if max_trajs: traj_keys = traj_keys[:max_trajs]
            for k in traj_keys:
                g = f[k]
                if "actions" not in g:
                    continue
                acts = g["actions"][:]
                action_mins = acts.min(axis=0) if action_mins is None else np.minimum(action_mins, acts.min(axis=0))
                action_maxs = acts.max(axis=0) if action_maxs is None else np.maximum(action_maxs, acts.max(axis=0))

    all_rgb, all_ir, all_labels = [], [], []
    traj_ids_per_frame = []      # ints aligned with frames
    key_map = []                 # index -> HDF5 key (for pretty-print)

    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(f.keys())
        if max_trajs: traj_keys = traj_keys[:max_trajs]

        for traj_idx, k in enumerate(tqdm(traj_keys, desc="Reading H5 demos", ncols=0)):
            g = f[k]
            if "camera_0" not in g:
                continue

            rgb_seq = g["camera_0"][:] * 255
            ir_seq  = g["camera_2"][:, :, :, :1] * 255
            heat_inner = g["hot_inner"][:]

            T = rgb_seq.shape[0]

            labels_list = []
            for t in range(T):
                hi = heat_inner[t]
                if not use_avg_rule:
                    frame_wax = hi > hot_mask_threshold
                    count = np.sum(frame_wax)
                    frac_too_hot = (np.sum(hi[frame_wax] > too_hot_threshold) / float(count)) if count > 0 else 0.0
                    heat_failure = frac_too_hot > frac_too_hot_threshold
                else:
                    heat_avg = float(np.mean(hi)) if hi.size > 0 else 0.0
                    heat_failure = heat_avg > avg_heat_threshold
                labels_list.append(int(heat_failure))

            all_rgb.append(rgb_seq.astype(np.uint8))
            all_ir.append(ir_seq.astype(np.uint8))
            all_labels.append(np.array(labels_list, dtype=np.int64))

            traj_ids_per_frame.extend([traj_idx] * T)
            key_map.append(k)  # remember which index corresponds to which HDF5 key

    all_rgb = np.concatenate(all_rgb, axis=0)
    all_ir = np.concatenate(all_ir, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    traj_ids_per_frame = np.asarray(traj_ids_per_frame, dtype=np.int64)

    return all_rgb, all_ir, all_labels, traj_ids_per_frame, key_map




def split_dataset(all_rgb, all_ir, all_labels, traj_ids_per_frame, seed=42):
    rgb    = torch.from_numpy(all_rgb).float()
    ir     = torch.from_numpy(all_ir).float()
    labels = torch.from_numpy(all_labels).long()
    traj_ids = torch.from_numpy(traj_ids_per_frame).long()  # <-- same length as frames

    dataset = TensorDataset(rgb, ir, labels, traj_ids)

    n_total = len(dataset)
    n_train = int(0.25 * n_total)
    n_calib = int(0.25 * n_total)
    n_eval  = n_total - n_train - n_calib

    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_calib, n_eval], generator=g)


def split_dataset_traj(all_rgb, all_ir, all_labels, traj_ids_per_frame, seed=42):
    num_trajs = traj_ids_per_frame.max().item() + 1
    all_traj_ids = np.arange(num_trajs)

    rng = np.random.default_rng(seed)
    rng.shuffle(all_traj_ids)

    n_train = int(0.8 * num_trajs)
    n_calib = int(0.1 * num_trajs)
    n_eval  = num_trajs - n_train - n_calib

    train_trajs = set(all_traj_ids[:n_train])
    calib_trajs = set(all_traj_ids[n_train:n_train+n_calib])
    eval_trajs  = set(all_traj_ids[n_train+n_calib:])

    # mask frames by trajectory id
    def subset(traj_set):
        idxs = [i for i, tid in enumerate(traj_ids_per_frame) if tid in traj_set]
        return torch.utils.data.Subset(
            TensorDataset(
                torch.from_numpy(all_rgb).float(),
                torch.from_numpy(all_ir).float(),
                torch.from_numpy(all_labels).long(),
                torch.from_numpy(traj_ids_per_frame).long(),
            ),
            idxs
        )

    return subset(train_trajs), subset(calib_trajs), subset(eval_trajs)


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
            # nn.Linear(in_dim, hidden),
            # nn.ReLU(),
            # nn.Linear(hidden, num_classes)
            nn.Linear(in_dim, num_classes)
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
def train_and_eval(encoder_fn, enc_dim, train_loader, calib_loader, eval_loader, device, tag, class_weights=None):
    print(f"\n==== Training {tag} classifier ====")

    clf = MLPClassifier(enc_dim).to(device)
    opt = optim.Adam(clf.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # train
    for epoch in range(10):
        clf.train()
        total_loss = 0
        for rgb, ir, labels, _ in train_loader:
            feats = encoder_fn(rgb.to(device), ir.to(device))
            labels = labels.to(device)

            logits = clf(feats)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: train loss {total_loss/len(train_loader):.4f}")

    # freeze MLP after training
    for p in clf.parameters():
        p.requires_grad = False

    # calibnrate
    calib_feats, calib_labels = [], []
    with torch.no_grad():
        for rgb, ir, labels, _ in calib_loader:
            calib_feats.append(encoder_fn(rgb.to(device), ir.to(device)))
            calib_labels.append(labels)
    calib_feats = torch.cat(calib_feats)
    calib_labels = torch.cat(calib_labels)

    calib_dataset = torch.utils.data.TensorDataset(calib_feats, calib_labels)
    calib_loader_feats = DataLoader(calib_dataset, batch_size=64, shuffle=False)

    clf_ts = ModelWithTemperature(clf)
    clf_ts.set_temperature(calib_loader_feats, device)
    
    # evaluate
    nll_criterion = nn.CrossEntropyLoss(reduction="sum")
    eval_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for rgb, ir, labels, _ in eval_loader:
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

def compute_class_weights_from_subset(subset, num_classes, device, eps=1e-6):
    """
    Returns (weights, counts) where `weights[c]` scales CE loss for class c.
    Uses N / (C * n_c) with safety for empty classes.
    """
    ds = subset.dataset                     # TensorDataset
    idx = subset.indices                    # indices into ds
    labels = ds.tensors[2][idx]             # 3rd tensor is labels
    counts = torch.bincount(labels, minlength=num_classes).float()
    # N / (C * n_c) with clamp to avoid inf when a class is missing
    weights = counts.sum() / (counts.clamp_min(eps) * num_classes)
    return weights.to(device), counts


# -------------------
# Compute entropy
# -------------------
def compute_entropy_and_mi(clf, val_loader, device):
    """
    clf: model returning logits
    val_loader: DataLoader over validation set (feats, labels)
    """
    # empirical label distribution
    all_labels = []
    with torch.no_grad():
        for feats, labels in val_loader:
            all_labels.append(labels)
    all_labels = torch.cat(all_labels).cpu().numpy()
    classes, counts = np.unique(all_labels, return_counts=True)
    probs = counts / counts.sum()
    # print(probs); quit()
    H_labels = -(probs * np.log(probs + 1e-12)).sum()

    # NLL on validation set
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

    # MI bound ---
    mi_bound = H_labels - avg_nll
    norm_mi  = mi_bound / H_labels if H_labels > 0 else 0.0

    print(f"H(Label) = {H_labels:.4f}, "
          f"NLL = {avg_nll:.4f}, "
          f"MI bound = {mi_bound:.4f}, "
          f"MI/H = {norm_mi:.4f}"
          )

    return H_labels, avg_nll, mi_bound

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
# Make eval loader
# -------------------
def make_eval_loader(encoder_fn, eval_loader, device):
    feats_list, labels_list = [], []
    with torch.no_grad():
        for rgb, ir, labels, _ in eval_loader:
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
    config = load_config(config_path="configs/configs_hw_merged_5hz_fast_avg_masked.yaml")
    tools.set_seed_everywhere(config.seed)
    image_size = 224
    
    cam_obs_space = gym.spaces.Box(
            low=0, high=1, shape=(image_size, image_size, 3), dtype=np.float32
        )
    policy_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
    bool_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)

    obs_observation_space = gym.spaces.Box(
        low=-1, high=1, shape=(8,), dtype=np.float32
    )

    heat_observation_space = gym.spaces.Box(
        low=0, high=1, shape=(image_size, image_size, 1), dtype=np.float32
    )

    obs_space = gym.spaces.Dict({
            'obs_state': obs_observation_space,
            'image': cam_obs_space,
            'heat': heat_observation_space,
            'is_first': bool_space,
            'is_last': bool_space,
            'is_terminal': bool_space,
            'policy': policy_obs_space,
            # 'wrist_cam': cam_obs_space,
        })

    low  = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    high = np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32)
    act_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    config.num_actions = act_space.shape[0]

    # Load pretrained WM
    wm = models.WorldModel(obs_space, act_space, 0, config)
    ckpt = torch.load(config.rssm_ckpt_path, map_location=config.device, weights_only=True)
    state_dict = {k[14:]: v for k,v in ckpt['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict, strict=True)
    wm.eval().to(config.device)

    for p in wm.encoder.parameters():
        p.requires_grad = False

    # dataset
    h5_path = config.dataset_path
    all_rgb, all_ir, all_labels, traj_ids_per_frame, key_map = load_dataset(
        h5_path,
        max_trajs=None,
        normalize_actions=True,
        use_avg_rule=config.avg,
        hot_mask_threshold=0.0,
        too_hot_threshold=0.6,
        frac_too_hot_threshold=0.5,
        avg_heat_threshold=0.1,
    )
    train_set, calib_set, eval_set = split_dataset_traj(all_rgb, all_ir, all_labels, traj_ids_per_frame, seed=config.seed)
    
    num_classes = 2  # you use 2-way classification
    class_weights, class_counts = compute_class_weights_from_subset(
        train_set, num_classes=num_classes, device=config.device
    )
    print(f"Train class counts: {class_counts.tolist()}")
    
    class_weights, class_counts = compute_class_weights_from_subset(
        eval_set, num_classes=num_classes, device=config.device
    )
    print(f"Eval class counts: {class_counts.tolist()}")
    print(f"CE class weights:   {class_weights.tolist()}")
    
    # print(len(train_set), len(calib_set), len(eval_set)); quit()
    
    # collect all trajectory ids in eval set
    traj_ids = [traj_id.item() for _, _, _, traj_id in eval_set]
    unique_traj_ids = sorted(set(traj_ids))

    print("Number of distinct trajectories in eval:", len(unique_traj_ids))
    for tid in unique_traj_ids:
        print("Trajectory:", key_map[tid])

    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    calib_loader = DataLoader(calib_set, batch_size=64, shuffle=False)
    eval_loader  = DataLoader(eval_set, batch_size=64)

    # encoders
    def enc_rgb(rgb, ir): 
        return wm.encoder._cnn(rgb)

    def enc_ir(rgb, ir):  
        return wm.encoder._heat_cnn(ir)

    def enc_both(rgb, ir):
        return torch.cat([wm.encoder._cnn(rgb),
                          wm.encoder._heat_cnn(ir)], dim=-1)

    # embedding dims
    enc_dim_rgb  = wm.encoder._cnn.outdim
    # enc_dim_ir   = wm.encoder._heat_cnn.outdim
    # enc_dim_both = enc_dim_rgb + enc_dim_ir

    clf_rgb  = train_and_eval(enc_rgb, enc_dim_rgb, train_loader, calib_loader, eval_loader, config.device, "RGB", class_weights)
    eval_loader_rgb = make_eval_loader(enc_rgb, eval_loader, config.device)
    compute_entropy_and_mi(clf_rgb, eval_loader_rgb, config.device)

    # clf_ir   = train_and_eval(enc_ir, enc_dim_ir, train_loader, calib_loader, eval_loader, config.device, "IR", class_weights)
    # eval_loader_ir = make_eval_loader(enc_ir, eval_loader, config.device)
    # compute_entropy_and_mi(clf_ir, eval_loader_ir, config.device)

    # clf_both = train_and_eval(enc_both, enc_dim_both, train_loader, calib_loader, eval_loader, config.device, "RGB+IR", class_weights)
    # eval_loader_both = make_eval_loader(enc_both, eval_loader, config.device)
    # compute_entropy_and_mi(clf_both, eval_loader_both, config.device)
    
    conf_rgb  = confusion_for_classifier(clf_rgb,  eval_loader_rgb,  config.device)
    # conf_ir   = confusion_for_classifier(clf_ir,   eval_loader_ir,   config.device)
    # conf_both = confusion_for_classifier(clf_both, eval_loader_both, config.device)

    print("RGB:",  conf_rgb)
    # print("IR:",   conf_ir)
    # print("Both:", conf_both)




if __name__ == "__main__":
    main()