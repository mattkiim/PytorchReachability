"""Check evaluation's recurrent state against Dreamer's training-time inference."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import ruamel.yaml as yaml


@pytest.mark.parametrize("config_name", ["configs_po.yaml", "configs_fo.yaml"])
def test_eval_posterior_matches_observe(config_name, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import eval_hj
    from networks import RSSM

    defaults = yaml.YAML(typ="safe").load(
        (root / "configs" / config_name).read_text()
    )["defaults"]
    config = SimpleNamespace(**defaults)
    config.device = "cpu"
    dynamics = RSSM(stoch=3, deter=8, hidden=8, num_actions=2, embed=5,
                    initial="zeros", std_act="sigmoid2", device="cpu")
    margin = torch.nn.Linear(11, 1)
    embeddings = []
    first_flags = []

    def encode(obs):
        # A small encoder keeps the test focused on recurrent inference.
        embed = torch.cat([
            obs["obs_state"],
            obs["image"].mean(dim=(-3, -2, -1)).unsqueeze(-1),
            obs["heat"].mean(dim=(-3, -2, -1)).unsqueeze(-1),
        ], dim=-1)
        embeddings.append(embed)
        first_flags.append(obs["is_first"])
        return embed

    wm = SimpleNamespace(
        dynamics=dynamics,
        heads={"margin": margin},
        encoder=encode,
        preprocess=lambda obs: {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in obs.items()
        },
    )
    torch.manual_seed(123)
    result = eval_hj.dreamer_posterior_obsstep(
        config, wm, np.array([0.8, 0.1, 0.3, 0.5]), heat0=0.4, K=3,
    )
    torch.manual_seed(123)
    posterior, _ = dynamics.observe(
        torch.cat(embeddings, dim=1), torch.zeros(1, 4, 2),
        torch.cat(first_flags, dim=1),
    )
    final = {key: value[:, -1] for key, value in posterior.items()}
    for key in final:
        torch.testing.assert_close(result["post_T"][key], final[key])
    torch.testing.assert_close(result["feat_T"], dynamics.get_feat(final))
    torch.testing.assert_close(
        result["lz_T"], torch.tanh(margin(dynamics.get_feat(final))).squeeze(-1),
    )


@pytest.mark.parametrize("batch_step", [1, 4])
def test_rollout_classifies_each_state_independently(monkeypatch, batch_step):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import eval_hj

    class Policy:
        def __call__(self, batch, model):
            return SimpleNamespace(act=torch.zeros(len(batch.obs), 2))

        def critic(self, obs, act):
            return torch.tensor([[1.0], [-1.0], [1.0], [1.0]])

    # Safe, unsafe, false-safe, and false-unsafe, respectively.
    wm = SimpleNamespace(heads={
        "margin": lambda feats: torch.tensor([[1.0], [1.0], [1.0], [-1.0]]),
    })
    result, trajectories, failures, fps, fns = eval_hj.rollout_from_posts(
        SimpleNamespace(device="cpu", heat_threshold=0.8), wm, Policy(),
        posts={}, feats=torch.zeros(4, 3), states_xythv=torch.zeros(4, 4),
        heats0=torch.tensor([0.0, 1.0, 1.0, 0.0]), T=0, batch_step=batch_step,
    )
    assert result == {"TP": 1, "TN": 1, "FP": 1, "FN": 1}
    assert failures.tolist() == [False, True, True, False]
    assert len(trajectories) == 4
    assert len(fps) == len(fns) == 1
