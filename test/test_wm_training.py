"""Regression checks for WM safety-head updates and policy action units."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from PyHJ.reach_rl_gym_envs.dubins_controls import policy_to_dynamics_action


def test_world_model_optimizer_updates_both_safety_heads(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from dreamer_offline import Dreamer

    wm = torch.nn.Module()
    wm.encoder = torch.nn.Linear(2, 2)
    wm.dynamics = torch.nn.Linear(2, 2)
    wm.heads = torch.nn.ModuleDict({name: torch.nn.Linear(2, 1)
                                  for name in ("decoder", "cont", "margin")})
    actor = torch.nn.Linear(2, 2)
    config = SimpleNamespace(precision=32, rssm_train_steps=1, model_lr=1e-3,
                             lx_lr=5e-4, opt_eps=1e-8, grad_clip=100, weight_decay=0,
                             opt="adam")
    agent = SimpleNamespace(_wm=wm, _config=config,
                            _task_behavior=SimpleNamespace(actor=actor))
    Dreamer._make_pretrain_opt(agent)
    assert {id(p) for p in agent.pretrain_params} == {id(p) for p in wm.parameters()}
    before = {name: wm.heads[name].bias.detach().clone() for name in wm.heads}
    actor_before = actor.weight.detach().clone()
    # A known nonzero supervised gradient for every head.
    features = wm.dynamics(wm.encoder(torch.ones(3, 2)))
    loss = sum(head(features).sum() for head in wm.heads.values())
    agent.pretrain_opt(loss, agent.pretrain_params, retain_graph=False)
    for name in wm.heads:
        assert not torch.equal(before[name], wm.heads[name].bias), name
    torch.testing.assert_close(actor.weight, actor_before)
    margin_group = next(group for group in agent.pretrain_opt._opt.param_groups
                        if any(p is wm.heads["margin"].weight for p in group["params"]))
    assert margin_group["lr"] == config.lx_lr


def test_dubins_environment_and_evaluation_use_same_action_units(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_hj
    Env = importlib.import_module("PyHJ.reach_rl_gym_envs.dubins-wm").Dubins_WM_Env
    seen = []

    class Dynamics:
        def imagine_with_action(self, actions, state):
            seen.append(actions.detach().clone().reshape(-1, 2))
            return {"deter": torch.ones(1, 1, 2)}

        def img_step(self, state, action):
            seen.append(action.detach().clone().reshape(-1, 2))
            return {"deter": torch.ones(1, 2)}

        def get_feat(self, state):
            return state["deter"]

    class Policy:
        def __call__(self, batch, model):
            return SimpleNamespace(act=torch.ones(len(batch.obs), 2))

        def critic(self, obs, act):
            return torch.ones(len(obs), 1)

        def actor(self, features):
            return torch.ones(len(features), 2), None

    config = SimpleNamespace(device="cpu", size=[128, 128], turnRate=1.25,
                              heat_threshold=0.8, dt=0.05)
    wm = SimpleNamespace(dynamics=Dynamics(), heads={"margin": lambda feats: feats[:, :1]})
    env = Env([config])
    env.wm = wm
    env.latent = {"deter": torch.ones(1, 1, 2)}
    env.safety_margin = lambda state: (1.0, 1.0)
    env.step(np.array([1.0, 1.0]))
    _, trajs, _, _, _ = eval_hj.rollout_from_posts(
        config, wm, Policy(), posts={"deter": torch.ones(1, 2)},
        feats=torch.ones(1, 2), states_xythv=torch.zeros(1, 4),
        heats0=torch.zeros(1), T=1, heat=False,
    )
    expected = torch.tensor([[1.25, 1.0]])
    assert len(seen) == 2
    for action in seen:
        torch.testing.assert_close(action, expected)
    assert abs(float(trajs[0][0][-1]) - 0.0025) < 1e-7
    np.testing.assert_allclose(policy_to_dynamics_action([2.0, -2.0], 1.25), [1.25, -1.0])
