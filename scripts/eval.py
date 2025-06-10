import argparse
import functools
import pathlib
import torch
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))

import tools
from parallel import Parallel, Damy
from dreamer import Dreamer
from make_env import make_env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir, 0)

    print("Create eval envs.")
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        eval_envs = [Damy(env) for env in eval_envs]
    acts = eval_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    print("Create agent.")
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        eval_envs[0].observation_space,
        eval_envs[0].action_space,
        config,
        logger,
        eval_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    checkpoint_path = logdir / "latest.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        print("Loaded agent from checkpoint.")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Exiting.")
        return

    print("Start evaluation.")
    eval_policy = functools.partial(agent, training=False)
    tools.simulate(
        eval_policy,
        eval_envs,
        eval_eps,
        config.evaldir,
        logger,
        is_eval=True,
        episodes=config.eval_episode_num,
    )
    if config.video_pred_log:
        video_pred = agent._wm.video_pred(next(eval_dataset))
        logger.video("eval_openl", to_np(video_pred))

    for env in eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    import ruamel.yaml as yaml
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
