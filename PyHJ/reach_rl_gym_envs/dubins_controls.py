"""Convert normalized safety-policy actions to the WM dataset's action units."""

import numpy as np
import torch


def policy_to_dynamics_action(action, turn_rate):
    """Return [turn rate, acceleration], with ranges ±turn_rate and ±1.

    This matches generate_data_traj_cont.py. The first control is angular
    velocity, not angular acceleration. Preserve tensor gradients and device.
    """
    if torch.is_tensor(action):
        return action.clamp(-1, 1) * action.new_tensor([turn_rate, 1.0])
    return np.clip(np.asarray(action), -1, 1) * np.array([turn_rate, 1.0])
