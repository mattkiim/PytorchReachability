# If you find any problem, please contact me at: james.jingqi.li@gmail.com
# Thanks for your support!

# We apply DDPG to learn the classical reach-avoid value function, using the Bellman backup in arXiv:2112.12288.
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from PyHJ.data import Batch, ReplayBuffer
from PyHJ.exploration import BaseNoise, GaussianNoise
from PyHJ.policy.modelfree.BasePolicy_Annealing_Avoid import BasePolicy_Annealing_Avoid as BasePolicy # correct


class avoid_DDPGPolicy_annealing(BasePolicy):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971, 
        for learning the classical reach-avoid value function, arXiv:2112.12288.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param BaseNoise exploration_noise: the exploration noise,
        add to the action. Default to ``GaussianNoise(sigma=0.1)``.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        Default to False.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        critic: Optional[torch.nn.Module],
        critic_optim: Optional[torch.optim.Optimizer],
        critic_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        estimation_step: int = 1,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        actor: Optional[torch.nn.Module] = None, # control policy
        actor_optim: Optional[torch.optim.Optimizer] = None,
        actor_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        actor_gradient_steps: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        assert action_bound_method != "tanh", "tanh mapping is not supported" \
            "in policies where action is used as input of critic , because" \
            "raw action in range (-inf, inf) will cause instability in training"
        if critic is not None and critic_optim is not None:
            self.critic: torch.nn.Module = critic
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim: torch.optim.Optimizer = critic_optim
            self.critic_scheduler = critic_scheduler
        if actor is not None and actor_optim is not None:
            self.actor: torch.nn.Module = actor
            self.actor_old = deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim: torch.optim.Optimizer = actor_optim
            self.actor_scheduler = actor_scheduler
        
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma
        self._noise = exploration_noise
        self._rew_norm = reward_normalization
        self._n_step = estimation_step
        self.actor_gradient_steps = actor_gradient_steps
        self.new_expl = True
        self.warmup = False

    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    def train(self, mode: bool = True) -> "avoid_DDPGPolicy_annealing":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.critic.train(mode)
        self.actor.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.actor_old, self.actor, self.tau)
        self.soft_update(self.critic_old, self.critic, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """Predict the value of a state"""
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        target_q = self.critic_old(
            batch.obs_next,
            self(batch, model='actor_old', input='obs_next').act
        )
        return target_q
    

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute the target q values"""
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        if model=='actor_old':
            actor_model = getattr(self, "actor_old")
        elif model=='actor':
            actor_model = getattr(self, "actor")
        obs = batch[input]
        actions1, hidden1 = actor_model(obs, state=state, info=batch.info)
        
        return Batch(act=actions1, 
                    state=hidden1)

    @staticmethod
    def _mse_optimizer(
        batch: Batch, critic: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        critic_loss = (td.pow(2) * weight).mean()

        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Update critic network and actor network"""
        # critic
        td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        # if self.critic_scheduler is not None:
            # self.critic_scheduler.step()
        batch.weight = td  # prio-buffer
        # actor
        
        """Note that we update actor 5 times for each critic update!"""
        # update actor
        if not self.warmup:
            for _ in range(self.actor_gradient_steps):
                act = self(batch, model="actor").act
                actor_loss = -self.critic(batch.obs, act).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
        else:
            actor_loss = torch.tensor(0.0)
               
        
        # soft update the parameters
        self.sync_weight()
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                            batch: Batch) -> Union[np.ndarray, Batch]:
        
        if self._noise is None:
            act = act
        if isinstance(act, np.ndarray):
            act =  act + self._noise(act.shape)
        else:
            warnings.warn("Cannot add exploration noise to non-numpy_array action.")

        if self.new_expl:
            rand_act = np.random.uniform(-1, 1, act.shape)
            values = self.critic(batch.obs, rand_act).cpu().detach().numpy()
            act = np.where(values < 0.0, act, rand_act)

        if self.warmup:
            act = np.random.uniform(-1, 1, act.shape)

        return act
    
    
