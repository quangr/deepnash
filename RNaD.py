import copy
from locale import currency
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils import RunningMeanStd


class RNaDPolicy(BasePolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1.
        Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        deterministic_eval: bool = False,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: Optional[float] = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        self.actor = actor
        self.optim = optim
        self.reg_actor = copy.deepcopy(actor)
        self.reg_actor_old = copy.deepcopy(actor)
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval
        self.critic = critic
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._lambda = gae_lambda
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self._grad_norm = max_grad_norm
        self._batch = max_batchsize
        self._actor_critic = ActorCritic(self.actor, self.critic)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        if not self._rew_norm:
            assert not self._value_clip, \
                "value clip is available only when `reward_normalization` is True"
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic
        self._delta_m = 3
        self._eta = 0.2

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        old_log_prob, reg_log_prob, old_reg_log_prob = [], [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(
                    self(minibatch).dist.log_prob(minibatch.act))
                reg_log_prob.append(
                    self.reg(minibatch).dist.log_prob(minibatch.act))
                old_reg_log_prob.append(
                    self.reg(minibatch, old=True).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        batch.reg_logp = torch.cat(reg_log_prob, dim=0)
        batch.reg_logp_old = torch.cat(old_reg_log_prob, dim=0)
        batch = self._transform_returns(batch, buffer, indices)
        batch = self._compute_values(batch, buffer, indices)
        return batch

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            print(self(Batch(obs=[[0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                                  [0, 1, 0, 1, 0, 0, 0, 1, 1,
                                      0, 0, 1, 0, 0, 1, 0, 0],
                                  [0, 1, 0, 0, 1, 0, 0, 1, 1,
                                      0, 0, 1, 0, 0, 1, 0, 0],
                                  [0, 1, 1, 0, 0, 0, 1, 0, 1,
                                      0, 0, 1, 0, 0, 1, 0, 0],
                                  [0, 1, 0, 1, 0, 0, 1, 0, 1,
                                      0, 0, 1, 0, 0, 1, 0, 0],
                                  [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])).logits.softmax(dim=1))
            if step > 0:
                #     batch = self._compute_returns(batch, self._buffer, self._indices)
                batch = self._compute_returns(
                    batch, self._buffer, self._indices)
                batch = self._compute_values(
                    batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                Q_s = minibatch.Q_s.clamp(-10000, 10000)
                logits = self(minibatch).logits#.gather(-1,minibatch.act.long().unsqueeze(-1))
                surr = logits * Q_s
                clip_loss = -surr.clamp(-2, 2).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                vf_loss = (minibatch.V_s - value).pow(2).mean()
                loss = clip_loss + self._weight_vf * vf_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                losses.append(loss.item())
        self.reg_actor_old = self.reg_actor
        # suppose to be average weight
        self.reg_actor = copy.deepcopy(self.actor)
        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }

    def _compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        # if self._rew_norm:  # unnormalize v_s & v_s_
        #     v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
        #     v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        # unnormalized_returns, advantages = self.compute_episodic_return(
        #     batch,
        #     buffer,
        #     indices,
        #     v_s_,
        #     v_s,
        #     gamma=self._gamma,
        #     gae_lambda=self._lambda
        # )
        # if self._rew_norm:
        #     batch.returns = unnormalized_returns / \
        #         np.sqrt(self.ret_rms.var + self._eps)
        #     self.ret_rms.update(unnormalized_returns)
        # else:
        #     batch.returns = unnormalized_returns
        # batch.returns = to_torch_as(batch.returns, batch.v_s)
        # batch.adv = to_torch_as(advantages, batch.v_s)
        return batch

    def _transform_returns(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        reg_reward = (batch.logp_old-batch.reg_logp).cpu().numpy()
        rew_ = np.zeros(batch.rew.shape)
        for i in range(2):
            rew_[:, i] = batch.rew[:, i] + \
                (1-2*(batch.info.current_player == i).astype(int)) * \
                self._eta*reg_reward
        batch.rew_ = rew_
        return batch

    def _compute_values(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        v_s = batch.v_s.cpu().numpy()
        rew = batch.rew_
        current_player = batch.info.current_player
        logp = self(batch).dist.log_prob(batch.act)
        reg_p = torch.exp(batch.reg_logp)
        reg_p_old = torch.exp(batch.reg_logp_old)
        p_ratio = torch.exp(logp-batch.logp_old)
        reg_p_ratio = torch.exp(logp-batch.reg_logp)
        V_s = np.zeros(batch.v_s.shape)
        Q_s = np.zeros(batch.v_s.shape+(2,))
        for player in range(2):
            for i in range(len(rew) - 1, -1, -1):
                if batch.done[i]:
                    v = 0
                    V = 0
                    r = 0
                    ksi = 1
                if current_player[i] != player:
                    r = rew[i][player]+p_ratio[i]*r
                    ksi = ksi*p_ratio[i]
                else:
                    rho = min(1, p_ratio[i]*ksi)
                    c = min(1, p_ratio[i]*ksi)
                    delta_V = rho*(rew[i][player]+r*p_ratio[i]+V-v_s[i])
                    for act in range(2):
                        ratio = reg_p
                        Q_s[i][act] = v_s[i]-self._eta*reg_p_ratio[i]+int(act == batch.act[i])/torch.exp(
                            batch.logp_old[i])*(rew[i][player]+self._eta*reg_p_ratio[i]+p_ratio[i]*(r+v)-v_s[i])
                    v = v_s[i]+delta_V+c*(v-V)
                    V = v_s[i]
                    r = 0
                    ksi = 1
                    V_s[i] = v
        batch.V_s = to_torch_as(V_s, batch.v_s)
        batch.Q_s = to_torch_as(Q_s, batch.v_s)
        return batch

    def reg(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        old=False,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        if old:
            logits, hidden = self.reg_actor_old(batch.obs, state=state)
        else:
            logits, hidden = self.reg_actor(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, hidden = self.actor(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)
