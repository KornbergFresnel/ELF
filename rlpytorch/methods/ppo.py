import torch
import torch.nn as nn
import torch.nn.functional as Func

from torch.autograd import Variable
from ..args_provider import ArgsProvider
from .discounted_reward import DiscountedReward
from .value_matcher import ValueMatcher
from .utils import *


class PPO:
    def __init__(self):
        self.discounted_reward = DiscountedReward()
        self.value_matcher = ValueMatcher()

        self.args = ArgsProvider(
            call_from=self,
            define_args=[
                ("ratio_clip", dict(type=float, help="For importance sampling ratio clipping.", default=0.1)),
                ("policy_grad_clip_norm", dict(type=float, help="Gradient norm clipping.", default=None)),
                ("min_prob", dict(type=float, help="Minimal probability used in training", default=1e-6)),
                ("policy_action_nodes", dict(type=str, help=";separated string that specify policy_action nodes.",
                                             default="pi,a")),
                ("method_name", dict(type=str, help="KL_pen or nn.", default="nn")),
                ("grad_steps", dict(type=int, help="Define grad steps for policy update.", default=5))
            ],
            more_args=["num_games", "batchsize", "value_node"],
            child_providers=[self.discounted_reward.args, self.value_matcher.args],
            on_get_args=self._init,  # callback function for arguments initialization
        )

    def _init(self, args):
        """ Initialize policy loss to be an `nn.NLLLoss` and parse `policy_action_nodes`.

        :param args:
        """

        self.policy_loss = nn.NLLLoss().cuda()
        self.policy_action_nodes = []
        for node in args.policy_action_nodes.split(";"):
            policy, action = node.split(",")
            self.policy_action_nodes.append((policy, action))

    # @deprecated
    def _act_policy(self, logits, actions, use_logits):
        """ GumbleSoftmax

        :param logits: policy logits
        :param actions: action
        :return: action policy with shape=(-1, 1)
        """

        p0 = logits
        if use_logits:
            a0 = logits - logits.max(dim=1, keepdim=True)  # torch.reduce_max(logits, axis=1, keepdims=True)
            ea0 = a0.exp()  # tf.exp(a0)
            z0 = ea0.sum(dim=1, keepdims=True)  # tf.reduce_sum(ea0, axis=1, keepdims=True)
            p0 = ea0 / z0
        # action_mask = Func.one_hot(actions, num_classes=logits.size(1))  # tf.one_hot(actions, self._act_flat)
        # prob = (p0 * action_mask).sum(dim=1, keepdim=True, dtype=torch.float32)  # tf.reduce_sum(p0 * action_mask, axis=1)
        prob = p0.gather(1, actions.view(-1, 1)).squeeze()
        return prob

    def _compute_one_policy_loss(self, logits, old_logits, act, adv, use_logits):
        """ Compute policy error for one.

        :param logits: newest policy logits
        :param old_logits: old policy logits
        :param act: action
        :param adv: advantage
        :param use_logits: turn on logits or not
        :return: policy loss
        """

        policy_loss = 0.0
        ratio = self._act_policy(logits, act, use_logits) / self._act_policy(old_logits, act, use_logits).detach()
        surr = ratio * adv

        if self.args.method_name == 'kl_pen':
            # for _ in range(self.args.grad_steps):
            #     _, kl =
            pass
        else:
            clipped_ratio = ratio.clamp(1. - self.args.ratio_clip, 1. + self.args.ratio_clip)
            policy_loss = -torch.mean(torch.min(surr, clipped_ratio * adv))

        return dict(policy_loss=policy_loss)

    def _compute_policy_loss(self, logits, old_logits, act, adv, use_logits=False):
        """ Compute policy error

        :param logits: newest policy net raw output
        :param old_logits: old policy net raw output
        :param act: action
        :param adv: advantage value
        :param use_logits: turn on logits or not, default is False
        :return: error
        """

        args = self.args
        errs = {}

        if isinstance(logits, list):
            for i, (pix, old_pix) in enumerate(zip(logits, old_logits)):
                for j, (pixy, old_pixy) in enumerate(zip(pix, old_pix)):
                    errs = accumulate(errs, self._compute_policy_loss(pixy, old_pixy, adv[:, i, j], args.min_prob))
        else:
            errs = self._compute_one_policy_loss(logits, old_logits, act, adv, use_logits)

        return errs

    def _sync_policy_net(self):
        pass

    def _feed(self, adv, pi_s, actions, stats, old_pi_s=dict()):
        args = self.args
        batch_size = adv.size(0)

        policy_err = None

        for pi_node, a_node in self.policy_action_nodes:
            pi = pi_s[pi_node]
            a = actions[a_node].squeeze()

            assert pi_node in old_pi_s, "[ppo::_feed] pi_node should in both pi_s and old_pi_s."
            old_pi = old_pi_s[pi_node].squeeze()
            errs = self._compute_policy_loss(pi, old_pi, Variable(a), adv)

            stats['nll_' + pi_node].feed(errs['policy_loss'].data.item())
            policy_err = add_err(policy_err, errs['policy_loss'])

        return policy_err

    def update(self, mi, batch, stats):
        """ PPO model update
        :param mi:
        :param batch: batch of data, keys in a batch:
                        `s`: state,
                        `r`: imediate reward,
                        `terminal`: whether game is terminated
        :param stats: feed stats for later summarization
        """

        m = mi["model"]
        args = self.args
        value_node = self.args.value_node

        T = batch["s"].size(0)

        state_curr = m(batch.hist(T - 1))
        self.discounted_reward.setR(state_curr[value_node].squeeze().data, stats)

        err = None

        for t in range(T - 2, -1, -1):
            bht = batch.hist(t)
            state_curr = m.forward(bht)

            # go through the sample and get the rewards.
            V = state_curr[value_node].squeeze()

            R = self.discounted_reward.feed(
                dict(r=batch["r"][t], terminal=batch["terminal"][t]),
                stats=stats)

            policy_err = self._feed(R - V.data, state_curr, bht, stats, old_pi_s=bht)
            err = add_err(err, policy_err)
            err = add_err(err, self.value_matcher.feed({value_node: V, "target": R}, stats))

        stats["cost"].feed(err.data.item() / (T - 1))
        err.backward()
