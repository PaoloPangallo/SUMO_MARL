import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    MAPPO-style centralized critic:
    - Actor: uses local observation
    - Critic: uses concatenation of all agents' observations
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Actor network (local obs)
        self.actor = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_actor",
        )

        # Centralized critic network
        critic_obs_space = model_config["custom_model_config"]["critic_obs_space"]
        self.critic = nn.Sequential(
            nn.Linear(critic_obs_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        logits, _ = self.actor(input_dict, state, seq_lens)
        return logits, state

    def value_function(self):
        if self._value_out is None:
            # RLlib chiama value_function() prima del primo forward
            return torch.zeros(1, device=next(self.parameters()).device)
        return torch.reshape(self._value_out, [-1])

    def forward_critic(self, critic_obs):
        self._value_out = self.critic(critic_obs)
