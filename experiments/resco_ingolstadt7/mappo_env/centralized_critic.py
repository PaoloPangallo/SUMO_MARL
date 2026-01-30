import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CentralizedCriticModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.actor = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_actor",
        )

        critic_obs_space = model_config["custom_model_config"]["critic_obs_space"]
        self.critic = nn.Sequential(
            nn.Linear(critic_obs_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._value_out = None
        self._last_batch_size = 1  # 🔑 FONDAMENTALE

    def forward(self, input_dict, state, seq_lens):
        logits, _ = self.actor(input_dict, state, seq_lens)
        self._last_batch_size = logits.shape[0]
        return logits, state

    def value_function(self):
        if self._value_out is None:
            return torch.zeros(
                (self._last_batch_size,),
                device=next(self.parameters()).device,
            )
        return self._value_out.view(-1)

    def forward_critic(self, critic_obs):
        self._value_out = self.critic(critic_obs)
