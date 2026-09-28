import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork


class LSTMCentralizedCritic(TorchModelV2, nn.Module):
    """
    MAPPO with centralized critic + LSTM (memory ONLY in critic)
    Actor is fully decentralized and feedforward.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config["custom_model_config"]

        self.n_agents = cfg["n_agents"]
        self.obs_dim = cfg["obs_dim"]
        self.hidden_dim = cfg.get("lstm_hidden_dim", 256)

        # =====================================================
        # ACTOR (DECENTRALIZED, FEEDFORWARD)
        # =====================================================
        self.actor = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_actor",
        )

        # =====================================================
        # CENTRALIZED CRITIC (LSTM)
        # =====================================================

        critic_input_dim = self.n_agents * self.obs_dim

        self.critic_encoder = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
            nn.ReLU(),
        )

        self.critic_lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        self.critic_head = nn.Linear(self.hidden_dim, 1)

        self._value_out = None
        self._last_batch_size = 1

    # =====================================================
    # ACTOR FORWARD
    # =====================================================
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]  # [B*T, obs_dim]
        B = seq_lens.shape[0]
        T = obs.shape[0] // B

        obs = obs.view(B, T, self.obs_dim)
        x = self.encoder(obs)

        h, c = state

        # 🔥 RLlib-safe reshape
        if h.dim() == 2:  # [B, H]
            h = h.unsqueeze(0)
            c = c.unsqueeze(0)

        lstm_out, (h, c) = self.lstm(x, (h, c))

        logits = self.policy_head(lstm_out)
        self._value_out = self.value_head(lstm_out).squeeze(-1)

        return logits.reshape(-1, logits.shape[-1]), [
            h.squeeze(0),
            c.squeeze(0),
        ]

    # =====================================================
    # CRITIC FORWARD (called manually from policy)
    # =====================================================
    def forward_critic(self, critic_obs):
        """
        critic_obs shape: [B, n_agents * obs_dim]
        """

        B = critic_obs.shape[0]

        # add fake time dimension: [B, T=1, obs]
        x = critic_obs.unsqueeze(1)

        x = self.critic_encoder(x)  # [B, 1, 256]

        lstm_out, _ = self.critic_lstm(x)  # [B, 1, hidden]

        value = self.critic_head(lstm_out[:, -1])  # [B, 1]

        self._value_out = value

    # =====================================================
    # RLlib VALUE HOOK
    # =====================================================
    def value_function(self):
        if self._value_out is None:
            return torch.zeros(
                (self._last_batch_size,),
                device=next(self.parameters()).device,
            )
        return self._value_out.view(-1)
