import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CentralizedCriticAttentionModel(TorchModelV2, nn.Module):
    """
    MAPPO with centralized critic + self-attention over agents.
    Actor is fully decentralized and unchanged.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config["custom_model_config"]
        self.num_agents = cfg["num_agents"]
        self.obs_dim = obs_space.shape[0]

        # =====================================================
        # ACTOR (DECENTRALIZED) - UNCHANGED
        # =====================================================
        self.actor = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_actor",
        )

        # =====================================================
        # CRITIC WITH ATTENTION
        # =====================================================

        # Shared embedding for each agent
        self.embed = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
        )

        # Self-attention across agents
        self.attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True,
        )

        # Value head
        self.critic_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._value_out = None
        self._last_batch_size = 1  # 🔑 important for RLlib

    # =====================================================
    # ACTOR FORWARD
    # =====================================================
    def forward(self, input_dict, state, seq_lens):
        logits, _ = self.actor(input_dict, state, seq_lens)
        self._last_batch_size = logits.shape[0]
        return logits, state

    # =====================================================
    # VALUE FUNCTION (RLlib hook)
    # =====================================================
    def value_function(self):
        if self._value_out is None:
            return torch.zeros(
                (self._last_batch_size,),
                device=next(self.parameters()).device,
            )
        return self._value_out.view(-1)

    # =====================================================
    # CENTRALIZED CRITIC WITH ATTENTION
    # =====================================================
    def forward_critic(self, critic_obs):
        """
        critic_obs shape: [B, num_agents * obs_dim]
        """

        B = critic_obs.shape[0]

        # 1️⃣ reshape: [B, N, obs_dim]
        x = critic_obs.view(B, self.num_agents, self.obs_dim)

        # 2️⃣ shared embedding
        x = self.embed(x)  # [B, N, 128]

        # 3️⃣ self-attention
        attn_out, _ = self.attn(x, x, x)  # [B, N, 128]

        # 4️⃣ global pooling
        global_ctx = attn_out.mean(dim=1)  # [B, 128]

        # 5️⃣ value
        self._value_out = self.critic_head(global_ctx)
