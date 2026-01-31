import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork


class GATCentralizedCritic(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})

        # ✅ NOMI COERENTI
        self.n_agents = cfg["n_agents"]
        self.obs_dim = cfg["obs_dim"]
        self.embed_dim = cfg.get("embed_dim", 128)
        self.n_heads = cfg.get("n_heads", 4)

        # Actor
        self.actor = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_actor",
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.embed_dim),
            nn.ReLU(),
        )

        # Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.n_heads,
            batch_first=True,
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._value_out = None

    # =====================================================
    # ACTOR FORWARD (policy)
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
    # CENTRALIZED CRITIC (called by RLlib)
    # =====================================================
    def forward_critic(self, critic_obs):
        """
        critic_obs shape: [B, num_agents * obs_dim]
        """

        B = critic_obs.shape[0]

        # 1️⃣ reshape → [B, N, obs_dim]
        x = critic_obs.view(B, self.num_agents, self.obs_dim)

        # 2️⃣ shared embedding
        x = self.embed(x)  # [B, N, embed_dim]

        # 3️⃣ graph attention (self-attention over agents)
        attn_out, _ = self.attn(x, x, x)  # [B, N, embed_dim]

        # 4️⃣ global pooling (team representation)
        team_ctx = attn_out.mean(dim=1)  # [B, embed_dim]

        # 5️⃣ value
        self._value_out = self.value_head(team_ctx)
