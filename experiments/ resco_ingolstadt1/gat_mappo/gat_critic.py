import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork


class GATCentralizedCritic(TorchModelV2, nn.Module):
    """
    MAPPO with Graph-Attention centralized critic
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config["custom_model_config"]
        self.n_agents = cfg["n_agents"]
        self.obs_dim = cfg["obs_dim"]
        self.embed_dim = cfg.get("embed_dim", 128)
        self.n_heads = cfg.get("n_heads", 4)

        # -------------------------
        # Actor (unchanged, local)
        # -------------------------
        self.actor = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_actor",
        )

        # -------------------------
        # Critic: encoder
        # -------------------------
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.embed_dim),
            nn.ReLU(),
        )

        # -------------------------
        # Graph Attention
        # -------------------------
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.n_heads,
            batch_first=True,
        )

        # -------------------------
        # Value head
        # -------------------------
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._value_out = None

    # -------------------------
    # Actor forward
    # -------------------------
    def forward(self, input_dict, state, seq_lens):
        logits, _ = self.actor(input_dict, state, seq_lens)
        return logits, state

    # -------------------------
    # Critic forward (called manually)
    # -------------------------
    def forward_critic(self, critic_obs):
        """
        critic_obs shape: [B, n_agents * obs_dim]
        """
        B = critic_obs.shape[0]

        # reshape → [B, N, obs_dim]
        x = critic_obs.view(B, self.n_agents, self.obs_dim)

        # encode each agent
        x = self.encoder(x)  # [B, N, embed]

        # self-attention over agents
        attn_out, _ = self.attn(x, x, x)  # [B, N, embed]

        # aggregate (mean pooling)
        pooled = attn_out.mean(dim=1)  # [B, embed]

        self._value_out = self.value_head(pooled)

    def value_function(self):
        if self._value_out is None:
            return torch.zeros(1, device=next(self.parameters()).device)
        return self._value_out.view(-1)
