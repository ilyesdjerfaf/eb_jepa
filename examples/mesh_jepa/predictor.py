"""State-only GRU predictor for Mesh JEPA.

Predicts next latent state from current state using a GRU.
No actions needed — the GRU uses the state itself as input.
Compatible with JEPA's autoregressive unroll mode.
"""

import torch.nn as nn


class MeshPredictor(nn.Module):
    """GRU-based predictor for state-only temporal prediction.

    In autoregressive mode, JEPA calls:
        pred_step = predictor(context_states, None)[:, :, -1:]

    where context_states is [B, D, ctxt_window, 1, 1].
    We use the state as GRU input and predict the next state.
    """

    def __init__(self, state_dim, hidden_dim=256, num_layers=1):
        super().__init__()
        self.is_rnn = True
        self.context_length = 0

        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, state_dim)
        self.ln = nn.LayerNorm(state_dim)

    def forward(self, state, action=None):
        """
        state: [B, D, T, 1, 1] — context states (JEPA 5D)
        action: ignored (state-only prediction)

        Returns: [B, D, T, 1, 1] — predicted states for each input timestep
        """
        B, D, T, _, _ = state.shape

        # Reshape: [B, D, T, 1, 1] → [B, T, D]
        x = state.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        # GRU forward
        out, _ = self.gru(x)  # (B, T, hidden_dim)

        # Project to state dim
        out = self.output_proj(out)  # (B, T, D)
        out = self.ln(out)

        # Reshape back: [B, T, D] → [B, D, T, 1, 1]
        out = out.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return out
