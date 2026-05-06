"""
utils/models.py
===============
Shared neural network architectures for the Federated Foundation Models pipeline.

These classes are defined here so they can be imported by any module notebook,
eliminating cross-kernel NameError issues.

Classes:
    HistoryAwareStateEncoder  – GRU-based state encoder (Module 2)
    OutcomeEmbeddingModel     – Transformer-based outcome embedder (Module 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ==============================================================================
# MODULE 2: HISTORY-AWARE STATE ENCODER
# ==============================================================================

class HistoryAwareStateEncoder(nn.Module):
    """
    GRU-based encoder that processes the full history of patient states up to
    each timestep, providing a richer latent representation for downstream tasks.

    Architecture:
        state_embed:        Linear -> LayerNorm -> GELU -> Dropout
        gru:                Multi-layer GRU (batch_first=True)
        latent_projection:  Linear -> Tanh
        dynamics_head:      Linear -> ReLU -> Dropout -> Linear  (for pre-training)

    Args:
        d_state:   Dimensionality of the input state vector.
        d_hidden:  Hidden size of GRU and embedding MLP.
        d_latent:  Size of the output latent representation.
        n_layers:  Number of GRU layers.
        dropout:   Dropout probability (applied inside GRU when n_layers > 1).
    """

    def __init__(
        self,
        d_state: int = 76,
        d_hidden: int = 256,
        d_latent: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_state = d_state
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        self.n_layers = n_layers

        # State embedding: project raw features to hidden dimension
        self.state_embed = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Recurrent encoder: captures temporal history
        self.gru = nn.GRU(
            input_size=d_hidden,
            hidden_size=d_hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        # Projection from GRU hidden space to latent space
        self.latent_projection = nn.Sequential(
            nn.Linear(d_hidden, d_latent),
            nn.Tanh(),
        )

        # Pre-training head: predicts next state given latent + action one-hot (dim=25)
        self.dynamics_head = nn.Sequential(
            nn.Linear(d_latent + 25, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_state),
        )

    def forward(
        self,
        states: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode state history up to each timestep.

        Args:
            states:  [B, T, d_state] – state sequences from MIMIC-III data.
            lengths: [B]             – actual sequence lengths (for packing).

        Returns:
            z: [B, T, d_latent] – history-aware encodings at each timestep.
        """
        x = self.state_embed(states)  # [B, T, d_hidden]

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        h, _ = self.gru(x)  # [B, T, d_hidden]

        if lengths is not None:
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        return self.latent_projection(h)  # [B, T, d_latent]

    def encode_single_timestep(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Return encoding at the final timestep only (for inference).

        Args:
            state_history: [B, T, d_state]

        Returns:
            z_current: [B, d_latent]
        """
        return self.forward(state_history)[:, -1, :]


# ==============================================================================
# MODULE 3: OUTCOME EMBEDDING MODEL
# ==============================================================================

class OutcomeEmbeddingModel(nn.Module):
    """
    Transformer-based model that embeds future patient trajectories into a
    clinically meaningful outcome space, trained via contrastive + mortality loss.

    Architecture:
        state_embed:          Linear -> LayerNorm -> GELU -> Dropout
        trajectory_encoder:   Transformer Encoder (norm_first, GELU)
        outcome_projection:   Linear -> ReLU -> Dropout -> Linear  (L2-normalised)
        mortality_head:       Linear -> ReLU -> Dropout -> Linear(1)
        good_outcome_reference: learnable buffer (updated post-training)

    Args:
        d_state:   Dimensionality of the input state vector.
        d_embed:   Size of the output outcome embedding.
        d_hidden:  Internal Transformer model dimension.
        n_layers:  Number of Transformer encoder layers.
        n_heads:   Number of attention heads.
        dropout:   Dropout probability.
    """

    def __init__(
        self,
        d_state: int = 76,
        d_embed: int = 64,
        d_hidden: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_state = d_state
        self.d_embed = d_embed
        self.d_hidden = d_hidden

        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer encoder over future trajectory
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=4 * d_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Project pooled representation to outcome embedding space
        self.outcome_projection = nn.Sequential(
            nn.Linear(d_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_embed),
        )

        # Auxiliary mortality prediction head (used during pre-training)
        self.mortality_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

        # Learnable reference vector for "good outcome" (updated after training)
        self.register_buffer("good_outcome_reference", torch.zeros(d_embed))

    def forward(
        self,
        future_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            future_states: [B, T, d_state]
            mask:          [B, T] – 1 for valid timesteps, 0 for padding.

        Returns:
            outcome_embed:  [B, d_embed]  – L2-normalised embedding.
            mortality_logit:[B, 1]        – raw logit for mortality prediction.
        """
        x = self.state_embed(future_states)

        attn_mask = ~mask.bool() if mask is not None else None
        h = self.trajectory_encoder(x, src_key_padding_mask=attn_mask)

        if mask is not None:
            h_pooled = (h * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-8)
        else:
            h_pooled = h.mean(1)

        outcome_embed = F.normalize(self.outcome_projection(h_pooled), dim=-1)
        mortality_logit = self.mortality_head(h_pooled)

        return outcome_embed, mortality_logit

    def compute_outcome_similarity(self, outcome_embedding: torch.Tensor) -> torch.Tensor:
        """Cosine similarity to the learned good-outcome reference vector."""
        return F.cosine_similarity(
            outcome_embedding,
            self.good_outcome_reference.unsqueeze(0),
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and architecture hyperparameters."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "d_state": self.d_state,
                "d_embed": self.d_embed,
                "d_hidden": self.d_hidden,
                "n_layers": self.trajectory_encoder.num_layers,
                "n_heads": self.trajectory_encoder.layers[0].self_attn.num_heads,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "OutcomeEmbeddingModel":
        """Load model from a checkpoint saved with :meth:`save`."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device)
        model = cls(
            d_state=ckpt.get("d_state", 76),
            d_embed=ckpt.get("d_embed", 64),
            d_hidden=ckpt.get("d_hidden", 256),
            n_layers=ckpt.get("n_layers", 3),
            n_heads=ckpt.get("n_heads", 8),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

# ==============================================================================
# MODULE 4: MULTI-OBJECTIVE REWARD MODEL
# ==============================================================================

class MultiObjectiveRewardModel(nn.Module):
    """
    Learns vector-valued rewards r: S × A → ℝᵏ from trajectory preferences.
    
    Architecture:
    - Causal transformer encoder (processes state-action history)
    - k objective-specific heads with uncertainty estimation
    - Learned temporal attention (not uniform discounting)
    """
    
    def __init__(self,
                 d_state: int = 70,
                 d_action: int = 25,
                 d_hidden: int = 512,
                 n_objectives: int = 4,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_state = d_state
        self.d_action = d_action
        self.d_hidden = d_hidden
        self.n_objectives = n_objectives
        
        # State-action embedding
        self.sa_embed = nn.Sequential(
            nn.Linear(d_state + d_action, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Causal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Objective-specific temporal attention heads
        # Each objective learns which timesteps matter most
        self.temporal_attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden // 2),
                nn.Tanh(),
                nn.Linear(d_hidden // 2, 1)
            )
            for _ in range(n_objectives)
        ])
        
        # Multi-objective reward heads (mean + log_std for each)
        self.reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden // 2, 2)  # [mean, log_std]
            )
            for _ in range(n_objectives)
        ])
        
        # Apply spectral normalization for stability
        # Critical for healthcare applications (prevents reward explosion)
        for head in self.reward_heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.utils.spectral_norm(layer)
        
        print(f"✅ MultiObjectiveRewardModel initialized:")
        print(f"   State: {d_state}, Action: {d_action}")
        print(f"   Hidden: {d_hidden}, Objectives: {n_objectives}")
        print(f"   Transformer: {n_layers} layers, {n_heads} heads")
        
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor,
                return_hidden: bool = False,
                return_uncertainty: bool = False,
                return_attention: bool = False) -> Tuple:
        """
        Compute multi-objective rewards with uncertainty.
        
        Args:
            states: [B, T, D_state]
            actions: [B, T, D_action] - one-hot encoded
            return_hidden: return transformer hidden states
            return_uncertainty: return uncertainty estimates
            return_attention: return temporal attention weights
            
        Returns:
            rewards: [B, T, k] - vector-valued rewards
            uncertainties: [B, T, k] (optional)
            hidden: [B, T, D_hidden] (optional)
            attention_weights: [B, T, k] (optional)
        """
        B, T, _ = states.shape
        
        # Validate dimensions
        assert states.shape[2] == self.d_state, f"Expected d_state={self.d_state}, got {states.shape[2]}"
        assert actions.shape[2] == self.d_action, f"Expected d_action={self.d_action}, got {actions.shape[2]}"
        
        # Embed state-action pairs
        sa = torch.cat([states, actions], dim=-1)
        sa_embed = self.sa_embed(sa)
        
        # Causal transformer (prevents future information leakage)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=states.device)
        h = self.transformer(sa_embed, mask=causal_mask)
        
        # Compute rewards for each objective
        reward_means = []
        reward_stds = []
        temporal_attentions = []
        
        for obj_idx in range(self.n_objectives):
            # Reward head (mean + log_std)
            out = self.reward_heads[obj_idx](h)
            mean = out[:, :, 0]
            log_std = out[:, :, 1]
            std = torch.exp(torch.clamp(log_std, -10, 2))  # Numerical stability
            
            reward_means.append(mean)
            reward_stds.append(std)
            
            # Temporal attention (if requested)
            if return_attention:
                attn_scores = self.temporal_attention_heads[obj_idx](h).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=1)
                temporal_attentions.append(attn_weights)
        
        # Stack into tensor [B, T, k]
        rewards = torch.stack(reward_means, dim=-1)
        uncertainties = torch.stack(reward_stds, dim=-1)
        
        # Batch normalization per objective
        normalized_rewards = rewards.clone()
        for obj_idx in range(self.n_objectives):
            r = rewards[:, :, obj_idx]
            mean = r.mean()
            std = r.std() + 1e-6
            normalized_rewards[:, :, obj_idx] = (r - mean) / std
        
        rewards = normalized_rewards
        
        # Prepare outputs
        outputs = [rewards]
        if return_uncertainty:
            outputs.append(uncertainties)
        if return_hidden:
            outputs.append(h)
        if return_attention:
            outputs.append(torch.stack(temporal_attentions, dim=-1))
        
        return tuple(outputs) if len(outputs) > 1 else rewards
    
    def compute_weighted_return(self,
                                  rewards: torch.Tensor,
                                  hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute temporally-weighted cumulative return (LEARNED weighting).
        """
        B, T, k = rewards.shape
        
        # Compute attention weights per objective
        temporal_weights = []
        
        for obj_idx in range(k):
            attn_scores = self.temporal_attention_heads[obj_idx](hidden).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=1)  # [B, T]
            temporal_weights.append(attn_weights)
        
        temporal_weights = torch.stack(temporal_weights, dim=-1)  # [B, T, k]
        
        # Weighted sum
        weighted_returns = (rewards * temporal_weights).sum(dim=1)  # [B, k]
        
        return weighted_returns



class ContinuousTimeEncoding(nn.Module):
    """
    Encodes irregular time intervals for continuous-time modeling.
    
    PROBLEM: ICU measurements are irregularly sampled (not every hour)
    SOLUTION: Fourier time embeddings that handle arbitrary Δt
    
    Based on: "Attention is All You Need" positional encoding +
              "SeFT: Learning Irregular Time Series" (Gong et al. 2020)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable frequency scaling
        self.omega = nn.Parameter(torch.randn(d_model // 2))
        
    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_t: [B, T] - time intervals in hours
        Returns:
            time_embed: [B, T, d_model]
        """
        # Fourier features
        # shape: [B, T, d_model//2]
        phase = delta_t.unsqueeze(-1) * self.omega.unsqueeze(0).unsqueeze(0)
        
        # sin/cos encoding
        time_embed = torch.cat([
            torch.sin(phase),
            torch.cos(phase)
        ], dim=-1)
        
        return time_embed

# ACUITY SCORING

class AcuityScorer(nn.Module):
    """
    Computes patient acuity score from state.
    
    CLINICAL DEFINITION:
    Acuity = severity of illness, used to stratify treatment intensity
    
    IMPLEMENTATION:
    - Primary: SOFA score (validated organ dysfunction measure)
    - Secondary: Vital sign instability, vasopressor dependency
    
    Output: c ∈ [0, 1] where 0=stable, 1=critical
    """
    def __init__(self, d_state: int = 70):
        super().__init__()
        
        # Indices in state vector 
        self.sofa_idx = 21  
        
        # Learnable weights for multi-signal acuity
        self.acuity_net = nn.Sequential(
            nn.Linear(d_state, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: [B, T, D_state]
        Returns:
            acuity: [B, T] in [0, 1]
        """
        return self.acuity_net(states).squeeze(-1)

# SAFETY-AWARE ACTION MASKING

class SafetyConstraint(nn.Module):
    """
    APPROACH:
    - Learn conservative Q-function Q_min(s, a, c)
    - Mask actions where Q_min < threshold
    - Prevents: excessive fluids (→ pulmonary edema), extreme vasopressors
    
    Based on: Kumar et al. "Conservative Q-Learning" (NeurIPS 2020)
    """
    def __init__(self,
                 d_state: int = 70,
                 d_acuity: int = 1,
                 n_actions: int = 25,
                 d_hidden: int = 256):
        super().__init__()
        
        self.n_actions = n_actions
        
        # Conservative Q-network
        self.q_net = nn.Sequential(
            nn.Linear(d_state + d_acuity, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_actions)
        )
        
        # Safety threshold (learned)
        self.log_threshold = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, states: torch.Tensor, acuity: torch.Tensor) -> torch.Tensor:
        """
        Compute action safety mask.
        
        Args:
            states: [B, T, D_state]
            acuity: [B, T]
        Returns:
            mask: [B, T, n_actions] - 1 if safe, 0 if unsafe
        """
        B, T, D = states.shape
        
        # Append acuity to state
        sa = torch.cat([states, acuity.unsqueeze(-1)], dim=-1)
        
        # Q-values
        q_values = self.q_net(sa)  # [B, T, n_actions]
        
        # Safety mask: Q > threshold
        threshold = torch.exp(self.log_threshold)
        mask = (q_values > threshold).float()
        
        # Ensure at least one action is available (fallback to safest)
        n_safe = mask.sum(dim=-1, keepdim=True)
        fallback = (n_safe == 0).float()
        
        if fallback.sum() > 0:
            # If no safe actions, allow the one with highest Q
            best_actions = q_values.argmax(dim=-1, keepdim=True)
            fallback_mask = torch.zeros_like(mask)
            fallback_mask.scatter_(-1, best_actions, 1.0)
            mask = mask + fallback * fallback_mask
        
        return mask
    
    def update(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               returns: torch.Tensor,
               acuity: torch.Tensor,
               optimizer: torch.optim.Optimizer) -> float:
        """
        Conservative Q-learning update for safety constraint.
        Expects flattened or properly shaped inputs.
        """
        # Flatten batch and time dimensions for Q-learning update
        if states.dim() == 3:   # [B, T, D] or [B, 1, D]
            states = states.view(-1, states.shape[-1])
        if actions.dim() == 3:  # [B, 1, T] or similar
            actions = actions.view(-1)
        if returns.dim() == 3:
            returns = returns.view(-1)
        if acuity.dim() == 3:
            acuity = acuity.view(-1)
        
        B = states.shape[0] 
        
        # Compute Q-values for all actions
        sa = torch.cat([states, acuity.unsqueeze(-1)], dim=-1)
        q_all = self.q_net(sa)                    # [B, n_actions]
        
        # Q-values for the taken actions
        q_taken = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)   # [B]
        
        # Bellman error (MSE)
        bellman_loss = F.mse_loss(q_taken, returns)
        
        # Conservative penalty (LogSumExp)
        logsumexp = torch.logsumexp(q_all, dim=-1)   # [B]
        conservative_penalty = logsumexp.mean() - q_taken.mean()
        
        # Total loss
        alpha = 1.0
        loss = bellman_loss + alpha * conservative_penalty
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()

# ACUITY-CONDITIONED DECISION TRANSFORMER

class AcuityConditionedDecisionTransformer(nn.Module):
    """
    Architecture:
        Input: (s_t, a_{t-1}, R̂_t, c_t, Δt_t) for t=0..T
        Embed each modality
        GPT-style causal transformer
        Predict: a_t ~ π(· | history, R̂_t, c_t, safety_mask)
    
    Key Differences from Standard DT:
    - Standard DT: π(a | s, R̂) with scalar R̂
    - AC-DT: π(a | s, R̂, c, Δt, mask) with vector R̂ ∈ ℝ^k
    """
    def __init__(self,
                 d_state: int = 70,
                 n_actions: int = 25,
                 n_objectives: int = 4,
                 d_model: int = 256,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 max_ep_len: int = 168):
        super().__init__()
        
        self.d_state = d_state
        self.n_actions = n_actions
        self.n_objectives = n_objectives
        self.d_model = d_model
        self.max_ep_len = max_ep_len
        
        # Embeddings for each modality
        self.state_embed = nn.Linear(d_state, d_model)
        self.action_embed = nn.Embedding(n_actions, d_model)
        self.return_ln = nn.LayerNorm(n_objectives)
        self.return_embed = nn.Sequential(
            nn.Linear(n_objectives, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.acuity_embed = nn.Linear(1, d_model)
        
        # Continuous time encoding
        self.time_encoder = ContinuousTimeEncoding(d_model)
        
        # Position embeddings (relative to episode start)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_ep_len * 3, d_model))
        
        # Layer norm for embeddings
        self.embed_ln = nn.LayerNorm(d_model)
        
        # GPT-style transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers
        )
        
        # Prediction heads

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, n_actions)
        )
        
        # Acuity scorer
        self.acuity_scorer = AcuityScorer(d_state)
        
        # Safety constraint
        self.safety = SafetyConstraint(d_state, 1, n_actions)
        
        print(f"✅ AcuityConditionedDecisionTransformer initialized:")
        print(f"   State: {d_state}, Actions: {n_actions}, Objectives: {n_objectives}")
        print(f"   Model dim: {d_model}, Layers: {n_layers}, Heads: {n_heads}")
        print(f"   Innovations: Acuity conditioning, Multi-objective, Continuous-time, Safety")
        
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor,
                returns_to_go: torch.Tensor,
                timesteps: torch.Tensor,
                delta_t: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_acuity: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through AC-DT.
        
        Args:
            states: [B, T, D_state]
            actions: [B, T] - previous actions (use dummy for t=0)
            returns_to_go: [B, T, k] - multi-objective returns
            timesteps: [B, T] - episode timesteps
            delta_t: [B, T] - time since last observation (hours)
            attention_mask: [B, T] - mask for variable length
            return_acuity: whether to return acuity scores
            
        Returns:
            action_logits: [B, T, n_actions]
            acuity: [B, T] (optional)
        """
        B, T, _ = states.shape
        
        # Compute acuity
        acuity = self.acuity_scorer(states)  # [B, T]
        
        # Embed each modality
        state_embeds = self.state_embed(states)
        action_embeds = self.action_embed(actions)
        returns_to_go = self.return_ln(returns_to_go)
        return_embeds = self.return_embed(returns_to_go)
        acuity_embeds = self.acuity_embed(acuity.unsqueeze(-1))
        
        # Time embeddings (continuous)
        time_embeds = self.time_encoder(delta_t)
        
        # Interleave: (R̂_0, s_0, a_0, R̂_1, s_1, a_1, ...)
        # This is the standard DT pattern
        sequence = torch.stack([
            return_embeds + time_embeds,
            state_embeds + acuity_embeds + time_embeds,
            action_embeds + time_embeds
        ], dim=2).reshape(B, 3 * T, self.d_model)
        
        # Add positional embeddings
        if 3 * T <= self.max_ep_len * 3:
            pos_embed = self.pos_embed[:, :3*T, :]
        else:
            # Interpolate if sequence too long
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=3*T,
                mode='linear'
            ).transpose(1, 2)
        
        sequence = sequence + pos_embed
        sequence = self.embed_ln(sequence)
        
        # Causal mask (prevent looking at future)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            3 * T,
            device=states.device
        )
        
        # Transformer
        hidden = self.transformer(
            sequence,
            sequence,  # Self-attention (decoder-only)
            tgt_mask=causal_mask
        )
        
        # Extract state positions (every 3rd token starting from index 1)
        # Pattern: [R̂, s, a, R̂, s, a, ...]
        #           0   1  2  3   4  5
        state_hidden = hidden[:, 1::3, :]  # [B, T, d_model]
        
        # Predict actions
        action_logits = self.action_head(state_hidden)  # [B, T, n_actions]
        
        # Apply safety mask
        safety_mask = self.safety(states, acuity)  # [B, T, n_actions]
        
        # Mask out unsafe actions (set logits to -inf)
        action_logits = action_logits + (1 - safety_mask) * (-1e9)
        
        if return_acuity:
            return action_logits, acuity
        else:
            return action_logits
    
    def get_action(self,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   returns_to_go: torch.Tensor,
                   timesteps: torch.Tensor,
                   delta_t: torch.Tensor,
                   temperature: float = 1.0,
                   deterministic: bool = False) -> torch.Tensor:
        """
        Sample action at last timestep (for autoregressive generation).
        
        Args:
            states: [B, T, D_state]
            actions: [B, T]
            returns_to_go: [B, T, k]
            timesteps: [B, T]
            delta_t: [B, T]
            temperature: sampling temperature
            deterministic: if True, take argmax
            
        Returns:
            action: [B] - sampled action
        """
        with torch.no_grad():
            action_logits = self.forward(
                states, actions, returns_to_go,
                timesteps, delta_t
            )
            
            # Take last timestep
            logits = action_logits[:, -1, :] / temperature
            
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return action
