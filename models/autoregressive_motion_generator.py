"""Autoregressive transformer that predicts motion tokens from scene/text context."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from transformers import T5Tokenizer, T5EncoderModel


class AutoregressiveMotionGenerator(nn.Module):
    """
    Clean autoregressive motion token generator conditioned on:
      - DINO patch features (per-patch tokens)
      - T5 text features (per-token tokens, with attention mask)

    Action labels are accepted but ignored to keep the trainer interface stable.

    The motion side uses a learned positional embedding and a standard
    pre-LN Transformer decoder with cross-attention to the (text + DINO) memory.
    """

    def __init__(
        self,
        motion_vocab_size: int = 1024,
        dino_feature_dim: int = 768,
        text_feature_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 64,
        dropout: float = 0.0,
        t5_model_name: str = "google/flan-t5-large",
        action_feature_dim: int = 10,  # accepted for API compatibility, unused
        freeze_t5: bool = True,
        disable_loss_masking: bool = True,  # kept for API compatibility, default off
        ff_mult: int = 4,
        use_pre_decoder_layer_norm: bool = False,  # accepted, unused (pre-LN inside decoder)
        max_memory_len: int = 1024,
        max_text_tokens: int = 64,
    ):
        super().__init__()

        self.motion_vocab_size = motion_vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.start_token_id = motion_vocab_size
        self.padding_token_id = -100
        self.disable_loss_masking = disable_loss_masking
        self.max_text_tokens = max_text_tokens

        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_name)
        if freeze_t5:
            for p in self.t5_encoder.parameters():
                p.requires_grad = False

        # --- Conditioning projections ---
        self.text_proj = nn.Linear(text_feature_dim, hidden_dim)
        self.dino_proj = nn.Linear(dino_feature_dim, hidden_dim)

        # Per-modality LayerNorm for stable cross-attention scale
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.dino_norm = nn.LayerNorm(hidden_dim)

        # Learned modality + positional embeddings for memory tokens
        self.memory_modality_embed = nn.Embedding(2, hidden_dim)  # 0: text, 1: dino
        self.memory_pos_embed = nn.Embedding(max_memory_len, hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)

        # --- Motion side ---
        self.motion_embedding = nn.Embedding(motion_vocab_size + 1, hidden_dim)
        self.motion_pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.motion_in_norm = nn.LayerNorm(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-LN; standard for stable training
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, motion_vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---------- Conditioning encoders ----------
    def encode_text(self, text_prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_tokens: [B, T_text, hidden]
            attn_mask:   [B, T_text]   (True for valid)
        """
        device = next(self.parameters()).device
        encoded = self.t5_tokenizer(
            list(text_prompts) if text_prompts is not None else [""],
            padding="longest",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)

        with torch.no_grad():
            outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)

        text_features = outputs.last_hidden_state  # [B, T_text, text_feature_dim]
        text_tokens = self.text_norm(self.text_proj(text_features))
        return text_tokens, attention_mask.bool()

    def encode_dino(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
          - [B, P, D] (P patches, D feature dim)
          - [B, D, H, W]
          - [B, D]
        Returns: [B, P, hidden]
        """
        if dino_features.dim() == 4:
            dino_features = dino_features.flatten(2).permute(0, 2, 1)
        elif dino_features.dim() == 2:
            dino_features = dino_features.unsqueeze(1)
        return self.dino_norm(self.dino_proj(dino_features))

    def build_memory(
        self,
        dino_features: torch.Tensor,
        text_prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_mask = self.encode_text(text_prompts)            # [B, T_text, H], [B, T_text]
        dino_tokens = self.encode_dino(dino_features)                       # [B, P, H]

        device = dino_tokens.device
        B, P, _ = dino_tokens.shape
        T_text = text_tokens.shape[1]

        memory = torch.cat([text_tokens, dino_tokens], dim=1)               # [B, T_text + P, H]

        # Modality + positional embeddings
        modality_ids = torch.zeros(memory.shape[1], dtype=torch.long, device=device)
        modality_ids[T_text:] = 1
        modality_emb = self.memory_modality_embed(modality_ids).unsqueeze(0).expand(B, -1, -1)

        pos_ids = torch.arange(memory.shape[1], device=device).clamp_max(self.memory_pos_embed.num_embeddings - 1)
        pos_emb = self.memory_pos_embed(pos_ids).unsqueeze(0).expand(B, -1, -1)

        memory = self.memory_norm(memory + modality_emb + pos_emb)

        # Build padding mask: True at padding positions to be ignored by attention
        text_padding = ~text_mask                                            # [B, T_text], True = pad
        dino_padding = torch.zeros(B, P, dtype=torch.bool, device=device)    # all valid
        memory_key_padding_mask = torch.cat([text_padding, dino_padding], dim=1)
        return memory, memory_key_padding_mask

    # ---------- AR forward ----------
    def forward(
        self,
        dino_features: torch.Tensor,
        text_prompts: List[str],
        action_labels: Optional[torch.Tensor],  # ignored, kept for API
        iterations: int,
        motion_tokens: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        memory, memory_key_padding_mask = self.build_memory(dino_features, text_prompts)

        if motion_tokens is not None:
            return self._compute_loss(motion_tokens, memory, memory_key_padding_mask)
        return self.generate_motion_tokens(memory, memory_key_padding_mask)

    def _decode(
        self,
        input_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = input_tokens.device
        B, T = input_tokens.shape

        token_embeds = self.motion_embedding(input_tokens)
        pos_ids = torch.arange(T, device=device).clamp_max(self.motion_pos_embed.num_embeddings - 1)
        pos_emb = self.motion_pos_embed(pos_ids).unsqueeze(0).expand(B, -1, -1)
        x = self.motion_in_norm(token_embeds + pos_emb)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        decoder_output = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        decoder_output = self.decoder_norm(decoder_output)
        logits = self.output_proj(decoder_output)
        return logits

    def _compute_loss(
        self,
        motion_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = motion_tokens.shape
        device = motion_tokens.device

        start_tokens = torch.full((B, 1), self.start_token_id, device=device, dtype=motion_tokens.dtype)
        input_tokens = torch.cat([start_tokens, motion_tokens], dim=1)[:, :-1]

        logits = self._decode(input_tokens, memory, memory_key_padding_mask)

        target_tokens = motion_tokens
        loss = F.cross_entropy(
            logits.reshape(-1, self.motion_vocab_size),
            target_tokens.reshape(-1),
            ignore_index=self.padding_token_id,
        )

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            valid = (target_tokens != self.padding_token_id)
            accuracy = (preds[valid] == target_tokens[valid]).float().mean()
        return loss, accuracy

    @torch.no_grad()
    def generate_motion_tokens(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        target_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        B = memory.shape[0]
        device = memory.device
        generated = torch.full((B, 1), self.start_token_id, dtype=torch.long, device=device)
        target_length = min(target_length, self.max_seq_len)

        for _ in range(target_length):
            logits = self._decode(generated, memory, memory_key_padding_mask)[:, -1, :]
            logits = logits / max(1e-6, temperature)
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated[:, 1:]
