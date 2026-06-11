from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from collections import OrderedDict
import math


class Global_Trajectory_Pred(nn.Module):

    def __init__(self,
                 input_feats = 262,
                 output_feats = 3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()
        self.encoder = Encoder(input_feats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        
        self.decoder = Decoder(output_feats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, local_features: Tensor):
        # Preprocess
        B, seq_len, dim = local_features.shape    
        
        x_in = self.preprocess(local_features)
        
        
        # Encode
        x_encoder = self.encoder(x_in)
        # decoder
        x_decoder = self.decoder(x_encoder)
        x_out = self.postprocess(x_decoder)
        return x_out

class CrossAttentionHeightmapDecoder(nn.Module):
    def __init__(self, latent_dim=256, contact_dim=256, heightmap_dim=128, output_dim=265, initpose_dim=128, num_heads=4):
        super().__init__()
        self.hidden_initpose_dim = initpose_dim//2
        # Contact attention
        self.query_proj = nn.Linear(latent_dim, contact_dim)
        self.key_proj = nn.Linear(3, contact_dim)
        self.value_proj = nn.Linear(3, contact_dim)
        self.attn = nn.MultiheadAttention(embed_dim=contact_dim, num_heads=num_heads, batch_first=True)

        # Heightmap embedding
        self.height_embed = nn.Sequential(
            nn.Linear(heightmap_dim, heightmap_dim),
            nn.SiLU(),
            nn.Linear(heightmap_dim, heightmap_dim)
        )

        # FiLM modulation from initial pose
        self.init_pose_film = nn.Sequential(
            nn.Linear(initpose_dim, self.hidden_initpose_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_initpose_dim, latent_dim * 2)
        )

        # Final decoder MLP
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim + contact_dim + heightmap_dim, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, vq_latents, contact_positions, heightmap, initial_pose):
        # vq_latents: [B, T, D]
        # contact_positions: [B, T, M, 3]
        # heightmap: [B, T, G]

        B, T, M, _ = contact_positions.shape
        D = vq_latents.shape[-1]

        # --- Contact Attention ---
        query = self.query_proj(vq_latents)  # [B, T, C]
        query = query.view(B * T, 1, -1)     # [B*T, 1, C]
        key = self.key_proj(contact_positions).view(B * T, M, -1)     # [B*T, M, C]
        value = self.value_proj(contact_positions).view(B * T, M, -1) # [B*T, M, C]

        contact_context, _ = self.attn(query, key, value)  # [B*T, 1, C]
        contact_context = contact_context.squeeze(1)      # [B*T, C]

        # --- Heightmap Embedding ---
        height_feat = self.height_embed(heightmap)  # [B, T, H]
        height_feat = height_feat.view(B * T, -1)   # [B*T, H]

         # --- Initial Pose Embedding ---
        film_out = self.init_pose_film(initial_pose)         # [B, 2D]
        scale, bias = film_out.chunk(2, dim=-1)           # Each is [B, D]
        scale = scale.unsqueeze(1)                        # [B, 1, D]
        bias = bias.unsqueeze(1)                          # [B, 1, D]
        modulated_latents = scale * vq_latents + bias     # [B, T, D]
        modulated_flat = modulated_latents.contiguous().view(B * T, -1) 

        # --- Fuse all ---
        # latent_flat = vq_latents.reshape(B * T, -1)  # [B*T, D]
        fused = torch.cat([modulated_flat, contact_context, height_feat], dim=-1)  # [B*T, D + C + H]

        output = self.out_proj(fused)  # [B*T, J*3]
        output = output.view(B, T, -1)  # [B, T, J*3]
        return output

class VQVAE_decoder_heightmap_contact_crossattn(nn.Module):
    def __init__(self,
                 input_feats: int,
                 output_feats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "silu",
                 gpu_id: list = [0],
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = code_dim
        self.condition_emb_dim = kwargs['condition_emb_dim']
         
        self.encoder = Encoder(input_feats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.cond_decoder = CrossAttentionHeightmapDecoder(
                        latent_dim=code_dim,
                        contact_dim=kwargs['condition_emb_dim'],
                        heightmap_dim=kwargs['scene_dim'],
                        output_dim=code_dim,
                        initpose_dim=output_feats,
                        num_heads=kwargs['n_heads']
                    )
        
        self.decoder = Decoder(output_feats,
                               code_dim,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99, gpu_id=gpu_id)


    def forward(self, features, heightmap, contact_positions, init_features):
        
        # Preprocess
        x_in = features.permute(0, 2, 1)
        # Encode
        x_encoder = self.encoder(x_in)
        # Quantize
        x_quantized, commit_loss, perplexity, code_idx = self.quantizer(x_encoder)
        x_cond_decoder = self.cond_decoder(x_quantized.permute(0, 2, 1), contact_positions,
                                           heightmap.reshape(heightmap.shape[0], heightmap.shape[1], -1), init_features)
        x_decoder = self.decoder(x_cond_decoder.permute(0, 2, 1))
        x_out = x_decoder.permute(0, 2, 1)
        return x_out, commit_loss, perplexity, code_idx

    def motion_encode(self, features):
        # Preprocess
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        code_idx = (self.quantizer.quantize(self.quantizer.preprocess(x_encoder))).reshape(features.shape[0], -1)
        return code_idx
    
    def motion_decode(self, code_idx, heightmap, contact_positions):
        B, T = code_idx.shape  
        x_quantized = self.quantizer.dequantize(code_idx)
        x_quantized = x_quantized.view(B, -1, self.code_dim).contiguous()
        x_cond_decoder = self.cond_decoder(x_quantized, contact_positions,
                                           heightmap.reshape(heightmap.shape[0], heightmap.shape[1], -1), init_features)
        x_decoder = self.decoder(x_cond_decoder.permute(0, 2, 1))
        x_out = x_decoder.permute(0, 2, 1)
        return x_out

class VQVAE_decoder_heightmap_contact(nn.Module):
    def __init__(self,
                 input_feats: int,
                 output_feats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "silu",
                 gpu_id: list = [0],
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = code_dim
        self.condition_emb_dim = kwargs['condition_emb_dim']
         
        self.encoder = Encoder(input_feats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        # self.cond_decoder = CrossAttentionHeightmapDecoder(
        #                 latent_dim=code_dim,
        #                 contact_dim=kwargs['condition_emb_dim'],
        #                 heightmap_dim=kwargs['scene_dim'],
        #                 output_dim=code_dim,
        #                 initpose_dim=output_feats,
        #                 num_heads=kwargs['n_heads']
        #             )
        
        self.decoder = Decoder(output_feats,
                               code_dim + kwargs['scene_dim'] + 2000,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99, gpu_id=gpu_id)


    def forward(self, features, heightmap, contact_maps):
        
        # Preprocess
        x_in = features.permute(0, 2, 1)
        # Encode
        x_encoder = self.encoder(x_in)
        # Quantize
        x_quantized, commit_loss, perplexity, code_idx = self.quantizer(x_encoder)
        x_cond_ = torch.cat((x_quantized, contact_maps.permute(0, 2, 1), heightmap.reshape(heightmap.shape[0], heightmap.shape[1], -1).permute(0, 2, 1)), dim=1)
        x_decoder = self.decoder(x_cond_)
        x_out = x_decoder.permute(0, 2, 1)
        return x_out, commit_loss, perplexity, code_idx

    def motion_encode(self, features):
        # Preprocess
        x_in = features.permute(0, 2, 1)
        x_encoder = self.encoder(x_in)
        code_idx = (self.quantizer.quantize(self.quantizer.preprocess(x_encoder))).reshape(features.shape[0], -1)
        return code_idx
    
    def motion_decode(self, code_idx, heightmap, contact_maps):
        B, T = code_idx.shape  
        x_quantized = self.quantizer.dequantize(code_idx)
        x_quantized = x_quantized.view(B, -1, self.code_dim).contiguous()
        x_cond_ = torch.cat((x_quantized.permute(0, 2, 1), contact_maps.permute(0, 2, 1), heightmap.reshape(heightmap.shape[0], heightmap.shape[1], -1).permute(0, 2, 1)), dim=1)
        x_decoder = self.decoder(x_cond_)
        x_out = x_decoder.permute(0, 2, 1)
        return x_out

class VQVAE_decoder_heightmap(nn.Module):
    def __init__(self,
                 input_feats: int,
                 output_feats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "silu",
                 gpu_id: list = [0],
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = code_dim
        self.condition_emb_dim = kwargs['condition_emb_dim']
         
        self.encoder = Encoder(input_feats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        # self.cond_decoder = CrossAttentionHeightmapDecoder(
        #                 latent_dim=code_dim,
        #                 contact_dim=kwargs['condition_emb_dim'],
        #                 heightmap_dim=kwargs['scene_dim'],
        #                 output_dim=code_dim,
        #                 initpose_dim=output_feats,
        #                 num_heads=kwargs['n_heads']
        #             )
        
        self.decoder = Decoder(output_feats,
                               code_dim + kwargs['scene_dim'],
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99, gpu_id=gpu_id)


    def forward(self, features, heightmap, contact_maps):
        
        # Preprocess
        x_in = features.permute(0, 2, 1)
        # Encode
        x_encoder = self.encoder(x_in)
        # Quantize
        x_quantized, commit_loss, perplexity, code_idx = self.quantizer(x_encoder)
        x_cond_ = torch.cat((x_quantized, heightmap.reshape(heightmap.shape[0], heightmap.shape[1], -1).permute(0, 2, 1)), dim=1)
        x_decoder = self.decoder(x_cond_)
        x_out = x_decoder.permute(0, 2, 1)
        return x_out, commit_loss, perplexity, code_idx

    def motion_encode(self, features):
        # Preprocess
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        code_idx = (self.quantizer.quantize(self.quantizer.preprocess(x_encoder))).reshape(features.shape[0], -1)
        return code_idx
    
    def motion_decode(self, code_idx, heightmap, contact_positions):
        B, T = code_idx.shape  
        x_quantized = self.quantizer.dequantize(code_idx)
        x_quantized = x_quantized.view(B, -1, self.code_dim).contiguous()
        x_cond_ = torch.cat((x_quantized.permute(0, 2, 1), heightmap.reshape(heightmap.shape[0], heightmap.shape[1], -1).permute(0, 2, 1)), dim=1)
        x_decoder = self.decoder(x_cond_)
        x_out = x_decoder.permute(0, 2, 1)
        return x_out

class TransformerHeightmapDecoder(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, dim_feedforward)
        self.memory_proj = nn.Linear(input_dim, dim_feedforward)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, output_dim)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(dim_feedforward, dropout)
        
    def forward(self, x, memory=None):
        # x shape: [batch_size, seq_len, input_dim]
        # memory shape: [batch_size, seq_len, input_dim] (optional)
        
        # Project and add positional encoding to input sequence
        x = self.input_proj(x)  # [batch_size, seq_len, dim_feedforward]
        x = self.pos_encoder(x)  # Add positional encoding
        
        if memory is not None:
            # Project and add positional encoding to memory sequence
            memory = self.memory_proj(memory)  # [batch_size, seq_len, dim_feedforward]
            memory = self.pos_encoder(memory)  # Add positional encoding
            
            # The transformer decoder will:
            # 1. First do self-attention on x
            # 2. Then do cross-attention between x and memory
            x = self.transformer(x, memory)
        else:
            # If no memory, just do self-attention
            x = self.transformer(x, x)
        
        x = self.output_proj(x)  # [batch_size, seq_len, output_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x: [B, T, D]
        # pe: [1, max_len, D]
        # We want to add pe to x, but only up to the sequence length of x
        x = x + self.pe[:, :x.size(1), :]  # [B, T, D] + [1, T, D] -> [B, T, D]
        return self.dropout(x)

class VQVAE_decoder_heightmap_contact_transformer(nn.Module):
    def __init__(self,
                 input_feats: int,
                 output_feats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "silu",
                 gpu_id: list = [0],
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = code_dim
        self.condition_emb_dim = kwargs['condition_emb_dim']
         
        self.encoder = Encoder(input_feats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        # Transformer decoder for feature processing
        self.transformer_decoder = TransformerHeightmapDecoder(
            input_dim=code_dim,  # Input dimension for the target sequence
            output_dim=code_dim,  # Keep same dimension for conv decoder
            num_heads=kwargs.get('n_heads', 8),
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu'
        )

        # Convolutional decoder for temporal upsampling
        self.conv_decoder = Decoder(
            output_feats,
            code_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm
        )

        # Projections for heightmap and contact maps
        self.heightmap_proj = nn.Sequential(
            nn.Linear(kwargs['scene_dim'], code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, code_dim)
        )
        
        self.contact_proj = nn.Sequential(
            nn.Linear(2000, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, code_dim)
        )

        # FiLM generator for contact conditioning
        self.film_generator = nn.Sequential(
            nn.Linear(code_dim, code_dim * 2),  # Generates scale and shift
            nn.SiLU(),
            nn.Linear(code_dim * 2, code_dim * 2)
        )

        self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99, gpu_id=gpu_id)

    def forward(self, features, heightmap, contact_maps):
        # Preprocess
        x_in = features.permute(0, 2, 1)
        # Encode
        x_encoder = self.encoder(x_in)
        # Quantize
        x_quantized, commit_loss, perplexity, code_idx = self.quantizer(x_encoder)
        
        # Project and prepare memory
        B, T, _ = x_quantized.shape
        
        # Project heightmap and contact maps
        heightmap_emb = self.heightmap_proj(
            heightmap.reshape(heightmap.shape[0], heightmap.shape[1], -1)
        )  # [B, T, code_dim]
        
        contact_emb = self.contact_proj(
            contact_maps
        )  # [B, T, code_dim]
        
        # Generate FiLM parameters from contact information
        film_params = self.film_generator(contact_emb)  # [B, T, code_dim * 2]
        scale, shift = film_params.chunk(2, dim=-1)  # Each [B, T, code_dim]
        
        # Apply FiLM modulation to heightmap features
        memory = heightmap_emb * (1 + scale) + shift  # [B, T, code_dim]
        
        # Use transformer decoder with memory
        x_transformer = self.transformer_decoder(x_quantized.permute(0, 2, 1), memory)
        
        # Use convolutional decoder for temporal upsampling
        x_decoder = self.conv_decoder(x_transformer.permute(0, 2, 1))
        x_out = x_decoder.permute(0, 2, 1)
        return x_out, commit_loss, perplexity, code_idx

    def motion_encode(self, features):
        # Preprocess
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        code_idx = (self.quantizer.quantize(self.quantizer.preprocess(x_encoder))).reshape(features.shape[0], -1)
        return code_idx
    
    def motion_decode(self, code_idx, heightmap, contact_maps):
        B, T = code_idx.shape  
        x_quantized = self.quantizer.dequantize(code_idx)
        x_quantized = x_quantized.view(B, -1, self.code_dim).contiguous()
        
        # Project and prepare memory
        heightmap_emb = self.heightmap_proj(
            heightmap.reshape(heightmap.shape[0], heightmap.shape[1], -1)
        )  # [B, T, code_dim]
        
        contact_emb = self.contact_proj(
            contact_maps
        )  # [B, T, code_dim]
        
        # Generate FiLM parameters from contact information
        film_params = self.film_generator(contact_emb)  # [B, T, code_dim * 2]
        scale, shift = film_params.chunk(2, dim=-1)  # Each [B, T, code_dim]
        
        # Apply FiLM modulation to heightmap features
        memory = heightmap_emb * (1 + scale) + shift  # [B, T, code_dim]
        
        # Use transformer decoder with memory
        x_transformer = self.transformer_decoder(x_quantized.permute(0, 2, 1), memory)
        
        # Use convolutional decoder for temporal upsampling
        x_decoder = self.conv_decoder(x_transformer)
        x_out = x_decoder.permute(0, 2, 1)
        return x_out

class VQVAE2(nn.Module):
    def __init__(self,
                 input_feats: int,
                 output_feats: int,
                 code_num_t=512,
                 code_dim_t=256,
                 code_num_b=512,
                 code_dim_b=256,
                 output_emb_width=256,
                 down_t=3,
                 stride_t=2,
                 width=256,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "silu",
                 gpu_id: list = [0],
                 **kwargs) -> None:
        super().__init__()
        # Top-level (coarse)
        self.encoder_t = Encoder(input_feats, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.quantizer_t = QuantizeEMAReset(code_num_t, code_dim_t, mu=0.99, gpu_id=gpu_id)
        self.decoder_t = Decoder(output_emb_width, code_dim_t, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.upsample_t = nn.Upsample(scale_factor=2, mode='nearest')
        # Bottom-level (fine)
        self.encoder_b = Encoder(output_emb_width + output_emb_width, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.quantizer_b = QuantizeEMAReset(code_num_b, code_dim_b, mu=0.99, gpu_id=gpu_id)
        self.decoder_b = Decoder(output_feats, code_dim_b, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    def forward(self, features):
        # Top-level encoding and quantization
        x_in = self.preprocess(features)
        enc_t = self.encoder_t(x_in)
        x_quantized_t, loss_t, perplexity_t, code_idx_t = self.quantizer_t(enc_t)
        dec_t = self.decoder_t(x_quantized_t)
        upsample_t = self.upsample_t(dec_t)

        # Bottom-level encoding and quantization
        enc_b = self.encoder_b(torch.cat([upsample_t, x_in], dim=1))
        x_quantized_b, loss_b, perplexity_b, code_idx_b = self.quantizer_b(enc_b)
        dec_b = self.decoder_b(x_quantized_b)
        x_out = self.postprocess(dec_b)

        return x_out, (loss_t + loss_b), (perplexity_t, perplexity_b), (code_idx_t, code_idx_b)

class VQVAE2_heightmap_contact_transformer(nn.Module):
    def __init__(self,
                 input_feats: int,
                 output_feats: int,
                 code_num_t=512,
                 code_dim_t=256,
                 code_num_b=512,
                 code_dim_b=256,
                 output_emb_width=256,
                 down_t=3,
                 stride_t=2,
                 width=256,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "silu",
                 gpu_id: list = [0],
                 **kwargs) -> None:
        super().__init__()
        # Top-level (coarse)
        self.encoder_t = Encoder(input_feats, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.quantizer_t = QuantizeEMAReset(code_num_t, code_dim_t, mu=0.99, gpu_id=gpu_id)
        self.decoder_t = Decoder(output_emb_width, code_dim_t, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.upsample_t = nn.Upsample(scale_factor=2, mode='nearest')
        # Projections for heightmap and contact maps
        self.heightmap_proj = nn.Sequential(
            nn.Linear(kwargs['scene_dim'], code_dim_b),
            nn.SiLU(),
            nn.Linear(code_dim_b, code_dim_b)
        )
        self.contact_proj = nn.Sequential(
            nn.Linear(2000, code_dim_b),
            nn.SiLU(),
            nn.Linear(code_dim_b, code_dim_b)
        )
        # FiLM generator for contact conditioning
        self.film_generator = nn.Sequential(
            nn.Linear(code_dim_b, code_dim_b * 2),
            nn.SiLU(),
            nn.Linear(code_dim_b * 2, code_dim_b * 2)
        )
        # Bottom-level transformer conditioning
        self.transformer_cond = TransformerHeightmapDecoder(
            input_dim=output_emb_width + output_emb_width,  # upsampled_t + input
            output_dim=code_dim_b,
            num_heads=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.2,
            activation=activation
        )
        self.quantizer_b = QuantizeEMAReset(code_num_b, code_dim_b, mu=0.99, gpu_id=gpu_id)
        self.decoder_b = Decoder(output_feats, code_dim_b, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    def forward(self, features, heightmap, contact_maps):
        # Top-level encoding and quantization
        x_in = self.preprocess(features)
        enc_t = self.encoder_t(x_in)
        x_quantized_t, loss_t, perplexity_t, code_idx_t = self.quantizer_t(enc_t)
        dec_t = self.decoder_t(x_quantized_t)
        upsample_t = self.upsample_t(dec_t)

        # Bottom-level encoding and quantization
        concat_b = torch.cat([upsample_t, x_in], dim=1)  # [B, C, T]
        enc_b = self.encoder_b(concat_b)
        x_quantized_b, loss_b, perplexity_b, code_idx_b = self.quantizer_b(enc_b)

        # Project heightmap and contact maps for conditioning
        heightmap_emb = self.heightmap_proj(heightmap.reshape(heightmap.shape[0], heightmap.shape[1], -1))
        contact_emb = self.contact_proj(contact_maps)
        film_params = self.film_generator(contact_emb)
        scale, shift = film_params.chunk(2, dim=-1)
        memory = heightmap_emb * (1 + scale) + shift

        # Bottom-level transformer conditioning (now after quantization)
        cond_b = self.transformer_cond(x_quantized_b, memory)

        # Bottom-level decoding
        dec_b = self.decoder_b(cond_b)
        x_out = self.postprocess(dec_b)

        return x_out, (loss_t + loss_b), (perplexity_t, perplexity_b), (code_idx_t, code_idx_b)