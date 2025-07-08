# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             Enhanced MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig
from typing import Optional, List


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = True,
            
            # 🚀 NEW: Enhanced Architectural Features
            attention_dropout: float = 0.0,
            sliding_window: Optional[int] = None,  # Sliding window attention
            use_alibi: bool = False,  # ALiBi positional encoding
            use_geglu: bool = False,  # GeGLU activation
            layer_norm_type: str = 'rms',  # 'rms' or 'layer'
            attention_bias: bool = False,
            mlp_bias: bool = False,
            
            # 🎯 Adaptive Computation
            use_early_exit: bool = False,
            early_exit_threshold: float = 0.8,
            early_exit_layers: Optional[List[int]] = None,
            
            # 🧠 Memory Optimization
            gradient_checkpointing: bool = False,
            use_memory_efficient_attention: bool = True,
            attention_chunk_size: int = 1024,
            
            # 🔄 Advanced Training Features
            layer_wise_lr_decay: float = 1.0,
            weight_init_method: str = 'normal',  # 'normal', 'xavier', 'kaiming'
            init_std: float = 0.02,
            
            ####################################################
            # Enhanced MOE Configuration
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            # 🎯 NEW: Enhanced MoE features
            moe_load_balancing: bool = True,
            moe_capacity_factor: float = 1.5,
            moe_eval_capacity_factor: float = 2.0,
            moe_min_capacity: int = 4,
            **kwargs
    ):
        super().__init__(**kwargs)
        
        # Basic configuration
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        
        # 🚀 Enhanced features
        self.sliding_window = sliding_window
        self.use_alibi = use_alibi
        self.use_geglu = use_geglu
        self.layer_norm_type = layer_norm_type
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        
        # 🎯 Adaptive computation
        self.use_early_exit = use_early_exit
        self.early_exit_threshold = early_exit_threshold
        self.early_exit_layers = early_exit_layers or []
        
        # 🧠 Memory optimization
        self.gradient_checkpointing = gradient_checkpointing
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.attention_chunk_size = attention_chunk_size
        
        # 🔄 Training features
        self.layer_wise_lr_decay = layer_wise_lr_decay
        self.weight_init_method = weight_init_method
        self.init_std = init_std
        
        # Enhanced MOE
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.moe_load_balancing = moe_load_balancing
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_eval_capacity_factor = moe_eval_capacity_factor
        self.moe_min_capacity = moe_min_capacity


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             Enhanced MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class EnhancedRMSNorm(nn.Module):
    """Enhanced RMSNorm with optional bias and different computation modes"""
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self.weight * self._norm(x.float()).type_as(x)
        if self.bias is not None:
            output += self.bias
        return output


class LayerNorm(nn.Module):
    """Standard Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x):
        output = F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)
        return output


def get_norm_layer(norm_type: str, dim: int, eps: float = 1e-5, bias: bool = False):
    """Factory function for normalization layers"""
    if norm_type == 'rms':
        return EnhancedRMSNorm(dim, eps, bias)
    elif norm_type == 'layer':
        return LayerNorm(dim, eps, bias)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class ALiBi(nn.Module):
    """Attention with Linear Biases (ALiBi) position encoding"""
    def __init__(self, num_heads: int, max_seq_len: int = 8192):
        super().__init__()
        self.num_heads = num_heads
        
        # Generate ALiBi slopes
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]
            
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + \
                       get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
        
        slopes = torch.tensor(get_slopes(num_heads))
        self.register_buffer('slopes', slopes)
        
        # Pre-compute bias matrix
        position_ids = torch.arange(max_seq_len).unsqueeze(0) - torch.arange(max_seq_len).unsqueeze(1)
        alibi_bias = position_ids * slopes.unsqueeze(-1).unsqueeze(-1)
        self.register_buffer('alibi_bias', alibi_bias)
    
    def forward(self, seq_len: int):
        return self.alibi_bias[:, :seq_len, :seq_len]


class GeGLU(nn.Module):
    """GeGLU activation function: x * GELU(Wx + b) ⊙ (Vx + c)"""
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        gate = self.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return gate * up


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention for efficient long sequence processing"""
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.config = args
        self.sliding_window = args.sliding_window
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # Projections
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=args.attention_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(args.attention_dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        
        # Flash attention flag
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        
        # ALiBi position encoding
        if args.use_alibi:
            self.alibi = ALiBi(args.num_attention_heads)
        
        # Memory efficient attention
        self.use_memory_efficient = args.use_memory_efficient_attention
        self.chunk_size = args.attention_chunk_size

    def _apply_sliding_window_mask(self, attention_scores, seq_len):
        """Apply sliding window mask to attention scores"""
        if self.sliding_window is None:
            return attention_scores
        
        # Create sliding window mask
        window_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        window_mask += torch.tril(torch.ones(seq_len, seq_len), diagonal=-self.sliding_window)
        window_mask = window_mask.bool().to(attention_scores.device)
        
        attention_scores = attention_scores.masked_fill(window_mask, float('-inf'))
        return attention_scores

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None):
        
        bsz, seq_len, _ = x.shape
        
        # Project to Q, K, V
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # Apply rotary position embedding if not using ALiBi
        if position_embeddings is not None and not self.config.use_alibi:
            cos, sin = position_embeddings
            xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # Handle KV cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        past_kv = (xk, xv) if use_cache else None
        
        # Repeat KV heads for multi-head attention
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        
        # Reshape for attention computation
        xq = xq.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Compute attention
        if self.flash and seq_len > 1:
            # Use Flash Attention
            output = self._flash_attention(xq, xk, xv, attention_mask)
        else:
            # Standard attention with optimizations
            output = self._standard_attention(xq, xk, xv, attention_mask)
        
        # Reshape and project output
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        
        return output, past_kv
    
    def _flash_attention(self, q, k, v, attention_mask):
        """Flash attention implementation"""
        dropout_p = self.config.attention_dropout if self.training else 0.0
        
        # Prepare attention mask for Flash Attention
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.bool()
        
        # Apply sliding window if specified
        is_causal = True
        if self.sliding_window is not None:
            is_causal = False  # We'll handle causality with sliding window
        
        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
        
        return output
    
    def _standard_attention(self, q, k, v, attention_mask):
        """Standard attention implementation with optimizations"""
        bsz, n_heads, seq_len, head_dim = q.shape
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply ALiBi bias if enabled
        if self.config.use_alibi:
            alibi_bias = self.alibi(seq_len)
            scores += alibi_bias.unsqueeze(0)
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.to(scores.device), float('-inf'))
        
        # Apply sliding window mask
        scores = self._apply_sliding_window_mask(scores, seq_len)
        
        # Apply attention mask
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -1e9
            scores += extended_mask
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output


class EnhancedFeedForward(nn.Module):
    """Enhanced Feed-Forward Network with multiple activation options"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        
        # Calculate intermediate size
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        # Choose activation function
        if config.use_geglu:
            self.activation = GeGLU(config.hidden_size, config.intermediate_size, config.mlp_bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        else:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
            self.act_fn = ACT2FN[config.hidden_act]
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.config.use_geglu:
            activated = self.activation(x)
            return self.dropout(self.down_proj(activated))
        else:
            gate = self.act_fn(self.gate_proj(x))
            up = self.up_proj(x)
            return self.dropout(self.down_proj(gate * up))


class EarlyExitClassifier(nn.Module):
    """Early exit classifier for adaptive computation"""
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.confidence = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        logits = self.classifier(hidden_states)
        confidence = torch.sigmoid(self.confidence(hidden_states))
        return logits, confidence


class EnhancedMiniMindBlock(nn.Module):
    """Enhanced transformer block with modern improvements"""
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        
        # Attention mechanism
        self.self_attn = SlidingWindowAttention(config)
        
        # Normalization layers
        self.input_layernorm = get_norm_layer(
            config.layer_norm_type, config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = get_norm_layer(
            config.layer_norm_type, config.hidden_size, config.rms_norm_eps
        )
        
        # Feed-forward network
        if config.use_moe:
            self.mlp = MOEFeedForward(config)
        else:
            self.mlp = EnhancedFeedForward(config)
        
        # Early exit classifier
        if config.use_early_exit and layer_id in config.early_exit_layers:
            self.early_exit = EarlyExitClassifier(config.hidden_size, config.vocab_size)
        else:
            self.early_exit = None

    def forward(self, 
                hidden_states: torch.Tensor,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None):
        
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention with gradient checkpointing if enabled
        if self.config.gradient_checkpointing and self.training:
            attn_output, present_key_value = checkpoint(
                self.self_attn,
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask
            )
        else:
            attn_output, present_key_value = self.self_attn(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask
            )
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Feed-forward with gradient checkpointing if enabled
        if self.config.gradient_checkpointing and self.training:
            mlp_output = checkpoint(self.mlp, hidden_states)
        else:
            mlp_output = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + mlp_output
        
        # Early exit logic
        early_exit_output = None
        if self.early_exit is not None:
            early_exit_output = self.early_exit(hidden_states)
        
        return hidden_states, present_key_value, early_exit_output


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embedding to query and key tensors"""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key-value tensors for grouped query attention"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """Precompute rotary position embedding frequencies"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


# Import existing MOE classes (keeping them as they are well-implemented)
# Continuing from where we left off...

class MoEGate(nn.Module):
    """Enhanced MoE Gate with improved load balancing"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        
        # Enhanced load balancing
        self.load_balancing = config.moe_load_balancing
        self.capacity_factor = config.moe_capacity_factor
        
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        
        # Load balancing components
        if self.load_balancing:
            self.register_buffer('expert_counts', torch.zeros(self.n_routed_experts))
            self.register_buffer('total_tokens', torch.tensor(0.0))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize gate weights using Xavier uniform"""
        nn.init.xavier_uniform_(self.weight)

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        
        # Compute gating logits
        logits = F.linear(hidden_states, self.weight, None)
        
        # Apply scoring function
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'Unsupported scoring function: {self.scoring_func}')
        
        # Select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        # Normalize top-k probabilities
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        
        # Compute auxiliary loss for load balancing
        aux_loss = 0.0
        if self.training and self.alpha > 0.0:
            aux_loss = self._compute_aux_loss(scores, topk_idx, bsz, seq_len)
        
        # Update expert usage statistics for load balancing
        if self.load_balancing and self.training:
            self._update_expert_stats(topk_idx)
        
        return topk_idx, topk_weight, aux_loss
    
    def _compute_aux_loss(self, scores, topk_idx, bsz, seq_len):
        """Compute auxiliary loss for load balancing"""
        aux_topk = self.top_k
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        
        if self.seq_aux:
            # Sequence-level auxiliary loss
            scores_for_seq_aux = scores.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, device=scores.device)
            ce.scatter_add_(1, topk_idx_for_aux_loss,
                           torch.ones(bsz, seq_len * aux_topk, device=scores.device))
            ce = ce.div_(seq_len * aux_topk / self.n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
        else:
            # Token-level auxiliary loss
            mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
            ce = mask_ce.float().mean(0)
            Pi = scores.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
        
        return aux_loss
    
    def _update_expert_stats(self, topk_idx):
        """Update expert usage statistics for load balancing"""
        flat_idx = topk_idx.view(-1)
        expert_counts = torch.bincount(flat_idx, minlength=self.n_routed_experts).float()
        self.expert_counts += expert_counts
        self.total_tokens += flat_idx.size(0)
    
    def get_load_balancing_loss(self):
        """Get load balancing loss based on expert usage statistics"""
        if not self.load_balancing or self.total_tokens == 0:
            return 0.0
        
        # Compute load balancing loss
        target_count = self.total_tokens / self.n_routed_experts
        expert_usage = self.expert_counts / self.total_tokens
        target_usage = torch.ones_like(expert_usage) / self.n_routed_experts
        
        # Compute KL divergence loss
        kl_loss = F.kl_div(expert_usage.log(), target_usage, reduction='sum')
        return kl_loss * self.alpha


class MOEFeedForward(nn.Module):
    """Enhanced Mixture of Experts Feed-Forward Network"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        
        # Routed experts
        self.experts = nn.ModuleList([
            EnhancedFeedForward(config) for _ in range(config.n_routed_experts)
        ])
        
        # Shared experts (always active)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                EnhancedFeedForward(config) for _ in range(config.n_shared_experts)
            ])
        
        # Gating mechanism
        self.gate = MoEGate(config)
        
        # Capacity and load balancing
        self.capacity_factor = config.moe_capacity_factor
        self.eval_capacity_factor = config.moe_eval_capacity_factor
        self.min_capacity = config.moe_min_capacity
        
        self.aux_loss = 0.0

    def forward(self, x):
        """
        Forward pass through MoE layer
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of same shape as input
        """
        identity = x
        orig_shape = x.shape
        bsz, seq_len, hidden_size = x.shape
        
        # Flatten input for expert routing
        x_flat = x.view(-1, hidden_size)
        
        # Get expert routing decisions
        topk_idx, topk_weight, aux_loss = self.gate(x)
        self.aux_loss = aux_loss
        
        # Compute expert outputs
        if self.training:
            output = self._moe_forward_train(x_flat, topk_idx, topk_weight)
        else:
            output = self._moe_forward_inference(x_flat, topk_idx, topk_weight)
        
        # Reshape output
        output = output.view(orig_shape)
        
        # Add shared expert outputs
        if self.config.n_shared_experts > 0:
            for shared_expert in self.shared_experts:
                output = output + shared_expert(identity)
        
        return output
    
    def _moe_forward_train(self, x, topk_idx, topk_weight):
        """Training-time MoE forward pass with capacity constraints"""
        num_tokens = x.size(0)
        
        # Calculate capacity per expert
        capacity = max(
            self.min_capacity,
            int(self.capacity_factor * num_tokens / self.n_routed_experts)
        )
        
        # Prepare output tensor
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.n_routed_experts):
            # Find tokens assigned to this expert
            expert_mask = (topk_idx == expert_idx).any(dim=-1)
            expert_tokens = x[expert_mask]
            
            if expert_tokens.size(0) == 0:
                continue
            
            # Apply capacity constraint
            if expert_tokens.size(0) > capacity:
                # Randomly sample tokens within capacity
                indices = torch.randperm(expert_tokens.size(0))[:capacity]
                expert_tokens = expert_tokens[indices]
                expert_mask_indices = torch.where(expert_mask)[0][indices]
            else:
                expert_mask_indices = torch.where(expert_mask)[0]
            
            # Process tokens through expert
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Get weights for this expert
            expert_weights = topk_weight[expert_mask_indices]
            expert_weights = expert_weights[topk_idx[expert_mask_indices] == expert_idx]
            
            # Weight and accumulate outputs
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            output[expert_mask_indices] += weighted_output
        
        return output
    
    def _moe_forward_inference(self, x, topk_idx, topk_weight):
        """Inference-time MoE forward pass (optimized)"""
        # Use the existing optimized inference implementation
        return self._moe_infer_optimized(x, topk_idx.view(-1), topk_weight.view(-1, 1))
    
    @torch.no_grad()
    def _moe_infer_optimized(self, x, flat_expert_indices, flat_expert_weights):
        """Optimized inference implementation"""
        expert_cache = torch.zeros_like(x)
        
        # Sort by expert indices for efficient processing
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # Process each expert's tokens in batch
        for expert_idx, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_idx == 0 else tokens_per_expert[expert_idx - 1]
            
            if start_idx == end_idx:
                continue
            
            # Get expert and token indices
            expert = self.experts[expert_idx]
            exp_token_idx = token_idxs[start_idx:end_idx]
            
            # Process tokens
            expert_tokens = x[exp_token_idx]
            expert_output = expert(expert_tokens).to(expert_cache.dtype)
            
            # Apply weights
            expert_weights = flat_expert_weights[idxs[start_idx:end_idx]]
            expert_output = expert_output * expert_weights
            
            # Accumulate in cache
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_output
            )
        
        return expert_cache


class EnhancedMiniMindModel(nn.Module):
    """Enhanced MiniMind model with advanced features"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EnhancedMiniMindBlock(layer_id, config) 
            for layer_id in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = get_norm_layer(
            config.layer_norm_type, config.hidden_size, config.rms_norm_eps
        )
        
        # Position embeddings (if not using ALiBi)
        if not config.use_alibi:
            head_dim = config.hidden_size // config.num_attention_heads
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=head_dim,
                end=config.max_position_embeddings,
                theta=config.rope_theta
            )
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
        # Early exit classifiers
        if config.use_early_exit:
            self.early_exit_classifiers = nn.ModuleList([
                EarlyExitClassifier(config.hidden_size, config.vocab_size)
                for _ in config.early_exit_layers
            ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            if self.config.weight_init_method == 'normal':
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            elif self.config.weight_init_method == 'xavier':
                torch.nn.init.xavier_uniform_(module.weight)
            elif self.config.weight_init_method == 'kaiming':
                torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                output_early_exits: bool = False,
                **kwargs):
        
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # Calculate starting position for KV cache
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Position embeddings (if not using ALiBi)
        position_embeddings = None
        if not self.config.use_alibi:
            position_embeddings = (
                self.freqs_cos[start_pos:start_pos + seq_length],
                self.freqs_sin[start_pos:start_pos + seq_length]
            )
        
        # Process through transformer layers
        presents = []
        early_exits = []
        aux_losses = []
        
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # Forward pass through layer
            layer_output = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            
            hidden_states, present, early_exit_output = layer_output
            presents.append(present)
            
            # Collect early exit outputs
            if early_exit_output is not None and output_early_exits:
                early_exits.append(early_exit_output)
            
            # Collect MoE auxiliary losses
            if hasattr(layer.mlp, 'aux_loss'):
                aux_losses.append(layer.mlp.aux_loss)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Aggregate auxiliary losses
        total_aux_loss = sum(aux_losses) if aux_losses else 0.0
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': presents,
            'aux_loss': total_aux_loss,
            'early_exits': early_exits if output_early_exits else None
        }


class EnhancedMiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """Enhanced MiniMind for Causal Language Modeling"""
    config_class = MiniMindConfig
    
    def __init__(self, config: MiniMindConfig = None):
        if config is None:
            config = MiniMindConfig()
        
        super().__init__(config)
        self.config = config
        
        # Main model
        self.model = EnhancedMiniMindModel(config)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head
        self.model.embed_tokens.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            if self.config.weight_init_method == 'normal':
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            elif self.config.weight_init_method == 'xavier':
                torch.nn.init.xavier_uniform_(module.weight)
            elif self.config.weight_init_method == 'kaiming':
                torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                output_early_exits: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs):
        
        # Forward pass through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_early_exits=output_early_exits,
            **kwargs
        )
        
        hidden_states = outputs['last_hidden_state']
        
        # Compute logits (optionally keep only last few tokens for efficiency)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, torch.Tensor):
            hidden_states = hidden_states[:, logits_to_keep, :]
        
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add auxiliary loss from MoE layers
            if outputs['aux_loss'] is not None and outputs['aux_loss'] > 0:
                loss += outputs['aux_loss']
        
        # Prepare output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs['past_key_values'],
            hidden_states=outputs['last_hidden_state'],
            # Add custom outputs
            aux_loss=outputs['aux_loss'],
            early_exits=outputs['early_exits']
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """Prepare inputs for generation"""
        if past_key_values is not None:
            # Only keep the last token if we have past_key_values
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search"""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


# 🎯 Utility functions for model analysis and optimization
def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """Get model size in megabytes"""
    total_params, _ = count_parameters(model)
    # Assuming float32 (4 bytes per parameter)
    size_mb = total_params * 4 / (1024 * 1024)
    return size_mb


def print_model_summary(model):
    """Print comprehensive model summary"""
    total_params, trainable_params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    print("=" * 60)
    print(f"🚀 Enhanced MiniMind Model Summary")
    print("=" * 60)
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"🎯 Trainable Parameters: {trainable_params:,}")
    print(f"💾 Model Size: {size_mb:.2f} MB")
    print(f"🏗️ Architecture: {model.config.num_hidden_layers} layers")
    print(f"🧠 Hidden Size: {model.config.hidden_size}")
    print(f"🎯 Attention Heads: {model.config.num_attention_heads}")
    print(f"🔑 Key-Value Heads: {model.config.num_key_value_heads}")
    print(f"📚 Vocabulary Size: {model.config.vocab_size}")
    
    # Advanced features
    print("\n🚀 Advanced Features:")
    print(f"  • Flash Attention: {model.config.flash_attn}")
    print(f"  • Sliding Window: {model.config.sliding_window}")
    print(f"  • ALiBi Position Encoding: {model.config.use_alibi}")
    print(f"  • GeGLU Activation: {model.config.use_geglu}")
    print(f"  • Mixture of Experts: {model.config.use_moe}")
    print(f"  • Early Exit: {model.config.use_early_exit}")
    print(f"  • Gradient Checkpointing: {model.config.gradient_checkpointing}")
    print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    # Create enhanced config
    config = MiniMindConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_key_value_heads=4,
        vocab_size=8192,
        max_position_embeddings=4096,
        # Enhanced features
        use_alibi=True,
        use_geglu=True,
        sliding_window=1024,
        use_early_exit=True,
        early_exit_layers=[4, 8],
        gradient_checkpointing=True,
        # MoE features
        use_moe=True,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_load_balancing=True,
    )
    
    # Create model
    model = EnhancedMiniMindForCausalLM(config)
    
    # Print model summary
    print_model_summary(model)
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\n🧪 Testing forward pass with shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_early_exits=True
        )
        
        print(f"✅ Forward pass successful!")
        print(f"📊 Output logits shape: {outputs.logits.shape}")
        print(f"🎯 Auxiliary loss: {outputs.aux_loss}")
        print(f"🚪 Early exits: {len(outputs.early_exits) if outputs.early_exits else 0}")