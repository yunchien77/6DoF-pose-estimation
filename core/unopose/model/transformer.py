import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import numpy as np

from typing import Union, Dict, Optional, Tuple
from einops import rearrange

from core.unopose.utils.model_utils import pairwise_distance
from core.unopose.model.pointnet2.pointnet2_utils import gather_operation

NORM_LAYERS = {
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "InstanceNorm1d": nn.InstanceNorm1d,
    "InstanceNorm2d": nn.InstanceNorm2d,
    "InstanceNorm3d": nn.InstanceNorm3d,
    "GroupNorm": nn.GroupNorm,
    "LayerNorm": nn.LayerNorm,
}


ACT_LAYERS = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "Softplus": nn.Softplus,
    "Tanh": nn.Tanh,
    "Identity": nn.Identity,
}


CONV_LAYERS = {
    "Linear": nn.Linear,
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
}


def _check_block_type(block):
    if block not in ["self", "cross"]:
        raise ValueError('Unsupported block type "{}".'.format(block))


def parse_cfg(cfg: Union[str, Dict]) -> Tuple[str, Dict]:
    assert isinstance(cfg, (str, Dict)), "Illegal cfg type: {}.".format(type(cfg))
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    else:
        cfg = cfg.copy()
    layer = cfg.pop("type")
    return layer, cfg


def build_dropout_layer(p: Optional[float], **kwargs) -> nn.Module:
    r"""Factory function for dropout layer."""
    if p is None or p == 0:
        return nn.Identity()
    else:
        return nn.Dropout(p=p, **kwargs)


def build_norm_layer(num_features, norm_cfg: Optional[Union[str, Dict]]) -> nn.Module:
    r"""Factory function for normalization layers."""
    if norm_cfg is None:
        return nn.Identity()
    layer, kwargs = parse_cfg(norm_cfg)
    assert layer in NORM_LAYERS, f"Illegal normalization: {layer}."
    if layer == "GroupNorm":
        kwargs["num_channels"] = num_features
    elif layer == "LayerNorm":
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    return NORM_LAYERS[layer](**kwargs)


def build_act_layer(act_cfg: Optional[Union[str, Dict]]) -> nn.Module:
    r"""Factory function for activation functions."""
    if act_cfg is None:
        return nn.Identity()
    layer, kwargs = parse_cfg(act_cfg)
    assert layer in ACT_LAYERS, f"Illegal activation: {layer}."
    if layer == "LeakyReLU":
        if "negative_slope" not in kwargs:
            kwargs["negative_slope"] = 0.2
    return ACT_LAYERS[layer](**kwargs)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("`d_model` ({}) must be a multiple of `num_heads` ({}).".format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(
        self, input_q, input_k, input_v, key_weights=None, key_masks=None, attention_factors=None, attention_masks=None
    ):
        """Vanilla Self-attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, N, C)
            input_k (Tensor): input tensor for key (B, M, C)
            input_v (Tensor): input tensor for value (B, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, H, N, M), attention scores before dropout
        """
        q = rearrange(self.proj_q(input_q), "b n (h c) -> b h n c", h=self.num_heads)
        k = rearrange(self.proj_k(input_k), "b m (h c) -> b h m c", h=self.num_heads)
        v = rearrange(self.proj_v(input_v), "b m (h c) -> b h m c", h=self.num_heads)

        attention_scores = torch.einsum("bhnc,bhmc->bhnm", q, k) / self.d_model_per_head**0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float("-inf"))
        if attention_masks is not None:
            attention_scores = attention_scores.masked_fill(attention_masks, float("-inf"))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, "b h n c -> b n (h c)")

        return hidden_states, attention_scores


class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(AttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class AttentionOutput(nn.Module):
    def __init__(self, d_model, dropout=None, activation_fn="ReLU"):
        super(AttentionOutput, self).__init__()
        self.expand = nn.Linear(d_model, d_model * 2)
        self.activation = build_act_layer(activation_fn)
        self.squeeze = nn.Linear(d_model * 2, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states):
        hidden_states = self.expand(input_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(input_states + hidden_states)
        return output_states


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn="ReLU"):
        super(TransformerLayer, self).__init__()
        self.attention = AttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores


class ConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn="ReLU", return_attention_scores=False):
        super(ConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == "self":
                feats0, scores0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f"Sinusoidal positional encoding with odd d_model: {d_model}")
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings


class GeometricStructureEmbedding(nn.Module):
    def __init__(self, cfg):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = cfg.sigma_d
        self.sigma_a = cfg.sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = cfg.angle_k

        self.embedding = SinusoidalPositionalEmbedding(cfg.hidden_dim)
        self.proj_d = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_a = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.reduction_a = cfg.reduction_a
        if self.reduction_a not in ["max", "mean"]:
            raise ValueError(f"Unsupported reduction mode: {self.reduction_a}.")

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == "max":
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings


class CrossGeometricStructureEmbedding(nn.Module):
    """Like GeometricStructureEmbedding, but for a QUERY point set attending
    to a DIFFERENT key point set (dense-to-sparse cross-attention) rather
    than a set attending to itself. Produces (B, N_query, N_key, hidden_dim)
    pairwise distance+angle embeddings, fed to RPEMultiHeadAttention via
    its embed_qk argument exactly the way GeometricStructureEmbedding
    already feeds GeometricTransformer's sparse self/cross blocks.

    Motivation (see BOTTLENECK_ANALYSIS.md section 4a): the fine stage's
    dense-to-sparse cross-attention (SparseToDenseTransformer.dense_layer)
    currently has NO geometric-structure awareness at all -- only a locally
    rotation-invariant positional encoding (LRF-based, PositionalEncoding in
    oneref_predator_fine_point_matching.py) that by design discards exactly
    the angular information needed to disambiguate points on a locally
    self-similar/rotationally-symmetric patch (verified: the true match's
    wrong-picked competitor shares the true point's height and radius from
    the object's own axis, differing mainly by angle -- "ring confusion").
    Full softmax attention (tested, ruled out with a matched control) made
    ranking worse because it sharpens discrimination among purely local
    CONTENT features but still has no access to non-local geometric context
    to break the tie. This gives the attention a genuinely different signal:
    "point i is at distance d and angle theta from point j", which differs
    even for two points that look content-identical up close.

    Angular reference vectors are drawn from each KEY point's own k nearest
    neighbors within the (smaller, more stable) key set, rather than
    attempting a per-query-point neighbor structure in the much larger
    dense query set -- this mirrors GeometricStructureEmbedding's own
    "neighbors define the local frame" convention, just anchored on the
    side with a stable canonical downsample (the 196-point FPS subset)
    instead of the full 2048-point dense cloud.
    """

    def __init__(self, cfg):
        super(CrossGeometricStructureEmbedding, self).__init__()
        self.sigma_d = cfg.sigma_d
        self.sigma_a = cfg.sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = cfg.angle_k
        # N_query (dense, ~2048) x N_key (sparse, ~196) x angle_k x hidden_dim
        # is >10x the element count of the self-only GeometricStructureEmbedding's
        # N_key x N_key case (coarse/sparse-only, ~196x196) it's modeled on --
        # materializing the full pre-reduction (B,N,M,k,hidden_dim) tensor at
        # once OOMs a 24GB GPU during training (verified). Looping over the
        # query dimension in chunks keeps the SAME math/order of operations
        # (embed indices -> project -> reduce over k, exactly mirroring
        # GeometricStructureEmbedding) but bounds peak memory to one chunk's
        # worth regardless of how many dense query points there are.
        self.query_chunk_size = cfg.get("query_chunk_size", 256)
        # Ablation switch (BOTTLENECK_ANALYSIS.md section 5b follow-up): isolates
        # whether the angle term contributes anything beyond the distance term,
        # now that large_marker's re-verification showed geo-attention's AR gain
        # is NOT coming from fixing ring confusion (the angle term's intended
        # purpose). When True, forward_chunk returns d_embeddings alone --
        # proj_a/the angle computation are skipped entirely, not just masked,
        # so this is a true ablation, not a zero-input pass-through.
        self.disable_angle = cfg.get("disable_angle", False)
        # Complementary ablation: the distance-only probe showed the angle term
        # contributes nothing detectable -- this isolates the mirror question,
        # whether the distance term alone accounts for the whole gain, or the
        # angle term alone can reproduce it too. When True, forward_chunk
        # returns a_embeddings alone -- proj_d/the distance computation are
        # skipped entirely, not just masked. Mutually exclusive with
        # disable_angle (checked in forward_chunk).
        self.disable_distance = cfg.get("disable_distance", False)
        assert not (self.disable_angle and self.disable_distance), \
            "disable_angle and disable_distance are mutually exclusive"

        self.embedding = SinusoidalPositionalEmbedding(cfg.hidden_dim)
        self.proj_d = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_a = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.reduction_a = cfg.reduction_a
        if self.reduction_a not in ["max", "mean"]:
            raise ValueError(f"Unsupported reduction mode: {self.reduction_a}.")

    @torch.no_grad()
    def _key_side_ref_vectors(self, key_points):
        r"""Key-point-only part of the angle computation (no query
        dependency) -- computed once and reused across all query chunks.

        Args:
            key_points: torch.Tensor (B, M, 3)

        Returns:
            ref_vectors: torch.Tensor (B, M, k, 3), each key point's vector
                to its own k nearest neighbors within the key set.
        """
        batch_size, num_key, _ = key_points.shape
        k = self.angle_k
        key_dist_map = torch.sqrt(pairwise_distance(key_points, key_points))  # (B, M, M)
        knn_indices = key_dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, M, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_key, k, 3)  # (B, M, k, 3)
        expanded_key_points = key_points.unsqueeze(1).expand(batch_size, num_key, num_key, 3)  # (B, M, M, 3)
        knn_points = torch.gather(expanded_key_points, dim=2, index=knn_indices)  # (B, M, k, 3)
        ref_vectors = knn_points - key_points.unsqueeze(2)  # (B, M, k, 3), key point -> its own neighbors
        return ref_vectors

    @torch.no_grad()
    def get_embedding_indices_chunk(self, query_chunk, key_points, ref_vectors):
        r"""
        Args:
            query_chunk: torch.Tensor (B, Nc, 3), a slice of the query points
            key_points: torch.Tensor (B, M, 3)
            ref_vectors: torch.Tensor (B, M, k, 3), from _key_side_ref_vectors

        Returns:
            d_indices: torch.FloatTensor (B, Nc, M)
            a_indices: torch.FloatTensor (B, Nc, M, k)
        """
        batch_size, num_query, _ = query_chunk.shape
        num_key = key_points.shape[1]
        k = self.angle_k

        dist_map = torch.sqrt(pairwise_distance(query_chunk, key_points))  # (B, Nc, M)
        d_indices = dist_map / self.sigma_d

        ref_vectors_exp = ref_vectors.unsqueeze(1).expand(batch_size, num_query, num_key, k, 3)  # (B, Nc, M, k, 3)
        anc_vectors = query_chunk.unsqueeze(2) - key_points.unsqueeze(1)  # (B, Nc, M, 3), key point -> query point
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_query, num_key, k, 3)  # (B, Nc, M, k, 3)

        sin_values = torch.linalg.norm(torch.cross(ref_vectors_exp, anc_vectors, dim=-1), dim=-1)  # (B, Nc, M, k)
        cos_values = torch.sum(ref_vectors_exp * anc_vectors, dim=-1)  # (B, Nc, M, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, Nc, M, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward_chunk(self, query_chunk, key_points, ref_vectors):
        """Embedding for ONE query chunk only -- the piece GeoAwareDenseLayer
        calls directly (interleaved with attention, per chunk) so the full
        (B, N, M, hidden_dim) embedding is never materialized/retained at
        once; see forward()'s docstring note for why that matters."""
        d_indices, a_indices = self.get_embedding_indices_chunk(query_chunk, key_points, ref_vectors)

        if not self.disable_distance:
            d_embeddings = self.embedding(d_indices)
            d_embeddings = self.proj_d(d_embeddings)

        if self.disable_angle:
            return d_embeddings

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == "max":
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        if self.disable_distance:
            return a_embeddings

        return d_embeddings + a_embeddings  # (B, Nc, M, hidden_dim)

    def forward(self, query_points, key_points):
        """Convenience full-tensor version (materializes and retains the
        entire (B, N, M, hidden_dim) result) -- fine for the coarse-scale
        self-attention use case this class is modeled on, but NOT what
        GeoAwareDenseLayer uses for the much larger dense-to-sparse case
        (see its forward, which calls forward_chunk directly per query
        chunk, interleaved with attention, to bound peak memory)."""
        num_query = query_points.shape[1]
        ref_vectors = self._key_side_ref_vectors(key_points)
        chunks = [
            self.forward_chunk(query_points[:, start:start + self.query_chunk_size, :], key_points, ref_vectors)
            for start in range(0, num_query, self.query_chunk_size)
        ]
        return torch.cat(chunks, dim=1)  # (B, N, M, hidden_dim)


class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("`d_model` ({}) must be a multiple of `num_heads` ({}).".format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        q = rearrange(self.proj_q(input_q), "b n (h c) -> b h n c", h=self.num_heads)
        k = rearrange(self.proj_k(input_k), "b m (h c) -> b h m c", h=self.num_heads)
        v = rearrange(self.proj_v(input_v), "b m (h c) -> b h m c", h=self.num_heads)
        p = rearrange(self.proj_p(embed_qk), "b n m (h c) -> b h n m c", h=self.num_heads)

        attention_scores_p = torch.einsum("bhnc,bhnmc->bhnm", q, p)
        attention_scores_e = torch.einsum("bhnc,bhmc->bhnm", q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head**0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float("-inf"))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, "b h n c -> b n (h c)")

        return hidden_states, attention_scores


class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            position_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn="ReLU"):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores


class GeometricTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn="ReLU",
        return_attention_scores=False,
        parallel=False,
    ):
        super(GeometricTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == "self":
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def forward(self, feats0, embeddings0, feats1, embeddings1, masks0=None, masks1=None):
        # observed, reference
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == "self":
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class LinearAttention(nn.Module):
    def __init__(self, d_model, num_heads, focusing_factor=3):
        super(LinearAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("`d_model` ({}) must be a multiple of `num_heads` ({}).".format(d_model, num_heads))
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.focusing_factor = focusing_factor
        self.kernel_function = nn.ReLU()

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, self.d_model)))

    def forward(self, input_q, input_k, input_v):
        """
        Args:
            q: dense
            k: sparse
            v: sparse
        """
        q = self.proj_q(input_q)
        k = self.proj_k(input_k)
        v = self.proj_v(input_v)
        scale = nn.Softplus()(self.scale)

        q = self.kernel_function(q) + 1e-6
        k = self.kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q**self.focusing_factor
        k = k**self.focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        return x


class LinearAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=False, focusing_factor=3):
        super(LinearAttentionLayer, self).__init__()
        self.attention = LinearAttention(d_model, num_heads, focusing_factor=focusing_factor)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
    ):
        """
        input states: dense features
        memory states: sparse features
        """
        hidden_states = self.attention(
            input_states,
            memory_states,
            memory_states,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states


class LinearTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn="ReLU", focusing_factor=3):
        super(LinearTransformerLayer, self).__init__()
        self.attention = LinearAttentionLayer(d_model, num_heads, dropout=dropout, focusing_factor=focusing_factor)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(self, input_states, memory_states):
        """
        input states: dense features
        memory states: sparse features
        """
        hidden_states = self.attention(input_states, memory_states)
        output_states = self.output(hidden_states)
        return output_states


class GeoAwareDenseLayer(nn.Module):
    """Geometric-structure-aware alternative to LinearTransformerLayer for
    SparseToDenseTransformer.dense_layer. Computes a CrossGeometricStructureEmbedding
    between the dense query points and the sparse key points, then feeds it
    into RPEAttentionLayer (already used by GeometricTransformer for the
    sparse self/cross blocks, reused verbatim here) so attention scores are
    informed by explicit pairwise distance/angle relationships, not just
    local content similarity. See CrossGeometricStructureEmbedding's
    docstring for the full motivation (ring-confusion / local rotational
    symmetry, BOTTLENECK_ANALYSIS.md section 4a).
    """

    def __init__(self, d_model, num_heads, geo_cfg, dropout=None, activation_fn="ReLU"):
        super(GeoAwareDenseLayer, self).__init__()
        self.geo_embedding = CrossGeometricStructureEmbedding(geo_cfg)
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)
        # Zero-init proj_p so attention_scores_p starts at exactly 0 --
        # RPEMultiHeadAttention.forward computes
        # attention_scores = (attention_scores_e + attention_scores_p) / sqrt(d),
        # so this makes the block start EXACTLY equivalent to content-only
        # attention (matching the old checkpoint's learned behavior) and
        # only gradually learn to weigh in the geometric bias during
        # fine-tuning, rather than corrupting attention with a random
        # geometric term from step 0 (same warm-start philosophy as
        # SpotGuidedAttention's zero-init, applied to the new term instead
        # of the whole block).
        nn.init.zeros_(self.attention.attention.proj_p.weight)
        nn.init.zeros_(self.attention.attention.proj_p.bias)

    def forward(self, input_states, memory_states, query_points, key_points):
        """
        input states: dense features (B, N, C)
        memory states: sparse features (B, M, C)
        query_points: dense point coordinates (B, N, 3)
        key_points: sparse point coordinates (B, M, 3)

        Computes the geo embedding and runs attention PER QUERY CHUNK, each
        wrapped in activation checkpointing (torch.utils.checkpoint):
        chunking the forward computation ALONE does not bound memory --
        autograd still needs to save every chunk's intermediate activations
        for the eventual single backward() call, so N=2048/M=196 OOMs a
        24GB GPU regardless of chunk size (verified: ~6.6-6.9GB per call
        whether chunk_size is 512, 256, or 128). Checkpointing each chunk
        (recompute its forward during backward instead of saving
        activations) is what actually bounds peak memory -- verified
        empirically to cut this same case from ~6.9GB down to ~1.1GB.
        """
        chunk_size = self.geo_embedding.query_chunk_size
        num_query = query_points.shape[1]
        ref_vectors = self.geo_embedding._key_side_ref_vectors(key_points)

        def run_chunk(query_chunk, input_chunk, memory_states, key_points, ref_vectors):
            embed_qk_chunk = self.geo_embedding.forward_chunk(query_chunk, key_points, ref_vectors)
            hidden_chunk, _attention_scores = self.attention(input_chunk, memory_states, embed_qk_chunk)
            return hidden_chunk

        hidden_chunks = []
        for start in range(0, num_query, chunk_size):
            query_chunk = query_points[:, start:start + chunk_size, :]
            input_chunk = input_states[:, start:start + chunk_size, :]
            if self.training:
                hidden_chunk = checkpoint(
                    run_chunk, query_chunk, input_chunk, memory_states, key_points, ref_vectors, use_reentrant=False
                )
            else:
                hidden_chunk = run_chunk(query_chunk, input_chunk, memory_states, key_points, ref_vectors)
            hidden_chunks.append(hidden_chunk)

        hidden_states = torch.cat(hidden_chunks, dim=1)
        output_states = self.output(hidden_states)
        return output_states


class SparseToDenseTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        sparse_blocks,
        num_heads=4,
        dropout=None,
        activation_fn="ReLU",
        parallel=False,
        focusing_factor=3,
        with_bg_token=True,
        replace_bg_token=True,
        dense_attention_type="linear",
        geo_cfg=None,
    ):
        super(SparseToDenseTransformer, self).__init__()
        self.with_bg_token = with_bg_token
        self.replace_bg_token = replace_bg_token
        self.dense_attention_type = dense_attention_type

        self.sparse_layer = GeometricTransformer(
            blocks=sparse_blocks,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            activation_fn=activation_fn,
            parallel=parallel,
            return_attention_scores=False,
        )
        # dense_attention_type="geo" swaps the sparse-to-dense cross-attention
        # from FLatten-style linear attention (content-only, no geometric
        # awareness) to CrossGeometricStructureEmbedding + RPE attention
        # (explicit pairwise distance/angle bias) -- see GeoAwareDenseLayer's
        # docstring for the motivation.
        if dense_attention_type == "linear":
            self.dense_layer = LinearTransformerLayer(d_model, num_heads, focusing_factor=focusing_factor)
        elif dense_attention_type == "geo":
            if geo_cfg is None:
                raise ValueError("dense_attention_type='geo' requires geo_cfg (sigma_d/sigma_a/angle_k/reduction_a/hidden_dim)")
            self.dense_layer = GeoAwareDenseLayer(d_model, num_heads, geo_cfg, activation_fn=activation_fn)
        else:
            raise ValueError(f"Unknown dense_attention_type: {dense_attention_type}")

    def forward(
        self,
        dense_feats0,
        embeddings0,
        fps_idx0,
        dense_feats1,
        embeddings1,
        fps_idx1,
        masks0=None,
        masks1=None,
        points0=None,
        points1=None,
    ):
        # visual feats, geo feats, fps idx
        feats0 = self._sample_feats(dense_feats0, fps_idx0)
        feats1 = self._sample_feats(dense_feats1, fps_idx1)
        need_points = self.dense_attention_type == "geo"
        sparse_points0 = self._sample_points(points0, fps_idx0) if need_points else None
        sparse_points1 = self._sample_points(points1, fps_idx1) if need_points else None
        feats0, feats1 = self.sparse_layer(feats0, embeddings0, feats1, embeddings1, masks0, masks1)

        dense_feats0 = self._get_dense_feats(dense_feats0, feats0, points0, sparse_points0)
        dense_feats1 = self._get_dense_feats(dense_feats1, feats1, points1, sparse_points1)
        return dense_feats0, dense_feats1

    def _sample_points(self, points, fps_idx):
        # Mirrors _sample_feats' bg_token handling, but the bg token has no
        # real spatial location, and (like the caller's dense_feats0/1) is
        # expected already prepended with a placeholder row at index 0 --
        # mirrors _sample_feats exactly (same gather-then-reprepend order on
        # the SAME already-bg-prepended tensor) so the sampled coordinates
        # line up with whatever point _sample_feats' identical indexing
        # actually selects, rather than introducing a second, possibly
        # inconsistent convention.
        if self.with_bg_token:
            bg_point = points[:, 0:1, :].contiguous()
        sampled = gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
        if self.with_bg_token:
            sampled = torch.cat([bg_point, sampled], dim=1)
        return sampled

    def _sample_feats(self, dense_feats, fps_idx):
        if self.with_bg_token:
            bg_token = dense_feats[:, 0:1, :].contiguous()
        feats = gather_operation(dense_feats.transpose(1, 2).contiguous(), fps_idx)
        feats = feats.transpose(1, 2).contiguous()
        if self.with_bg_token:
            feats = torch.cat([bg_token, feats], dim=1)
        return feats

    def _get_dense_feats(self, dense_feats, feats, points=None, sparse_points=None):
        extra_args = ()
        if self.dense_attention_type == "geo":
            # GeoAwareDenseLayer additionally needs the real (bg-stripped)
            # coordinates of both sides to compute the cross geometric
            # embedding -- LinearTransformerLayer takes no such argument.
            extra_args = (points[:, 1:, :].contiguous(), sparse_points[:, 1:, :].contiguous())
        if self.with_bg_token and self.replace_bg_token:
            bg_token = feats[:, 0:1, :].contiguous()
            dense_feats = self.dense_layer(dense_feats[:, 1:, :].contiguous(), feats[:, 1:, :].contiguous(), *extra_args)
            dense_feats = torch.cat([bg_token, dense_feats], dim=1)
        else:
            dense_feats = self.dense_layer(dense_feats, feats, *extra_args)
        return dense_feats
