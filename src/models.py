"""Model architectures. One encoder class per modality; fusion classes added in notebook 6."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock
from torch_geometric.utils import scatter, softmax, to_dense_batch

from src.featurizers import PAD


def dense_radius_graph(pos: torch.Tensor, batch: torch.Tensor, r: float) -> torch.Tensor:
    """Pure-PyTorch radius graph (no pyg-lib/torch-cluster dependency).

    Molecules here are small (tens of atoms), so a dense pairwise-distance
    approach is simple and fast enough, and avoids pyg-lib's fragile Windows
    install story entirely.
    """
    dense_pos, mask = to_dense_batch(pos, batch)  # (B, N_max, 3), (B, N_max)
    B, N, _ = dense_pos.shape

    dist = torch.cdist(dense_pos, dense_pos)                    # (B, N, N)
    valid = mask.unsqueeze(1) & mask.unsqueeze(2)                # both endpoints real atoms
    eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
    adj = (dist <= r) & valid & (~eye)                           # within cutoff, not self-loop

    b_idx, i_idx, j_idx = adj.nonzero(as_tuple=True)
    counts = mask.sum(dim=1)                                     # atoms per graph
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])  # flat-index offset per graph
    row = offsets[b_idx] + i_idx
    col = offsets[b_idx] + j_idx
    return torch.stack([row, col], dim=0)


class SelfiesTransformerEncoder(nn.Module):
    """Token embedding + learned positional embedding + Transformer encoder + mean-pool.

    Mean-pooling (over non-pad positions) is used instead of a [CLS] token since
    there's no pretraining objective here to make a [CLS] token meaningful.
    """

    def __init__(self, vocab_size, pad_id, d_model=128, n_heads=4, n_layers=4, d_ff=256, dropout=0.1, max_len=128):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_dim = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        pad_mask = input_ids == self.pad_id  # True where padded -> tells attention to ignore these
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (batch, seq_len, d_model)

        # mean-pool over non-pad tokens only
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled  # (batch, d_model)


class ClassifierHead(nn.Module):
    """Small MLP head shared by every single-modality and multimodal model in this project."""

    def __init__(self, in_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,) raw logit


class Selfies1DModel(nn.Module):
    """Standalone 1D-only model: SELFIES encoder + classifier head."""

    def __init__(self, vocab_size, pad_id, **encoder_kwargs):
        super().__init__()
        self.encoder = SelfiesTransformerEncoder(vocab_size, pad_id, **encoder_kwargs)
        self.head = ClassifierHead(self.encoder.out_dim)

    def forward(self, input_ids):
        emb = self.encoder(input_ids)
        return self.head(emb)


class GINE2DEncoder(nn.Module):
    """GINEConv stack: message passing with edge features folded in, mean+max pooled.

    Each layer lets an atom aggregate info from bonded neighbors -> after
    n_layers, each atom's representation reflects its n_layers-bond neighborhood.
    """

    def __init__(self, atom_feat_dim, bond_feat_dim, hidden_dim=128, n_layers=4, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(atom_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINEConv(mlp, edge_dim=bond_feat_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2  # mean+max concatenated

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = torch.relu(h)
            h = self.dropout(h)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        return pooled


class Graph2DModel(nn.Module):
    """Standalone 2D-only model: GINEConv encoder + classifier head."""

    def __init__(self, atom_feat_dim, bond_feat_dim, **encoder_kwargs):
        super().__init__()
        self.encoder = GINE2DEncoder(atom_feat_dim, bond_feat_dim, **encoder_kwargs)
        self.head = ClassifierHead(self.encoder.out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        emb = self.encoder(x, edge_index, edge_attr, batch)
        return self.head(emb)


class SchNet3DEncoder(nn.Module):
    """Per-conformer 3D encoder built from SchNet's own official building blocks
    (GaussianSmearing distance expansion + InteractionBlock continuous-filter
    convolutions), stopping before SchNet's final scalar-readout MLP since we
    want an embedding per conformer, not a scalar prediction directly.
    """

    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=3,
                 num_gaussians=50, cutoff=10.0, max_z=100):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_interactions)
        ])
        self.out_dim = hidden_channels

    def forward(self, atom_z, atom_pos, atom_conf_batch):
        # atom_conf_batch: which (flattened) conformer each atom belongs to
        edge_index = dense_radius_graph(atom_pos, atom_conf_batch, r=self.cutoff)
        row, col = edge_index
        edge_weight = (atom_pos[row] - atom_pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        h = self.embedding(atom_z)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return global_add_pool(h, atom_conf_batch)  # (n_conformers_in_batch, hidden_channels)


class BoltzmannAggregator(nn.Module):
    """Combines per-conformer embeddings into one per-molecule embedding.

    mode='uniform_mean'      -> plain average, ignores energies (control, to prove Boltzmann weighting matters)
    mode='weighted_mean'     -> direct Boltzmann-weighted average (baseline)
    mode='learned_attention' -> learned logits + Boltzmann log-weights as a prior bias, softmax per molecule
                                 (satisfies "learnable attention informed by Boltzmann probabilities")
    """

    def __init__(self, embed_dim, mode="weighted_mean"):
        super().__init__()
        assert mode in ("uniform_mean", "weighted_mean", "learned_attention")
        self.mode = mode
        if mode == "learned_attention":
            self.attn_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1)
            )

    def forward(self, conf_embeddings, conf_mol_batch, conf_boltzmann_weights, num_mols):
        if self.mode == "uniform_mean":
            ones = torch.ones_like(conf_boltzmann_weights)
            counts = scatter(ones, conf_mol_batch, dim=0, dim_size=num_mols, reduce="sum").clamp(min=1)
            summed = scatter(conf_embeddings, conf_mol_batch, dim=0, dim_size=num_mols, reduce="sum")
            return summed / counts.unsqueeze(-1)

        if self.mode == "weighted_mean":
            weighted = conf_embeddings * conf_boltzmann_weights.unsqueeze(-1)
            return scatter(weighted, conf_mol_batch, dim=0, dim_size=num_mols, reduce="sum")

        # learned_attention: Boltzmann weights enter as a log-prior bias on learned attention logits
        logits = self.attn_mlp(conf_embeddings).squeeze(-1)
        prior = torch.log(conf_boltzmann_weights.clamp(min=1e-8))
        attn = softmax(logits + prior, conf_mol_batch, num_nodes=num_mols)
        weighted = conf_embeddings * attn.unsqueeze(-1)
        return scatter(weighted, conf_mol_batch, dim=0, dim_size=num_mols, reduce="sum")


class Conformer3DModel(nn.Module):
    """Standalone 3D-only model: SchNet per-conformer encoder + Boltzmann aggregation + classifier head."""

    def __init__(self, agg_mode="weighted_mean", **schnet_kwargs):
        super().__init__()
        self.encoder = SchNet3DEncoder(**schnet_kwargs)
        self.aggregator = BoltzmannAggregator(self.encoder.out_dim, mode=agg_mode)
        self.head = ClassifierHead(self.encoder.out_dim)
        self.out_dim = self.encoder.out_dim

    def forward(self, atom_z, atom_pos, atom_conf_batch, conf_mol_batch, conf_weights, num_mols):
        conf_emb = self.encoder(atom_z, atom_pos, atom_conf_batch)
        mol_emb = self.aggregator(conf_emb, conf_mol_batch, conf_weights, num_mols)
        return self.head(mol_emb)


class GatedFusion(nn.Module):
    """Learns per-molecule trust weights across the three modalities before combining them.

    Projects each modality embedding to a shared fusion_dim, computes a softmax gate over
    the three modalities from their concatenation (so the gate itself sees all three at
    once -- this is the cross-modal interaction), then combines as a gated sum. This is
    the "more meaningful multimodal interaction" beyond the concat baseline.
    """

    def __init__(self, dim_1d, dim_2d, dim_3d, fusion_dim=128):
        super().__init__()
        self.proj_1d = nn.Linear(dim_1d, fusion_dim)
        self.proj_2d = nn.Linear(dim_2d, fusion_dim)
        self.proj_3d = nn.Linear(dim_3d, fusion_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(dim_1d + dim_2d + dim_3d, fusion_dim), nn.ReLU(), nn.Linear(fusion_dim, 3)
        )
        self.out_dim = fusion_dim

    def forward(self, e1d, e2d, e3d):
        gates = torch.softmax(self.gate_net(torch.cat([e1d, e2d, e3d], dim=-1)), dim=-1)  # (batch, 3)
        p1, p2, p3 = self.proj_1d(e1d), self.proj_2d(e2d), self.proj_3d(e3d)
        return gates[:, 0:1] * p1 + gates[:, 1:2] * p2 + gates[:, 2:3] * p3


class MultimodalModel(nn.Module):
    """Combines SELFIES Transformer, GINEConv graph, and SchNet+Boltzmann 3D encoders.

    fusion_type='concat' -> simple concatenation baseline (required ablation row)
    fusion_type='gated'  -> GatedFusion (the "more meaningful interaction" requirement)
    """

    def __init__(self, vocab_size, pad_id, selfies_kwargs, graph_kwargs, schnet_kwargs,
                 fusion_type="concat", fusion_dim=128):
        super().__init__()
        self.enc1d = SelfiesTransformerEncoder(vocab_size, pad_id, **selfies_kwargs)
        self.enc2d = GINE2DEncoder(**graph_kwargs)
        agg_mode = schnet_kwargs.pop("agg_mode", "weighted_mean")
        self.enc3d = SchNet3DEncoder(**schnet_kwargs)
        self.agg3d = BoltzmannAggregator(self.enc3d.out_dim, mode=agg_mode)

        self.fusion_type = fusion_type
        if fusion_type == "concat":
            head_in = self.enc1d.out_dim + self.enc2d.out_dim + self.enc3d.out_dim
        elif fusion_type == "gated":
            self.fusion = GatedFusion(self.enc1d.out_dim, self.enc2d.out_dim, self.enc3d.out_dim, fusion_dim)
            head_in = fusion_dim
        else:
            raise ValueError(fusion_type)
        self.head = ClassifierHead(head_in)

    def forward(self, input_ids, graph_x, graph_edge_index, graph_edge_attr, graph_batch,
                atom_z, atom_pos, atom_conf_batch, conf_mol_batch, conf_weights, conf_num_mols):
        e1d = self.enc1d(input_ids)
        e2d = self.enc2d(graph_x, graph_edge_index, graph_edge_attr, graph_batch)
        conf_emb = self.enc3d(atom_z, atom_pos, atom_conf_batch)
        e3d = self.agg3d(conf_emb, conf_mol_batch, conf_weights, conf_num_mols)

        if self.fusion_type == "concat":
            fused = torch.cat([e1d, e2d, e3d], dim=-1)
        else:
            fused = self.fusion(e1d, e2d, e3d)
        return self.head(fused)
