# VENDORED from ROAD @ c847391 — training/uni3d/models/point_encoder.py
# (originally "Modified from Uni3D: reduced to the frozen point-encoder training path.")
#
# THE ONE SUBSTITUTION in this vendored tree: upstream's farthest_point_sample wraps
# pointnet2_ops (compiled CUDA extension; uninstallable here — cluster policy forbids
# building CUDA extensions into the production env). Replaced with a torch-native FPS
# with the same greedy semantics: start at point 0, then repeatedly take the point
# farthest from the selected set. pointnet2's kernel also starts at index 0. Known
# divergences (independently audited 2026-08-31): (1) floating-point tie-breaking
# order — both remain valid FPS solutions; (2) pointnet2's CUDA kernel never selects
# points with |p|^2 <= 1e-3 (an upstream quirk) — in our [-0.5,0.5] frame that ball
# contains the 8 central voxel centres of the 32^3 grid, which THIS version can
# select and upstream cannot; (3) when a cloud has fewer than `count` DISTINCT
# points, argmax over an all-zero distance field returns index 0 repeatedly (both
# implementations degrade; ~3.7% of the pool has <512 distinct voxels — graceful,
# the teacher just sees duplicated groups).
# Everything below farthest_point_sample is byte-identical to upstream.
import logging

import torch
import torch.nn as nn


def farthest_point_sample(points, count):
    """Greedy FPS, torch-native. points (B, N, 3) -> (B, count, 3)."""
    B, N, _ = points.shape
    device = points.device
    sel = torch.zeros(B, count, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device, dtype=points.dtype)
    cur = torch.zeros(B, dtype=torch.long, device=device)       # start at index 0, like pointnet2
    batch = torch.arange(B, device=device)
    for i in range(count):
        sel[:, i] = cur
        d = ((points - points[batch, cur].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        cur = dist.argmax(-1)
    return points[batch.unsqueeze(1), sel]


def square_distance(source, target):
    distance = -2 * torch.matmul(source, target.transpose(1, 2))
    distance += torch.sum(source.square(), dim=-1).unsqueeze(2)
    distance += torch.sum(target.square(), dim=-1).unsqueeze(1)
    return distance


def knn_point(count, points, centers):
    return torch.topk(
        square_distance(centers, points), count, dim=-1, largest=False, sorted=False
    ).indices


class PatchDropout(nn.Module):
    def __init__(self, probability, exclude_first_token=True):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("Patch dropout probability must be in [0, 1)")
        self.probability = probability
        self.exclude_first_token = exclude_first_token
        logging.info("Uni3D patch dropout probability: %s", probability)

    def forward(self, tokens):
        if not self.training or self.probability == 0:
            return tokens
        if self.exclude_first_token:
            first, tokens = tokens[:, :1], tokens[:, 1:]
        batch_size, token_count = tokens.shape[:2]
        keep_count = max(1, int(token_count * (1 - self.probability)))
        indices = torch.rand(batch_size, token_count, device=tokens.device).topk(
            keep_count, dim=-1
        ).indices
        batch = torch.arange(batch_size, device=tokens.device).unsqueeze(1)
        tokens = tokens[batch, indices]
        return torch.cat([first, tokens], dim=1) if self.exclude_first_token else tokens


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        batch_size, num_points, _ = xyz.shape
        centers = farthest_point_sample(xyz, self.num_group)
        if centers.shape[1] != self.num_group:
            raise ValueError(
                f"Expected {self.num_group} group centers, got {centers.shape[1]}"
            )
        indices = knn_point(self.group_size, xyz, centers)
        indices = indices + (
            torch.arange(batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )
        indices = indices.reshape(-1)
        neighborhoods = xyz.reshape(batch_size * num_points, 3)[indices]
        neighborhoods = neighborhoods.reshape(
            batch_size, self.num_group, self.group_size, 3
        )
        colors = color.reshape(batch_size * num_points, 3)[indices]
        colors = colors.reshape(batch_size, self.num_group, self.group_size, 3)
        neighborhoods = neighborhoods - centers.unsqueeze(2)
        return centers, torch.cat([neighborhoods, colors], dim=-1)


class Encoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, output_dim, 1),
        )

    def forward(self, point_groups):
        batch_size, group_count, point_count, _ = point_groups.shape
        point_groups = point_groups.reshape(batch_size * group_count, point_count, 6)
        features = self.first_conv(point_groups.transpose(1, 2))
        global_features = torch.max(features, dim=2, keepdim=True).values
        features = torch.cat(
            [global_features.expand(-1, -1, point_count), features], dim=1
        )
        features = self.second_conv(features).max(dim=2).values
        return features.reshape(batch_size, group_count, self.output_dim)


class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args):
        super().__init__()
        self.trans_dim = args.pc_feat_dim
        self.embed_dim = args.embed_dim
        self.group_divider = Group(args.num_group, args.group_size)
        self.encoder = Encoder(args.pc_encoder_dim)
        self.encoder2trans = nn.Linear(args.pc_encoder_dim, self.trans_dim)
        self.trans2embed = nn.Linear(self.trans_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )
        self.patch_dropout = (
            PatchDropout(args.patch_dropout)
            if args.patch_dropout > 0
            else nn.Identity()
        )
        self.visual = point_transformer

    def forward(self, points, colors):
        centers, groups = self.group_divider(points, colors)
        encoder_dtype = next(self.encoder.parameters()).dtype
        groups = groups.to(dtype=encoder_dtype)
        centers = centers.to(dtype=self.pos_embed[0].weight.dtype)
        tokens = self.encoder2trans(self.encoder(groups))
        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        positions = torch.cat(
            [
                self.cls_pos.expand(tokens.shape[0], -1, -1),
                self.pos_embed(centers),
            ],
            dim=1,
        )
        tokens = self.patch_dropout(torch.cat([cls_token, tokens], dim=1) + positions)
        tokens = self.visual.pos_drop(tokens)
        for block in self.visual.blocks:
            tokens = block(tokens)
        tokens = self.visual.norm(tokens)
        tokens = self.visual.fc_norm(tokens)
        return self.trans2embed(tokens)
