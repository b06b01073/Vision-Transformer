import torch.nn as nn
import torch
from einops.layers.torch import Rearrange, Reduce
from einops import repeat, rearrange
import torch.nn.functional as F


class LinearProj(nn.Module):
    def __init__(self, patch_size, in_channels, embedded_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, embedded_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )

    def forward(self, x):
        x = self.net(x)
        return x


class PatchEmbedder(nn.Module):
    def __init__(self, num_patches, embedded_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.rand(1, embedded_dim))
        self.pos_embedding = nn.Parameter(torch.rand(num_patches + 1, embedded_dim))
    
    def forward(self, x):
        batch_size = x.shape[0]
        cls = repeat(self.cls_token, 'n e -> b n e', b=batch_size)
        pos_emb = repeat(self.pos_embedding, 'n e -> b n e', b=batch_size)

        x = torch.cat([cls, x], dim=1)
        x += self.pos_embedding

        return x


class TranformerEncoder(nn.Module):
    def __init__(self, embedded_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            EncoderSublayer(MultiAtt(8, embedded_dim), embedded_dim),
            EncoderSublayer(MLP(embedded_dim, hidden_dim), embedded_dim),
        )

    def forward(self, x):
        return self.net(x)


class EncoderSublayer(nn.Module):
    def __init__(self, sublayer, embedded_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedded_dim)
        self.sublayer = sublayer

    def forward(self, x):
        x = x + self.sublayer(x)
        x = self.layer_norm(x)
        
        return x


class MultiAtt(nn.Module):
    def __init__(self, heads, embedded_dim):
        super().__init__()
        self.num_head = heads
        self.scaling = (embedded_dim // heads) ** (-0.5)

        self.pre_proj = nn.Linear(embedded_dim, embedded_dim * 3) # project the input into 3 * embedded_dim, and unpack them into q, k, v
        self.proj = nn.Linear(embedded_dim, embedded_dim)
        

    def forward(self, x):
        x = rearrange(self.pre_proj(x), 'b n (u h e) -> u b h n e', u=3, h=self.num_head)
        q, k, v = x[0], x[1], x[2]

        similarity = torch.einsum('bhqi, bhki -> bhqk', q, k)
        similarity = F.softmax(similarity, dim=-1) * self.scaling

        att_score = torch.einsum('bhnk, bhke -> bhne', similarity, v)
        att_score = rearrange(att_score, 'b h n e -> b n (h e)')
        att_score = self.proj(att_score)

        return att_score



class MLP(nn.Module):
    def __init__(self, embedded_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedded_dim)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, embedded_dim, num_class):
        super().__init__()
        self.net = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(embedded_dim),
            nn.Linear(embedded_dim, num_class)
        )

    def forward(self, x):
        return self.net(x)


class Vit(nn.Module):
    def __init__(self, img_shape, patch_size=7, in_channels=1, embedded_dim=128, encoder_layers=4, num_class=10):
        super().__init__()
        self.img_h, self.img_w = img_shape
        self.num_patch = int((self.img_h // patch_size) * (self.img_w // patch_size))
        encoder_layers = [TranformerEncoder(embedded_dim, embedded_dim*4) for _ in range(encoder_layers)]

        self.net = nn.ModuleList([
            LinearProj(patch_size, in_channels, embedded_dim),
            PatchEmbedder(self.num_patch, embedded_dim),
            *encoder_layers,
            Classifier(embedded_dim, num_class)
        ])


    def forward(self, x):
        for layer in self.net:
            x = layer(x)

        return x