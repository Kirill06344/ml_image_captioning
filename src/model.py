"""Frozen ImageNet ResNet18 + a small randomly initialized causal Transformer decoder."""
import math
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class Captioner(nn.Module):
    def __init__(self, vocab_size, dim=256, layers=2, max_target=32, pretrained=True, dropout=0.1):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.encoder.requires_grad_(False)
        self.projection = nn.Linear(512, dim)
        self.visual_position = nn.Parameter(torch.randn(1, 49, dim) * .02)
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position = nn.Embedding(max_target, dim)
        layer = nn.TransformerDecoderLayer(dim, 4, dim * 4, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.head = nn.Linear(dim, vocab_size)
        self.max_target = max_target
        self.dim = dim

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Also freeze BatchNorm running statistics.
        return self

    @torch.no_grad()
    def encode_images(self, images):
        self.encoder.eval()
        return self.encoder(images).flatten(2).transpose(1, 2)

    def forward(self, features, input_ids):
        length = input_ids.shape[1]
        memory = self.projection(features) + self.visual_position
        hidden = self.embedding(input_ids) * math.sqrt(self.dim)
        hidden = hidden + self.position(torch.arange(length, device=input_ids.device))[None]
        causal = torch.ones(length, length, device=input_ids.device, dtype=torch.bool).triu(1)
        return self.head(self.decoder(hidden, memory, tgt_mask=causal, tgt_key_padding_mask=input_ids.eq(0)))

    @torch.no_grad()
    def generate(self, features, beams=1):
        """Beam search with mean log probability, EOS termination and no special-token output."""
        if beams < 1:
            raise ValueError('beams должен быть >= 1')
        results = []
        for feature in features:
            active = [([1], 0.0)]
            for _ in range(self.max_target - 1):
                expanded = []
                for seq, score in active:
                    if seq[-1] == 2:
                        expanded.append((seq, score))
                        continue
                    ids = torch.tensor([seq], device=features.device)
                    logits = self(feature[None], ids)[0, -1].clone()
                    logits[[0, 1, 3]] = -torch.inf
                    if len(seq) == 1:
                        logits[2] = -torch.inf
                    values, indices = logits.log_softmax(-1).topk(min(beams, logits.numel() - 3))
                    expanded.extend((seq + [i], score + v) for i, v in zip(indices.tolist(), values.tolist()))
                active = sorted(expanded, key=lambda item: item[1] / (len(item[0]) - 1), reverse=True)[:beams]
                if all(seq[-1] == 2 for seq, _ in active):
                    break
            results.append(active[0][0])
        return results
