import torch
from transformers import AutoModel
import torch.nn as nn
from torchvision.transforms import v2

class Midnight(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained('kaiko-ai/midnight')
        self.model.eval()
        self.embed_dim = 1536

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]  # (batch_size, embed_dim)
        return features

    def extract_classification_embedding(self, x):
        cls_embedding, patch_embeddings = x[:, 0, :], x[:, 1:, :]
        return torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)

    def get_transform(self):

        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        return transform

