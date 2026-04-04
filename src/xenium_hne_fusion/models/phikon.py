import torch
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
from torchvision.transforms import v2


class Phikon(nn.Module):

    def __init__(self):
        super().__init__()
        self.preprocess = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        self.model = AutoModel.from_pretrained("owkin/phikon-v2")
        self.model.eval()
        self.embed_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]  # (batch_size, embed_dim)
        return features

    def get_transform(self):
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[
                    0.485,
                    0.456,
                    0.406
                ], std=[
                    0.229,
                    0.224,
                    0.225
                ]),
            ]
        )
        return transform
