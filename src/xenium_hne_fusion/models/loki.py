# %%
import torch
import torch.nn as nn
from beat_xe_hne.transforms.utils import get_normalize_from_transform
from open_clip import create_model_from_pretrained, get_tokenizer
from torchvision.transforms import v2

class Loki(nn.Module):

    def __init__(self, ckpt_path: str, device: str = 'cpu'):
        super().__init__()
        self.model, self.preprocess = create_model_from_pretrained(
            "coca_ViT-L-14", device=device, pretrained=ckpt_path, weights_only=False,
        )
        self.tokenizer = get_tokenizer("coca_ViT-L-14")
        self.model.eval()
        self.embed_dim = 768

    def forward(self, x):
        return self.model.encode_image(x)

    def get_transform(self):
        normalize = get_normalize_from_transform(self.preprocess)

        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                normalize,
            ]
        )
        return transform

