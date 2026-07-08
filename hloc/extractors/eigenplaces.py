"""
EigenPlaces global descriptor extractor.

EigenPlaces paper: https://arxiv.org/abs/2308.10832
Official PyTorch Hub: https://github.com/gmberton/EigenPlaces
"""

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class EigenPlaces(BaseModel):
    default_conf = {
        "backbone": "ResNet50",
        "fc_output_dim": 2048,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        self.net = torch.hub.load(
            "gmberton/EigenPlaces",
            "get_trained_model",
            backbone=conf["backbone"],
            fc_output_dim=conf["fc_output_dim"],
        ).eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data["image"])
        desc = self.net(image)
        return {"global_descriptor": desc}
