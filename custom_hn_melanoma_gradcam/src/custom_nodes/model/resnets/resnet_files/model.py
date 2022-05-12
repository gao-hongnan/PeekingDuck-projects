import torch
import timm
from typing import Dict, Callable


class CustomNeuralNet(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        out_features: int,
        in_channels: int,
        pretrained: bool,
        use_meta: bool,
    ):
        """Construct a new model.
        Args:
            model_name ([type], str): The name of the model to use. Defaults to MODEL_PARAMS.model_name.
            out_features ([type], int): The number of output features, this is usually the number of classes, but if you use sigmoid, then the output is 1. Defaults to MODEL_PARAMS.output_dimension.
            in_channels ([type], int): The number of input channels; RGB = 3, Grayscale = 1. Defaults to MODEL_PARAMS.input_channels.
            pretrained ([type], bool): If True, use pretrained model. Defaults to MODEL_PARAMS.pretrained.
        """
        super().__init__()

        self.in_channels = in_channels
        self.pretrained = pretrained
        self.use_meta = use_meta

        self.backbone = timm.create_model(
            model_name, pretrained=self.pretrained, in_chans=self.in_channels
        )

        # removes head from backbone: # TODO: Global pool = "avg" vs "" behaves differently in shape, caution!
        self.backbone.reset_classifier(num_classes=0, global_pool="avg")

        # get the last layer's number of features in backbone (feature map)
        self.in_features = self.backbone.num_features
        self.out_features = out_features

        # Custom Head
        self.single_head_fc = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.out_features),
        )

        if self.use_meta:
            pass

        self.architecture: Dict[str, Callable] = {
            "backbone": self.backbone,
            "bottleneck": None,
            "head": self.single_head_fc,
        }

    def extract_features(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """Extract the features mapping logits from the model.
        This is the output from the backbone of a CNN.
        Args:
            image (torch.FloatTensor): The input image.
        Returns:
            feature_logits (torch.FloatTensor): The features logits.
        """
        # TODO: To rename feature_logits to image embeddings, also find out what is image embedding.
        feature_logits = self.architecture["backbone"](image)
        return feature_logits

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """The forward call of the model.
        Args:
            image (torch.FloatTensor): The input image.
        Returns:
            classifier_logits (torch.FloatTensor): The output logits of the classifier head.
        """

        feature_logits = self.extract_features(image)
        classifier_logits = self.architecture["head"](feature_logits)

        return classifier_logits
