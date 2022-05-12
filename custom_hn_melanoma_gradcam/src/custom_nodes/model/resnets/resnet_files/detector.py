import logging
from pathlib import Path

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from typing import Dict, Any, List

from custom_hn_melanoma_gradcam.src.custom_nodes.model.resnets.resnet_files import (
    model,
)


class Detector:  # pylint: disable=too-few-public-methods, too-many-instance-attributes, not-callable
    """Object detection class using YOLOX to predict object bboxes.

    Attributes:
        logger (logging.Logger): Events logger.
        config (Dict[str, Any]): YOLOX node configuration.
        model_dir (pathlib.Path): Path to directory of model weights files.
        device (torch.device): Represents the device on which the torch.Tensor
            will be allocated.
        half (bool): Flag to determine if half-precision should be used.
        yolox (YOLOX): The YOLOX model for performing inference.
    """

    def __init__(self, config: Dict[str, Any], model_dir: Path) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_dir = model_dir
        self.class_label_map = self.config["class_label_map"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Half-precision only supported on CUDA
        self.config["half"] = (
            self.config["half"] and self.device.type == "cuda"
        )

        self.transform_image = self._get_transforms()

        self.resnet50d = self._create_resnet50d_model()

    def _get_model(self):
        """Constructs YOLOX model based on parsed configuration.

        Args:
            model_size (Dict[str, float]): Depth and width of the model.

        Returns:
            (YOLOX): YOLOX model.
        """
        return model.CustomNeuralNet(
            model_name=self.config["model_params"]["model_name"],
            out_features=self.config["model_params"]["out_features"],
            in_channels=self.config["model_params"]["in_channels"],
            pretrained=self.config["model_params"]["pretrained"],
            use_meta=self.config["model_params"]["use_meta"],
        )

    # the latest version uses @torch.inference_mode(mode=True)
    @torch.no_grad()
    def predict_class_from_image(self, image):
        image = self.transform_image(image=image)["image"].to(self.device)
        if len(image.shape) != 4:
            image = image.unsqueeze(0)
        logits = self.resnet50d(image)
        probs = getattr(torch.nn, "Softmax")(dim=1)(logits).cpu().numpy()
        class_name = self.class_label_map[np.argmax(probs)]
        return {"pred_label": class_name, "pred_score": probs}

    def _create_resnet50d_model(self):
        model_type = self.config["model_params"]["model_name"]
        model_path = (
            self.model_dir / self.config["weights"]["model_file"][model_type]
        )
        return self._load_resnet50d_weights(model_path)

    def _load_resnet50d_weights(self, model_path: Path):

        """Loads YOLOX model weights.

        Args:
            model_path (Path): Path to model weights file.
            model_settings (Dict[str, float]): Depth and width of the model.

        Returns:
            (YOLOX): YOLOX model.

        Raises:
            ValueError: `model_path` does not exist.
        """
        # TODO: Consider making state_dict as an attribute since I will be using on gradcam
        if model_path.is_file():
            # cpkt = checkpoint
            ckpt = torch.load(str(model_path), map_location="cpu")
            model = self._get_model().to(self.device)
            if self.config["half"]:
                model.half()
            model.eval()
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded model weights from {model_path}")
            return model

        raise ValueError(
            f"Model file does not exist. Please check that {model_path} exists."
        )

    def _get_transforms(self):
        """Performs Augmentation on test dataset.
        Remember tta transforms need resize and normalize.
        Args:
            pipeline_config (global_params.PipelineConfig): The pipeline config.
            image_size (int, optional): The image size. Defaults to TRANSFORMS.image_size.
            mean (List[float], optional): The mean. Defaults to TRANSFORMS.mean.
            std (List[float], optional): The std. Defaults to TRANSFORMS.std.
        Returns:
            transforms_dict (Dict[str, albumentations.core.composition.Compose]): Returns the transforms for inference in a dictionary which can hold TTA transforms.
        """

        return albumentations.Compose(
            [
                albumentations.Resize(
                    self.config["input_size"],
                    self.config["input_size"],
                ),
                albumentations.Normalize(
                    mean=self.config["mean"],
                    std=self.config["std"],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )

    def show_resnet_gradcam(
        self, image: np.ndarray, original_image, plot_gradcam=True
    ):
        """Log gradcam images into wandb for error analysis.
        # TODO: Consider getting the logits for error analysis, for example, if a predicted image which is correct has high logits this means the model is very sure, conversely, if a predicted image has low logits and also wrong, we also check why.
        """
        model = self.resnet50d
        # only for resnet variants! for simplicity we don't enumerate other models!
        target_layers = [model.backbone.layer4[-1]]
        reshape_transform = None

        # input is np array so turn to tensor as well as resize aand as well as turn to tensor by totensorv2
        # so no need permute since albu does it
        image = self.transform_image(image=image)["image"].to(self.device)

        # if image tensor is 3 dim, unsqueeze it to 4 dim with 1 in front.
        image = image.unsqueeze(0)
        gradcam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=reshape_transform,
        )

        # # If targets is None, the highest scoring category will be used for every image in the batch.
        gradcam_output = gradcam(
            input_tensor=image,
            target_category=None,
            aug_smooth=False,
            eigen_smooth=False,
        )
        original_image = original_image / 255.0

        gradcam_image = show_cam_on_image(
            original_image, gradcam_output[0], use_rgb=False
        )

        if plot_gradcam:
            _fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
            axes[0].imshow(original_image)
            # axes[0].set_title(f"y_true={y_true:.4f}")
            axes[1].imshow(gradcam_image)
            # axes[1].set_title(f"y_pred={y_pred}")
            plt.show()
            torch.cuda.empty_cache()

        return gradcam_image
