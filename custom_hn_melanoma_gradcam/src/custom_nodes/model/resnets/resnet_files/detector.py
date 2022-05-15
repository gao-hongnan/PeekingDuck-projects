import logging
from pathlib import Path
from typing import Any, Dict, Union

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from custom_hn_melanoma_gradcam.src.custom_nodes.model.resnets.resnet_files import (
    model,
)


class Detector:  # pylint: disable=too-few-public-methods, too-many-instance-attributes, not-callable
    """Image Classification Model ResNet used to predict Melanoma.

    Attributes:
        logger (logging.Logger): Events logger.
        config (Dict[str, Any]): ResNet node configuration.
        model_dir (pathlib.Path): Path to directory of model weights files.
        device (torch.device): Represents the device on which the torch.Tensor
            will be allocated.
        half (bool): Flag to determine if half-precision should be used.
        transform_image (albumentations.Compose): Albumentations Compose object.
        resnet50d (model.CustomNeuralNet): The ResNet model for performing inference.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_dir: Path,
        class_label_map: Dict[int, str],
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_dir = model_dir
        self.class_label_map = class_label_map
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Half-precision only supported on CUDA
        self.config["half"] = (
            self.config["half"] and self.device.type == "cuda"
        )

        self.transform_image = self._get_transforms()

        self.resnet50d = self._create_resnet50d_model()

    # the latest version uses @torch.inference_mode(mode=True)
    @torch.no_grad()
    def predict_class_from_image(
        self, image: np.ndarray
    ) -> Union[Dict[str, str], Dict[str, float]]:
        """Predicts the class of an image.

        Note:
            Our pipeline takes in input as np.ndarray but inference needs to be done as torch.Tensor.
            Albumentations handles this by converting the np.ndarray to torch.Tensor.
            Note we use max to get pred_score, so it corresponds to the probability of the predicted class.

        Args:
            image (np.ndarray): The input image.

        Returns:
            Union[Dict[str, str], Dict[str, float]]:
                - pred_label (str): The predicted class label.
                - pred_score (float): The predicted class score.
        """

        image = self.transform_image(image=image)["image"].to(self.device)

        if len(image.shape) != 4:
            image = image.unsqueeze(0)  # np.expand_dims(image, 0)

        logits = self.resnet50d(image)  # model.forward(image)
        # only 1 image per inference take 0 index.
        probs = getattr(torch.nn, "Softmax")(dim=1)(logits).cpu().numpy()[0]
        pred_score = probs.max() * 100
        class_name = self.class_label_map[np.argmax(probs)]
        return {"pred_label": class_name, "pred_score": pred_score}

    def _create_resnet50d_model(self) -> model.CustomNeuralNet:
        """Creates ResNet50d model and loads its weights.

        Returns:
            (model.CustomNeuralNet): ResNet50d model.
        """
        model_type = self.config["model_params"]["model_name"]
        model_path = (
            self.model_dir / self.config["weights"]["model_file"][model_type]
        )
        return self._load_resnet50d_weights(model_path)

    def _get_model(self) -> model.CustomNeuralNet:
        """Constructs ResNet50d model based on parsed configuration.

        Args:
            config["model_params"] (Dict[str, Any]): Model parameters from config file to be unpacked.

        Returns:
            (model.CustomNeuralNet): ResNet50d model.
        """
        return model.CustomNeuralNet(**self.config["model_params"])

    def _load_resnet50d_weights(
        self, model_path: Path
    ) -> model.CustomNeuralNet:
        """Loads ResNet50d model weights.

        Args:
            model_path (Path): Path to model weights file.

        Returns:
            model (model.CustomNeuralNet): ResNet50d model.

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
            self.logger.info(f"Loaded model weights from {model_path}")

            return model

        raise ValueError(
            f"Model file does not exist. Please check that {model_path} exists."
        )

    def _get_transforms(self) -> albumentations.Compose:
        """Gets the transforms function for the image.

        Args:
            image_size (int, optional): The image size. Defaults to config["input_size"].
            mean (List[float], optional): The mean. Defaults to config["mean"].
            std (List[float], optional): The std. Defaults to config["std"].

        Returns:
            albumentations.Compose: An Albumentations Compose object.
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
        self,
        image: np.ndarray,
        original_image: np.ndarray,
        plot_gradcam: bool = True,
    ) -> np.ndarray:
        """Show the gradcam of the image for ResNet variants.

        This will not work on other types of network architectures.

        Args:
            image (np.ndarray): The input image.
            original_image (np.ndarray): The original image.
            plot_gradcam (bool): If True, will plot the gradcam. Defaults to True.

        Returns:
            gradcam_image (np.ndarray): The gradcam image.
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
