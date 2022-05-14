import logging

from pathlib import Path
from typing import Any, Dict, List, Tuple


import numpy as np

import yaml
import cv2
from peekingduck.weights_utils import checker

from custom_hn_melanoma_gradcam.src.custom_nodes.model.resnets.resnet_files import (
    detector,
    downloader,
)


class ResnetModel:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)

        # replace the original finder to a more hardcoded config
        weights_dir = Path(config["weights_parent_dir"])
        model_dir = Path.joinpath(
            weights_dir, config["weights"]["model_subdir"]
        )

        if not checker.has_weights(weights_dir, model_dir):
            self.logger.warning(
                "No weights detected. Proceeding to download..."
            )
            downloader.download_weights(
                weights_dir, config["weights"]["blob_file"]
            )
            self.logger.info(f"Weights downloaded to {weights_dir}.")

        with open(model_dir / config["weights"]["classes_file"]) as infile:
            # self.class_names change to self.class_label_map
            self.class_label_map = yaml.safe_load(infile)

        self.detector = detector.Detector(config, model_dir)
        self.input_shape = (config["input_size"], config["input_size"])

    def show_gradcam(
        self, image: np.ndarray, plot_gradcam: bool = False
    ) -> np.ndarray:
        """Shows the gradcam of the image.

        Args:
            image (np.ndarray): The input image frame.
            plot_gradcam (bool, optional): Whether to plot the gradcam image. Defaults to False.

        Returns:
            gradcam_image (np.ndarray): The gradcam image.
        """
        reshaped_original_image = cv2.resize(image, self.input_shape)
        gradcam_image = self.detector.show_resnet_gradcam(
            image, reshaped_original_image, plot_gradcam
        )

        return gradcam_image

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """Predicts bboxes from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[List[np.ndarray], List[str], List[float]]): Returned tuple
                contains:
                - A list of detection bboxes
                - A list of human-friendly detection class names
                - A list of detection scores

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        return self.detector.predict_class_from_image(image)


if __name__ == "__main__":
    # This config_copy is to mimic the yaml file that is passed in the pkd pipeline, want to test locally first.
    config_copy = yaml.safe_load(
        Path(
            r"C:\Users\reighns\reighns_ml\ml_projects\pkd_exercise_counter\custom_hn_melanoma_gradcam\src\custom_nodes\configs\model\melanoma_classifier.yml"
        ).read_text()
    )
    # resnet_detector = Detector(
    #     config=config_copy, model_dir=MODEL_DIR, class_names=class_label_map
    # )
    resnet_det = ResnetModel(config_copy)
    image_path = r"C:\Users\reighns\reighns_ml\ml_projects\pkd\melanoma_model\melanoma_data\inspection\ISIC_0076262.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshaped_original_image = cv2.resize(image, (224, 224))
    prediction_dict = resnet_det.predict(image)
    print(prediction_dict)
    # prediction = resnet_detector.predict_class_from_image(image)
    _ = resnet_det.show_gradcam(image, plot_gradcam=True)
