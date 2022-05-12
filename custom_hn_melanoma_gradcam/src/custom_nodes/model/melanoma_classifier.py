"""
Casting classification model.
"""

from typing import Any, Dict

import cv2
import numpy as np


from peekingduck.pipeline.nodes.node import AbstractNode
from custom_hn_melanoma_gradcam.src.custom_nodes.model.resnets import (
    resnet_model,
)


class Node(AbstractNode):
    """Initializes and uses a CNN to predict if an image frame shows a normal
    or defective casting.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = resnet_model.ResnetModel(self.config)

        self.input_shape = (
            self.config["input_size"],
            self.config["input_size"],
        )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the image input and returns the predicted class label and
        confidence score.

        Args:
              inputs (dict): Dictionary with key "img".

        Returns:
              outputs (dict): Dictionary with keys "pred_label" and "pred_score".
        """

        img = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)

        reshaped_original_image = cv2.resize(img, self.input_shape)
        prediction_dict = self.model.predict(img)
        gradcam_image = self.model.show_gradcam(reshaped_original_image)
        print(prediction_dict)
        return {
            **prediction_dict,
            "gradcam_image": gradcam_image,
        }
