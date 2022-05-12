# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shows the outputs on your display.
"""

import imp
from pathlib import Path
from typing import Any, Dict
import numpy as np
import cv2

from peekingduck.pipeline.nodes.node import AbstractNode

# in source of visual.py, the input should either be images or videos
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".m4v", ".mkv"]


class Node(AbstractNode):
    """Streams the output on your display.

    Inputs:
        |img_data|

    Outputs:
        |pipeline_end_data|

    Configs:
        window_name (:obj:`str`): **default = "PeekingDuck"** |br|
            Name of the displayed window.
        window_size (:obj:`Dict`):
            **default = { do_resizing: False, width: 1280, height: 720 }** |br|
            Resizes the displayed window to the chosen width and weight, if
            ``do_resizing`` is set to ``true``. The size of the displayed
            window can also be adjusted by clicking and dragging.
        window_loc (:obj:`Dict`): **default = { x: 0, y: 0 }** |br|
            X and Y coordinates of the top left corner of the displayed window,
            with reference from the top left corner of the screen, in pixels.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(
            self.window_name, self.window_loc["x"], self.window_loc["y"]
        )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Show the outputs on your display"""

        img = inputs["img"]
        gradcam_image = inputs["gradcam_image"]
        source_name = inputs["filename"]

        # TODO: Find ways to concat two images with legends on cv2 imshow, this won't work if the images are not the same size.
        if self.window_size["do_resizing"]:
            img = cv2.resize(
                img, (self.window_size["width"], self.window_size["height"])
            )
            gradcam_image = cv2.resize(
                gradcam_image,
                (self.window_size["width"], self.window_size["height"]),
            )
            numpy_horizontal_concat = np.concatenate(
                (img, gradcam_image), axis=1
            )
        cv2.imshow(self.window_name, numpy_horizontal_concat)

        if Path(source_name).suffix in VIDEO_EXTENSIONS:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow(self.window_name)
                return {"pipeline_end": True}
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow(self.window_name)
                return {"pipeline_end": True}
            # cv2.waitKey(0)
            # cv2.destroyWindow(self.window_name)
            # return {"pipeline_end": True}

        return {"pipeline_end": False}
