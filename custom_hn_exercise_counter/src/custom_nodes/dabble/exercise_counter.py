"""
Custom node to show keypoints and count the number of push ups (sit ups).
"""
# pylint: disable=import-error
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode

from custom_hn_exercise_counter.src.custom_nodes.dabble.utils import (
    draw_text,
    map_bbox_to_image_coords,
    map_keypoint_to_image_coords,
)


@dataclass(frozen=True)
class GlobalParams:
    """Global parameters for the node."""

    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    WHITE: Tuple[int] = (255, 255, 255)  # opencv loads file in BGR format
    YELLOW: Tuple[int] = (0, 255, 255)

    # PoseNet/MoveNet's skeletal keypoints name to index mapping.
    KP_NAME_TO_INDEX: Dict[str, int] = field(
        default_factory=lambda: {
            "nose": 0,
            "left_eye": 1,
            "right_eye": 2,
            "left_ear": 3,
            "right_ear": 4,
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
            "left_hip": 11,
            "right_hip": 12,
            "left_knee": 13,
            "right_knee": 14,
            "left_ankle": 15,
            "right_ankle": 16,
        }
    )


# frozen false since will be mutating the correct_pose_msg
@dataclass(frozen=False)
class PushupPose:

    # angle > 160
    starting_elbow_angle: float = 155
    # angle < 90
    ending_elbow_angle: float = 90


class Node(AbstractNode):
    """Custom node to display keypoints and count number of hand waves

    Args:
       config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
       config.exercise_name (str): Name of the exercise. Default: "pushups".
       config.keypoint_threshold (float): Ignore keypoints below this threshold. Default: 0.3.

    Attributes:
        self.frame_count (int): Track the number of frames processed.
        self.expected_pose (str): The expected pose. Default: "down".
        self.num_push_ups (float): Cumulative number of push ups.
        self.have_started_push_ups (bool): Whether or not the push ups have started.
        self.elbow_angle (float): Angle of the elbow.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        # setup object working variables
        self.exercise_name: str
        self.keypoint_threshold: float  # ignore keypoints below this threshold

        self.frame_count = 0

        self.expected_pose = "down"
        self.num_push_ups = 0
        self.have_started_push_ups = False
        self.elbow_angle = None

        self.global_params = GlobalParams()
        self.push_up_pose = PushupPose()

        self.interested_keypoints = [
            "left_elbow",
            "left_shoulder",
            "left_wrist",
        ]

        self.reset_keypoints_to_none()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialize Exercise Type: {self.exercise_name}!")

    def reset_keypoints_to_none(self) -> None:
        """Reset all keypoints attributes to None after each frame."""

        for interested_keypoint in self.interested_keypoints:
            setattr(self, interested_keypoint, None)

    def inc_num_push_ups(self) -> float:
        """Increments the number of push ups by 0.5 for every directional change.

        Returns:
            self.num_push_ups (float): Cumulative number of push ups.
        """
        self.num_push_ups += 0.5
        return self.num_push_ups

    def is_up_pose(self, elbow_angle: float) -> bool:
        """Checks if the pose is an "up" pose by checking if the elbow angle is
        more than the starting elbow angle threshold.

        Elbow angle is defined by connecting the wrist to the elbow to the shoulder.

        Args:
            elbow_angle (np.float): The angle of the elbow.

        Returns:
            bool: True if the pose is a start pose, False otherwise.
        """
        return elbow_angle > self.push_up_pose.starting_elbow_angle

    def is_down_pose(self, elbow_angle: float) -> None:
        """Checks if the pose is an "down" pose by checking if the elbow angle is
        more than the starting elbow angle threshold.

        Elbow angle is defined by connecting the wrist to the elbow to the shoulder.

        Args:
            elbow_angle (np.float): The angle of the elbow.

        Returns:
            bool: True if the pose is a start pose, False otherwise.
        """
        return elbow_angle <= self.push_up_pose.ending_elbow_angle

    @staticmethod
    def is_bbox_or_keypoints_empty(
        bboxes: np.ndarray,
        keypoints: np.ndarray,
        keypoint_scores: np.ndarray,
    ) -> bool:
        """Checks if the bounding box or keypoints are empty.

        If any of them are empty, then we will not draw the bounding box or keypoints.

        Args:
            bboxes (np.ndarray): The bounding boxes.
            keypoints (np.ndarray): The keypoints.
            keypoint_scores (np.ndarray): The keypoint scores.

        Returns:
            bool: True if the bounding box and keypoints are empty, False otherwise.
        """

        return (
            len(bboxes) == 0
            or len(keypoints) == 0
            or len(keypoint_scores) == 0
        )

    # pylint: disable=too-many-locals
    def count_push_ups(
        self,
        img: np.ndarray,
        img_size: Tuple[int, int],
        the_keypoints: np.ndarray,
        the_keypoint_scores: np.ndarray,
    ) -> None:
        """Counts the number of push ups.

        Args:
            img (np.ndarray): The image in each frame.
            img_size (Tuple[int, int]): The width and height of the image in each frame.
            the_keypoints (np.ndarray): The keypoints predicted in each frame.
            the_keypoint_scores (np.ndarray): The keypoint scores predicted in each frame.
        """

        interested_keypoints_names_to_index = {
            self.global_params.KP_NAME_TO_INDEX[
                interested_keypoint
            ]: interested_keypoint
            for interested_keypoint in self.interested_keypoints
        }

        self.reset_keypoints_to_none()

        for keypoint_idx, (keypoints, keypoint_score) in enumerate(
            zip(the_keypoints, the_keypoint_scores)
        ):
            if keypoint_score >= self.keypoint_threshold:

                x, y = map_keypoint_to_image_coords(
                    keypoints.tolist(), img_size
                )
                x_y_str = f"({x}, {y})"

                if keypoint_idx in interested_keypoints_names_to_index:
                    keypoint_name = interested_keypoints_names_to_index[
                        keypoint_idx
                    ]
                    setattr(self, keypoint_name, (x, y))
                    the_color = self.global_params.YELLOW
                else:
                    the_color = self.global_params.WHITE

                draw_text(
                    img,
                    x_y_str,
                    (x, y),
                    color=the_color,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    thickness=2,
                )

        # all keypoints must be non-none
        if self.left_elbow and self.left_shoulder and self.left_wrist:

            left_elbow_angle = self.calculate_angle_using_dot_prod(
                self.left_shoulder, self.left_elbow, self.left_wrist
            )

            self.elbow_angle = left_elbow_angle

            # Check to ensure right form before starting the program
            if self.is_up_pose(left_elbow_angle):
                self.have_started_push_ups = True

            # Check for full range of motion for the pushup
            if self.have_started_push_ups:
                # the two if-statements are mutually exclusive: won't happen at the same time.
                if (
                    self.is_down_pose(left_elbow_angle)
                    and self.expected_pose == "down"
                ):
                    self.inc_num_push_ups()
                    self.expected_pose = "up"

                if (
                    self.is_up_pose(left_elbow_angle)
                    and self.expected_pose == "up"
                ):
                    self.inc_num_push_ups()
                    self.expected_pose = "down"

            pushup_str = f"#push_ups = {self.num_push_ups}"
            draw_text(
                img,
                pushup_str,
                (20, 30),
                color=self.global_params.YELLOW,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=3,
            )

    # pylint: disable=trailing-whitespace
    @staticmethod
    def calculate_angle_using_dot_prod(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        return_as_degrees: bool = True,
    ) -> float:
        r"""Takes in three points and calculates the angle between them using dot product.
        Let B be the common point of two vectors BA and BC, then angle ABC is the angle between vectors BA and BC.

        This function calculates the angle ABC using the dot product formula:
      
        $$
        \begin{align*}
        BA &= a - b \\
        BC &= c - b \\
        \cos(angle(ABC)) &= \dfrac{(BA \cdot BC)}{(|BA||BC|)}
        \end{align*}
        $$

        Args:
            a (np.ndarray): Point a corresponding to A.
            b (np.ndarray): Point b corresponding to B.
            c (np.ndarray): Point c corresponding to C.
            return_as_degrees (bool): Returns angle in degrees if True else radians. Default: True.

        Returns:
            angle (float): Angle between vectors BA and BC.

        Shape:
            - Input:
                - a (np.ndarray): (2, )
                - b (np.ndarray): (2, )
                - c (np.ndarray): (2, )

        Examples:
            >>> import numpy as np
            >>> a = (6, 0)
            >>> b = (0, 0)
            >>> c = (6, 6)
            >>> calculate_angle_using_dot_prod(a,b,c)
            45.0
        """
        # turn the points into numpy arrays
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (
            np.linalg.norm(ba) * np.linalg.norm(bc)
        )
        # arccos range is [0, pi]
        angle = np.arccos(cosine_angle)

        if return_as_degrees:
            return np.degrees(angle)
        return angle

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node draws keypoints and counts the number of push ups.

        Args:
            inputs (dict): Dictionary with keys
                "img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores".

        Returns:
            outputs (dict): Dictionary with keys
                "frame_count", "num_waves", "body_direction", "elbow_angle", "shoulder_keypoint", "elbow_keypoint", "wrist_keypoint"
        """

        # get required inputs from pipeline
        img = inputs["img"]
        bboxes = inputs["bboxes"]
        bbox_scores = inputs["bbox_scores"]
        keypoints = inputs["keypoints"]
        keypoint_scores = inputs["keypoint_scores"]
        filename = inputs["filename"]

        img_size = (img.shape[1], img.shape[0])  # image width, height

        # frame count should not be in the if-clause
        self.frame_count += 1

        if not self.is_bbox_or_keypoints_empty(
            bboxes, keypoints, keypoint_scores
        ):

            # note this bbox is from the pose estimation model and not from yolo but bbox_scores is from yolo
            the_bbox = bboxes[0]  # image only has one person
            # only one set of scores and handle this differently from is_bbox_or_keypoints_empty cause this is from yolo..
            the_bbox_score = bbox_scores[0] if len(bbox_scores) > 0 else 0

            # y1 and x2 are private since it isn't called
            x1, _y1, _x2, y2 = map_bbox_to_image_coords(the_bbox, img_size)
            score_str = f"BBox {the_bbox_score:0.2f}"

            # get bounding box confidence score and draw it at the left-bottom
            # (x1, y2) corner of the bounding box (offset by 30 pixels)
            draw_text(
                img,
                score_str,
                (x1, y2 - 30),
                color=self.global_params.WHITE,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                thickness=3,
            )

            the_keypoints = keypoints[0]  # image only has one person
            the_keypoint_scores = keypoint_scores[0]  # only one set of scores

            self.count_push_ups(
                img, img_size, the_keypoints, the_keypoint_scores
            )
        # careful not to indent this return statement
        # if the if-clause is false, then no dict will be returned and will crash the pipeline
        return {
            "filename": filename,
            "expected_pose": self.expected_pose,
            "num_push_ups": self.num_push_ups,
            "frame_count": self.frame_count,
            "elbow_angle": self.elbow_angle,
            "elbow_keypoint": self.left_elbow,
            "shoulder_keypoint": self.left_shoulder,
            "wrist_keypoint": self.left_wrist,
        }
