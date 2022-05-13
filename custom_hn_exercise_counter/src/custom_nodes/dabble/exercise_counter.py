"""
Custom node to show keypoints and count the number of push ups (sit ups).
"""
# pylint: disable=import-error
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Type

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
    """Global parameters for the node.

    Attributes:
        FONT (int): Font for the text.
        WHITE (int): White color in BGR.
        YELLOW (int): Yellow color in BGR.
        KP_NAME_TO_INDEX (Dict[str, int]): PoseNet/MoveNet's skeletal keypoints name to index mapping.
    """

    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    WHITE: Tuple[int] = (255, 255, 255)
    YELLOW: Tuple[int] = (0, 255, 255)

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


@dataclass(frozen=False)
class PushupPoseParams:
    """Push up pose parameters.

    Attributes:
        starting_elbow_angle (float): The threshold angle formed between the wrist, elbow and shoulder for starting (up) pose.
        ending_elbow_angle (float): The threshold angle formed between the wrist, elbow and shoulder for ending (down) pose.
    """

    starting_elbow_angle: float
    ending_elbow_angle: float

    @classmethod
    def from_dict(
        cls: Type["PushupPoseParams"], params_dict: Dict[str, Any]
    ) -> Type["PushupPoseParams"]:
        """Takes in a dictionary of parameters and returns a PushupPoseParams Dataclass.

        Args:
            params_dict (Dict[str, Any]): Dictionary of parameters.

        Returns:
            (PushupPoseParams): Dataclass with the parameters initalized.
        """
        return cls(
            starting_elbow_angle=params_dict["starting_elbow_angle"],
            ending_elbow_angle=params_dict["ending_elbow_angle"],
        )


class Node(AbstractNode):
    """Custom node to display keypoints and count number of hand waves

    Args:
       config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
       config.exercise_name (str): Name of the exercise. Default: "pushups".
       config.keypoint_threshold (float): Ignore keypoints below this threshold. Default: 0.3.
       config.push_up_pose_params (:obj:`Dict[str, Any]`): Parameters for the push up pose. Default: {starting_elbow_angle: 155, ending_elbow_angle: 90}.


    Attributes:
        self.frame_count (int): Track the number of frames processed.
        self.expected_pose (str): The expected pose. Default: "down".
        self.num_push_ups (float): Cumulative number of push ups. Default: 0.
        self.have_started_push_ups (bool): Whether or not the push ups have started. Default: False.
        self.elbow_angle (float): Angle of the elbow. Default: None.
        self.global_params_dataclass (GlobalParams): Global parameters for the node.
        self.push_up_pose_params_dataclass (PushupPoseParams): Push up pose parameters.
        self.interested_keypoints (List[str]): List of keypoints to track. Default: ["left_elbow", "left_shoulder", "left_wrist"].
        self.left_elbow (float): Keypoints of the left elbow. Default: None.
        self.left_shoulder (float): Keypoints of the left shoulder. Default: None.
        self.left_wrist (float): Keypoints of the left wrist. Default: None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.logger = logging.getLogger(__name__)

        self.exercise_name: str
        self.keypoint_threshold: float  # ignore keypoints below this threshold
        self.push_up_pose_params: Dict[str, Any]

        self.logger.info(f"Initialize Exercise Type: {self.exercise_name}!")

        self.frame_count = 0

        self.expected_pose = "down"
        self.num_push_ups = 0
        self.have_started_push_ups = False
        self.elbow_angle = None

        self.global_params_dataclass = GlobalParams()
        self.push_up_pose_params_dataclass = PushupPoseParams.from_dict(
            self.push_up_pose_params
        )

        self.interested_keypoints = [
            "left_elbow",
            "left_shoulder",
            "left_wrist",
        ]

        # each element in the self.interested_keypoints list will now become an attribute initialized to None
        self.reset_keypoints_to_none()

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
            elbow_angle (float): The angle of the elbow.

        Returns:
            bool: True if the pose is a up pose, False otherwise.
        """
        return (
            elbow_angle
            > self.push_up_pose_params_dataclass.starting_elbow_angle
        )

    def is_down_pose(self, elbow_angle: float) -> None:
        """Checks if the pose is an "down" pose by checking if the elbow angle is
        more than the ending elbow angle threshold.

        Elbow angle is defined by connecting the wrist to the elbow to the shoulder.

        Args:
            elbow_angle (float): The angle of the elbow.

        Returns:
            bool: True if the pose is a down pose, False otherwise.
        """
        return (
            elbow_angle
            <= self.push_up_pose_params_dataclass.ending_elbow_angle
        )

    @staticmethod
    def is_bbox_or_keypoints_empty(
        bboxes: np.ndarray,
        keypoints: np.ndarray,
        keypoint_scores: np.ndarray,
    ) -> bool:
        """Checks if the bounding box or keypoints are empty.

        If any of them are empty, then we won't perform any further processing and go to next frame.

        Args:
            bboxes (np.ndarray): The bounding boxes.
            keypoints (np.ndarray): The keypoints.
            keypoint_scores (np.ndarray): The keypoint scores.

        Returns:
            bool: True if the bounding box or keypoints are empty, False otherwise.
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
            self.global_params_dataclass.KP_NAME_TO_INDEX[
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
                    the_color = self.global_params_dataclass.YELLOW
                else:
                    the_color = self.global_params_dataclass.WHITE

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
                color=self.global_params_dataclass.YELLOW,
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
            angle (float): Angle between vectors BA and BC in radians or degrees.

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
                - img (np.ndarray): The image in each frame.
                - bboxes (np.ndarray): The bounding boxes predicted in each frame. Note this belongs to MoveNet.
                - bbox_scores (np.ndarray): The bounding box scores predicted in each frame. Note this belongs to Yolo.
                - keypoints (np.ndarray): The keypoints predicted in each frame.
                - keypoint_scores (np.ndarray): The keypoint scores predicted in each frame.
                - filename (str): The filename of the image/video.

        Note:
            To check the shapes of bboxes, bbox_scores, keypoints, and keypoint_scores, please refer to [PeekingDuck API Documentation](https://peekingduck.readthedocs.io/en/stable/nodes/model.movenet.html#module-model.movenet).

        Returns:
            outputs (dict): Dictionary with keys
                - filename: The filename of the image/video.
                - expected_pose: The expected pose of the image/video.
                - num_push_ups: The number of cumulative push ups.
                - frame_count: The number of frames processed.
                - elbow_angle: The angle between the wrist, elbow and shoulder.
                - elbow_keypoint: The keypoint coordinates of the elbow.
                - shoulder_keypoint: The keypoint coordinates of the shoulder.
                - wrist_keypoint: The keypoint coordinates of the wrist.
        """

        # get required inputs from pipeline
        img = inputs["img"]
        bboxes = inputs["bboxes"]
        bbox_scores = inputs["bbox_scores"]
        keypoints = inputs["keypoints"]
        keypoint_scores = inputs["keypoint_scores"]
        filename = inputs["filename"]

        # image width, height
        img_size = (img.shape[1], img.shape[0])

        # frame count should not be in the if-clause
        self.frame_count += 1

        if not self.is_bbox_or_keypoints_empty(
            bboxes, keypoints, keypoint_scores
        ):
            # assume each frame has only one person;
            # note this bbox is from the posenet/movenet and not from yolo.
            the_bbox = bboxes[0]
            # bbox_scores are from yolo and not posenet/movenet.
            the_bbox_score = bbox_scores[0] if len(bbox_scores) > 0 else 0

            x1, _y1, _x2, y2 = map_bbox_to_image_coords(the_bbox, img_size)
            score_str = f"BBox {the_bbox_score:0.2f}"

            # get bounding box confidence score and draw it at the left-bottom
            # (x1, y2) corner of the bounding box (offset by 30 pixels)
            draw_text(
                img,
                score_str,
                (x1, y2 - 30),
                color=self.global_params_dataclass.WHITE,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                thickness=3,
            )

            # assume each frame has only one person;
            the_keypoints = keypoints[0]
            the_keypoint_scores = keypoint_scores[0]

            # count the number of push ups
            self.count_push_ups(
                img, img_size, the_keypoints, the_keypoint_scores
            )

        # careful not to indent this return statement;
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
