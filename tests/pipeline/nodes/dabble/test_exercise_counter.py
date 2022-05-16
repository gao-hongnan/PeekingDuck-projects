import warnings

import numpy as np
import pytest

from custom_hn_exercise_counter.src.custom_nodes.dabble.exercise_counter import (
    Node,
)

# pylint: disable=missing-function-docstring


@pytest.fixture
def calculate_angle():
    node = Node(
        {
            "input": ["none"],
            "output": ["none"],
            "exercise_name": "push_ups",
            "keypoint_threshold": 0.3,
            "push_up_pose_params": {
                "starting_elbow_angle": 155,
                "ending_elbow_angle": 90,
            },
        }
    )
    return node


class TestExerciseCounter:
    def test_angle_empty_inputs(self, calculate_angle):
        a = np.asarray([])
        b = np.asarray([])
        c = np.asarray([])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            angle = calculate_angle.calculate_angle_using_dot_prod(a, b, c)

        assert np.isnan(angle)

    def test_angle_test_same_vectors(self, calculate_angle):
        a = np.asarray([6, 0])
        b = np.asarray([6, 0])
        c = np.asarray([10, 0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            angle = calculate_angle.calculate_angle_using_dot_prod(a, b, c)

        assert np.isnan(angle)

    def test_angle_is_zero(self, calculate_angle):
        a = (6, 0)
        b = (0, 0)
        c = (10, 0)

        angle = calculate_angle.calculate_angle_using_dot_prod(a, b, c)
        assert angle == 0

    def test_angle_is_hundred_eighty(self, calculate_angle):
        a = (0, 6)
        b = (0, 0)
        c = (0, -3)

        angle = calculate_angle.calculate_angle_using_dot_prod(a, b, c)
        assert angle == 180

    def test_angle_first_quadrant(self, calculate_angle):
        a = (6, 0)
        b = (0, 0)
        c = (6, 6)

        angle = calculate_angle.calculate_angle_using_dot_prod(a, b, c)
        assert angle == 45

    def test_angle_second_quadrant(self, calculate_angle):
        a = (-10, 5)
        b = (0, 5)
        c = (10, 15)

        angle = calculate_angle.calculate_angle_using_dot_prod(a, b, c)
        assert angle == 135

    def test_angle_third_quadrant(self, calculate_angle):
        a = (-15, -10)
        b = (-10, -5)
        c = (-5, -10)

        angle = calculate_angle.calculate_angle_using_dot_prod(a, b, c)
        assert angle == 90

    def test_angle_fourth_quadrant(self, calculate_angle):
        a = (5, -10)
        b = (10, -10)
        c = (10, -5)

        angle = calculate_angle.calculate_angle_using_dot_prod(a, b, c)
        assert angle == 90
