"""
Module for Data Controller testing.
"""

__author__ = "Romanyuk Artem"
__copyright__ = "Copyright (c) 2024 Romanyuk Artem"
__license__ = "SPDX-License-Identifier: MIT"

import random

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from pysatl_cpd.core.data_controller import DataController


class TestDataController:
    @settings(max_examples=1000)
    @given(st.integers(0, 100), st.integers(0, 100), st.integers(0, 100), st.integers(0, 100))
    def test_restart(self, data_len, window_length, change_points_len, data_start_index):
        data = [random.randint(0, 100) for _ in range(data_len)]
        change_points = [random.randint(0, 100) for _ in range(change_points_len)]
        data_controller = DataController(data, window_length)
        data_controller.change_points = change_points
        data_controller._data_start_index = data_start_index
        data_controller.restart()
        assert data_controller._data_start_index == 0
        assert data_controller.change_points == []

    @settings(max_examples=1000)
    @given(st.integers(1, 100), st.integers(1, 100))
    def test_get_data(self, data_length, window_length):
        data = [random.randint(0, 100) for _ in range(data_length)]
        data_start_index = random.randint(0, data_length)
        data_controller = DataController(data, window_length)
        data_controller._data_start_index = data_start_index
        if data_start_index == data_length:
            with pytest.raises(StopIteration):
                next(iter(data_controller.get_data()))
        else:
            assert next(iter(data_controller.get_data())) == data[data_start_index : data_start_index + window_length]

    @settings(max_examples=1000)
    @given(st.integers(1, 100), st.integers(1, 100), st.integers(1, 100), st.integers(1, 100))
    def test_add_change_points(self, data_length, window_length, window_start_index, change_points_length):
        data = [random.randint(0, 100) for _ in range(data_length)]
        change_points = [random.randint(0, 100) for _ in range(change_points_length)]

        data_controller = DataController(data, window_length)
        data_controller._data_start_index = window_start_index
        data_controller.add_change_points(change_points)
        assert data_controller.change_points == list(map(lambda point: point + window_start_index, change_points))
