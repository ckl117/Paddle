# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from tensorrt_test_base import TensorRTBaseTest

import paddle


class TestMaxTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.max
        self.api_args = {
            "x": np.array([[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]]).astype(
                "float32"
            ),
            "axis": [0, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4], "y": [1, 4]}
        self.max_shape = {"x": [5, 4], "y": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
