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
from functools import partial
from typing import List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertAtan2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([1, 80, 1]).astype(np.float32)

        def generate_input2():
            return np.random.random([1, 80, 1]).astype(np.float32)

        ops_config = [
            {
                "op_type": "atan2",
                "op_inputs": {
                    "X1": ["input_data1"],
                    "X2": ["input_data2"],
                },
                "op_outputs": {
                    "Out": [
                        "output_data0",
                    ]
                },
                "op_attrs": {},
            }
        ]
        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data1": TensorConfig(data_gen=partial(generate_input1)),
                "input_data2": TensorConfig(data_gen=partial(generate_input2)),
            },
            outputs=["output_data0"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 3

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data1": [1, 80, 1],
                "input_data2": [1, 80, 1],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data1": [2, 80, 1],
                "input_data2": [2, 80, 1],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data1": [1, 80, 1],
                "input_data2": [1, 80, 1],
            }

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        # clear_dynamic_shape()

        # self.trt_param.precision = paddle_infer.PrecisionType.Float32
        # yield self.create_inference_config(), (0, 6), 1e-5
        # self.trt_param.precision = paddle_infer.PrecisionType.Half
        # yield self.create_inference_config(), (0, 6), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
