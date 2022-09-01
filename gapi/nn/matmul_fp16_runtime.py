# Copyright (c) 2020-2022, Groq Inc. All rights reserved.
#
# Groq, Inc. and its licensors retain all intellectual property and proprietary
# rights in and to this software, related documentation and any modifications
# thereto.

"""Example script for using the Neural Network component: MatMul

This script run the precompiled IOP file created by running the matmul_fp16_compile.py script.
To run this example:

$ python matmul_fp16_runtime.py

This example uses input tensors as inputs and writes to program output.
"""
# pylint: disable=import-error
import numpy as np
from groq.runner import tsp
from groq.common import print_utils
import sys
import os

helpers_dir = os.path.join(os.path.dirname(__file__), "../example_helpers")
sys.path.insert(0, helpers_dir)
import helpers


def main():

    # Before getting started, check that the SDK is installed
    if helpers.find_sdk("groq-runtime") is not True:
        return

    iop_file = os.path.join(os.getcwd(), "mm_fp16.iop")

    # Create 2 input tensors matching the size of the compiled tensors
    t1_shape = (100, 1000)
    t2_shape = (400, 1000)

    # Generate random input data and oracle for comparision.
    inp1 = np.random.rand(t1_shape[0], t1_shape[1]).astype(np.float16)
    inp2 = np.random.rand(t2_shape[0], t2_shape[1]).astype(np.float16)
    oracle = np.matmul(inp1, inp2.transpose(), dtype=np.float32)

    # Before running on hardware, check that the runtime is installed
    if helpers.get_num_chips_available() is None:
        print_utils.err(
            f"No Hardware Available. Copy {iop_file} to a system with GroqCard™ accelerators installed."
        )
        return

    print_utils.infoc("\nRunning on HW ...")

    # Create TSP runner and pass input dictionary of "tensor name" : "tensor data".
    runner = tsp.create_tsp_runner(iop_file)
    inputs = {"A": inp1, "B": inp2}
    results = runner(**inputs)

    print_utils.infoc("\nComparing results with oracle ...")
    actual = results["MatMul_result"]
    max_atol = max(abs(oracle.reshape(-1) - actual.reshape(-1)))
    if max_atol <= 0.001:
        print_utils.success(f"Test PASSED with a max tolerance of {max_atol}")
    else:
        print_utils.err(
            f"Test FAILED with a max tolerance of {max_atol} (should be <= 0.001)"
        )


if __name__ == "__main__":
    main()
