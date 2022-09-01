# Copyright (c) 2022, Groq Inc. All rights reserved.
#
# Groq, Inc. and its licensors retain all intellectual property and proprietary
# rights in and to this software, related documentation and any modifications
# thereto. Any use, reproduction, modification, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Groq, Inc. is strictly prohibited.

"""
Example script for using a nn GEMM component.
This example uses default values of shape (outer,inner) = (100,100), alpha = 2.3, gamma = 1.5
Users could change these parameters by running the command-line interface.
e.g. python gemm.py --shape 10 50 --alpha 1.1 --gamma 2.1
"""
# pylint: disable=no-member
# pylint: disable=import-error
import sys
import os
import argparse
import numpy as np
import groq.api as g
from groq.runner import tsp
from groq.api import nn
from groq.common import print_utils

helpers_dir = os.path.join(os.path.dirname(__file__), "../example_helpers")
sys.path.insert(0, helpers_dir)
import helpers


def compile_gemm(wt1_shape, wt2_shape, st_shape, alpha, gamma, w1_op, w2_op):
    """
    Using GEMM component with supplied data.
    """
    print_utils.infoc("Creating GEMM model")
    # Initialize input tensors
    weights1_mt = g.input_tensor(shape=wt1_shape, dtype=g.float16, name="weights1_mt")
    weights2_mt = g.input_tensor(shape=wt2_shape, dtype=g.float16, name="weights2_mt")
    state_mt = g.input_tensor(shape=st_shape, dtype=g.float32, name="state_mt")

    # Instantiate the GEMM component
    gemm_model = nn.GEMM(alpha, gamma, time=0)
    result_mt = gemm_model(weights1_mt, weights2_mt, state_mt, w1_op=w1_op, w2_op=w2_op)

    print_utils.infoc("Compiling GEMM...")
    # Create IOP file with tensor data from Groq API compile function
    iop_file = g.compile(
        base_name="gemm",
        result_tensor=result_mt,
        gen_vis_data=True,
        output_dir=os.getcwd(),
    )

    print_utils.infoc(f"Successfully compiled and save IOP file at: {iop_file}")

    return iop_file


def run_gemm(iop_file, wt1_data, wt2_data, st_data, alpha, gamma, w1_op, w2_op):
    # Before running on hardware, check that hardware is available
    if helpers.get_num_chips_available() is None:
        print_utils.err(
            f"No Hardware Available. Copy {iop_file} to a system with GroqCard™ accelerators installed."
        )
        return

    print_utils.infoc("Running GEMM on hardware")
    # Create program instance with TSP runner and pass in IOP file to run
    gemm_program = tsp.create_tsp_runner(iop_file)
    actual = gemm_program(weights1_mt=wt1_data, weights2_mt=wt2_data, state_mt=st_data)
    gemm_result = actual["BLASL3_result"]

    # Compute Oracle.
    process_mt = (
        lambda mt, op: mt if not op else (np.transpose(mt) if op == "T" else mt)
    )

    weights1_np = process_mt(wt1_data, w1_op)
    weights2_np = process_mt(wt2_data, w2_op)

    pro = np.matmul(weights1_np, weights2_np, dtype=np.float32)
    scaled_state = gamma * st_data
    product = alpha * pro

    oracle = np.add(scaled_state, product)
    oracle = np.float32(oracle)

    print_utils.infoc(f"GEMM result is: {gemm_result}\n")
    print_utils.infoc(f"Numpy oracle result is: {oracle}\n")

    if np.allclose(gemm_result, oracle, atol=0.00001):
        print_utils.infoc("GEMM results match oracle.")
    else:
        print_utils.infoc(f"Oracle shape={oracle.shape} data=\n{oracle}")
        print_utils.infoc(f"GEMM result shape={actual.shape} data=\n{gemm_result}")
        print_utils.err(
            "Unexpected GEMM results. Check that your alpha and gamma values have not changed since you compiled.\n"
        )

    return (actual, oracle)


if __name__ == "__main__":
    # Before getting started, check that the SDK is installed
    if helpers.find_sdk("groq-devtools") is None:
        raise Exception("No SDK Found")

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="GEMM example")
    parser.add_argument(
        "--shape",
        default=[100, 100],
        type=int,
        nargs=2,
        help="Size of the update matrices A and B. Default: 100",
    )
    parser.add_argument(
        "--ops",
        default=[None, "T"],
        type=str,
        nargs=2,
        help="Operations of the update matrices A and B. Default: None, 'T'",
    )
    parser.add_argument(
        "--alpha",
        default=[2.3],
        type=float,
        nargs=1,
        help="Scaling factor for the squared update matrix. Default: 2.3",
    )
    parser.add_argument(
        "--gamma",
        default=[1.5],
        type=float,
        nargs=1,
        help="Scaling factor for the state matrix. Default: 1.5",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use this flag to compile the example but not run on hardware.",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Use this flag to run the previously compiled IOP file.",
    )

    args = parser.parse_args()
    alpha = args.alpha[0]
    gamma = args.gamma[0]
    size = args.shape[0]
    compile = args.compile
    runtime = args.runtime

    # Generate random inputs and run the test
    print("Testing gemm with shape", [size, size], "...")

    weights_shape = (size, size)
    weights2_shape = (size, size)

    state_shape = (size, size)

    w1_op = args.ops[0] if args.ops[0] != "None" else None
    w2_op = args.ops[1] if args.ops[1] != "None" else None

    # Create activation and weight tensors for FP16 gemm.
    wt_data = 10 * np.random.rand(*weights_shape).astype(np.float16).reshape(
        weights_shape
    )
    wt2_data = 10 * np.random.rand(*weights2_shape).astype(np.float16).reshape(
        weights2_shape
    )
    st_data = 100 * np.random.rand(*state_shape).astype(np.float32).reshape(state_shape)

    if not compile and not runtime:
        print_utils.infoc(
            "You have not specified whether you want to compile or run this example. Therefore, we'll assume you wanted to both compile and run this example."
        )
        compile = True
        runtime = True
    if compile:
        iop_file = compile_gemm(
            weights_shape, weights2_shape, state_shape, alpha, gamma, w1_op, w2_op
        )
    if runtime:
        iop_file = os.path.join(os.getcwd(), "gemm.iop")
        if os.path.exists(iop_file) is True:
            # TODO: check that runtime input sizes match compiled sizes
            run_gemm(iop_file, wt_data, wt2_data, st_data, alpha, gamma, w1_op, w2_op)
        else:
            print_utils.err(
                "No IOP file found. Run the entire script using `python unpack_multibyte.py`"
            )
