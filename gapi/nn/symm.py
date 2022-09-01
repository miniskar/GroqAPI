# Copyright (c) 2022, Groq Inc. All rights reserved.
#
# Groq, Inc. and its licensors retain all intellectual property and proprietary
# rights in and to this software, related documentation and any modifications
# thereto. Any use, reproduction, modification, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Groq, Inc. is strictly prohibited.

"""
Example script for using a nn SYRK component.

To run this example:
>> python syrk.py

This example uses default values of shape (outer,inner) = (100,100), alpha = 2.3, gamma = 1.5
Users could change these parameters by running the command-line interface
>> python symm.py --shape 10 50 --alpha 1.1 --gamma 2.1

For more information on running this example, use:
>> python symm.py --help
"""

# pylint: disable=no-member
# pylint: disable=import-error
import sys
import os
import argparse
import numpy as np
from groq.runner import tsp
import groq.api as g
from groq.api import nn
from groq.common import print_utils

helpers_dir = os.path.join(os.path.dirname(__file__), "../example_helpers")
sys.path.insert(0, helpers_dir)
import helpers


def compile_symm(symm_mt, wt_data, st_data, alpha, gamma):
    """Compile a program that creates a SYMM component with the supplied data"""

    # Initialize input tensors
    symm_matrix = g.input_tensor(shape=symm_mt, dtype=g.float16, name="symm_mt")
    weight_mt = g.input_tensor(shape=wt_data, dtype=g.float16, name="weight_mt")
    state_mt = g.input_tensor(shape=st_data, dtype=g.float32, name="state_mt")

    # Instantiate the SYMM component
    symm_model = nn.SYMM(alpha, gamma, time=0)

    result_mt = symm_model(symm_matrix, weight_mt, state_mt)

    # Create IOP file
    iop_file = g.compile(
        base_name="symm",
        result_tensor=result_mt,
        output_dir=os.getcwd(),
        gen_vis_data=True,
    )

    return iop_file


def runtime_symm(iop_file, symm_mt, wt_data, st_data, alpha, gamma):

    # Before running on hardware, check that hardware is available
    if helpers.get_num_chips_available() is None:
        print_utils.err(
            f"No Hardware Available. Copy {iop_file} to a system with GroqCard™ accelerators installed."
        )
        return

    # Use create tsp runner function to run IOP file
    symm_program = tsp.create_tsp_runner(iop_file)
    actual = symm_program(symm_mt=symm_mt, weight_mt=wt_data, state_mt=st_data)
    symm_result = actual["BLASL3_result"]

    # Compute Oracle.
    weights2_np = np.reshape(wt_data, [outer, inner])
    pro = np.matmul(symm_mt, weights2_np, dtype=np.float32)
    scaled_state = gamma * np.reshape(st_data, [outer, inner])
    product = alpha * pro
    oracle = np.reshape(np.add(scaled_state, product), [outer, inner])
    oracle = np.float32(oracle)

    print_utils.info(f"SYMM result is: \n{symm_result}")
    print_utils.info(f"Numpy oracle result is: \n{oracle}")

    if np.allclose(symm_result, oracle, atol=0.00001):
        print_utils.info("SYMM results match oracle.")
    else:
        print_utils.info(f"Oracle shape={oracle.shape} data=\n{oracle}")
        print_utils.info(f"SYMM result shape={symm_result.shape} data=\n{symm_result}")
        print_utils.err(
            "Unexpected SYRK results. Check that your alpha and gamma values have not changed since you compiled.\n"
        )
    return (actual, oracle)


if __name__ == "__main__":
    # Before getting started, check that the SDK is installed
    if helpers.find_sdk("groq-devtools") is None:
        raise Exception("No SDK Found")

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="SYMM example")
    parser.add_argument(
        "--shape",
        default=[100, 100],
        type=int,
        nargs=2,
        help="Shape of the update matrix (before squaring). Default: 100,100",
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
    outer = args.shape[0]
    inner = args.shape[1]
    compile = args.compile
    runtime = args.runtime

    # Generate random inputs and run the test
    print_utils.info(f"Testing symm with shape [{outer}, {inner}]...")
    weights_shape = (outer, inner)
    weights2_shape = (outer, inner)
    state_shape = (outer, inner)

    # Create activation and weight tensors for FP16 symm.
    wt_data = 10 * np.random.rand(*weights_shape).astype(np.float16).reshape(
        weights_shape
    )
    wt2_data = 10 * np.random.rand(*weights2_shape).astype(np.float16).reshape(
        weights2_shape
    )
    st_data = 100 * np.random.rand(*state_shape).astype(np.float32).reshape(state_shape)

    # get symmetric input matrix
    symm_mt = wt_data @ wt_data.T

    # Run the test
    if not compile and not runtime:
        print_utils.infoc(
            "You have not specified whether you want to compile or run this example. Therefore, we'll assume you wanted to both compile and run this example."
        )
        compile = True
        runtime = True
    if compile:
        compile_symm(symm_mt.shape, weights_shape, state_shape, alpha, gamma)
    if runtime:
        iop_file = os.path.join(os.getcwd(), "symm.iop")
        if os.path.exists(iop_file) is True:
            # TODO: check that runtime input sizes match compiled sizes
            runtime_symm(iop_file, symm_mt, wt2_data, st_data, alpha, gamma)
        else:
            print_utils.err(
                "No IOP file found. Run the entire script using `python symm.py`"
            )
