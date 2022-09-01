# Copyright (c) 2022, Groq Inc. All rights reserved.
#
# Groq, Inc. and its licensors retain all intellectual property and proprietary
# rights in and to this software, related documentation and any modifications
# thereto. Any use, reproduction, modification, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Groq, Inc. is strictly prohibited.

""" Unpack Multibyte Example

To run this example:
>> python unpack_multibyte.py

For more information on running this example, use:
>> python syrk.py --help

This example sets the input tensor format to "CONTIGUOUS" to instruct the
host runtime to serialize multibyte values into the vector's inner dimension.

`UnpackBytes` will transpose the multibyte values across vectors into the
architecture's native format.

The program's result tensor is in the default "STRIDED" format to instruct the
host runtime to transpose the multibyte values back into a serialized format.
"""
# pylint: disable=no-member
# pylint: disable=import-error
import os
import sys
import numpy as np
import groq.api as g
from groq.runner import tsp
import groq.api.nn as nn
from groq.common import print_utils
from typing import cast
import argparse

helpers_dir = os.path.join(os.path.dirname(__file__), "../example_helpers")
sys.path.insert(0, helpers_dir)
import helpers


def build_program(lshape, hemi="W"):
    print_utils.infoc("Beginning build of program")
    np.random.seed(15)

    input_layout = f"-1, H1({hemi}), S4(inner)"

    # Create input tensor.
    inp_t = g.input_tensor(
        lshape,
        g.float32,
        "input_contiguous",
        format=g.MultiByteDataFormat.CONTIGUOUS,
        layout=input_layout,
    )
    comp = nn.UnpackBytes(time=0, hemi=hemi)
    result_mt: g.MemoryTensor = cast(g.MemoryTensor, comp(inp_t))
    result_mt.name = "result_mt"

    print_utils.infoc("Compiling program...")
    iop_file = g.compile("multibyte_example", result_mt, output_dir=os.getcwd())
    print_utils.infoc(f"\nModel successfully compiled and saved at {iop_file}")
    return iop_file


def run_program(iop_file, shape: g.Shape):

    # Before running on hardware, check that hardware is available
    if helpers.get_num_chips_available() is None:
        print_utils.err(
            f"No Hardware Available. Copy {iop_file} to a system with GroqCard™ accelerators installed."
        )
        return

    print_utils.infoc("Running Multibyte Example on hardware...")
    pgm = tsp.create_tsp_runner(iop_file)

    data = np.random.rand(*shape).astype(np.float32)
    result = pgm(input_contiguous=data)

    if np.array_equal(data, result["result_mt"]):
        print_utils.infoc(
            "Returned value from Groq hardware matches the input provided"
        )
    else:
        print_utils.err(
            "The input provided does not match the results returned. Please recompile and try again."
        )


if __name__ == "__main__":
    # Before getting started, check that the SDK is installed
    if helpers.find_sdk("groq-devtools") is None:
        raise Exception("No SDK Found")

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Unpack multibyte example")
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

    compile = args.compile
    runtime = args.runtime

    shape = (112, 320)

    if not compile and not runtime:
        print_utils.infoc(
            "You have not specified whether you want to compile or run this example. Therefore, we'll assume you wanted to both compile and run this example."
        )
        compile = True
        runtime = True
    if compile:
        iop_file = build_program(shape, hemi="W")
    if runtime:
        # TODO: check that runtime input sizes match compiled sizes
        # Get IOP file of program
        iop_file = os.path.join(os.getcwd(), "multibyte_example.iop")
        if os.path.exists(iop_file) is True:
            run_program(iop_file, shape)
        else:
            print_utils.err(
                "No IOP file found. Run the entire script using `python unpack_multibyte.py`"
            )
