# Copyright (c) 2020-2021, Groq Inc. All rights reserved.
#
# Groq, Inc. and its licensors retain all intellectual property and proprietary
# rights in and to this software, related documentation and any modifications
# thereto. Any use, reproduction, modification, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Groq, Inc. is strictly prohibited.

"""A basic GroqAPI example.

In this example, we showcase how to do a simple addition between 2 int16
tensors.  This example will build and compile the model, generate an executable
binary and run on your GroqChip.  The data for the 2 tensors will be supplied
"dynamically" from the host.
"""

import sys
import shutil

import numpy as np

import groq.api as g
from groq.runner import tsp
from groq.common import print_utils


def main(iop_file):

    # Build the model.
    print_utils.infoc("\nBuilding model ...")
    shape = (2, 320)
    t1 = g.input_tensor(shape, g.int16, name="t1")
    t2 = g.input_tensor(shape, g.int16, name="t2")
    result_st = t1.add(t2, input_streams=["SG4_0", "SG4_1"], time=15)
    result_mt = result_st.write(name="result", program_output=True)

    # Compile the model and generate the executable binary.
    print_utils.infoc("\nCompiling model ...")
    compiled_iop = g.compile(base_name="add_tensors", gen_vis_data=True)

    if iop_file is not None and compiled_iop != iop_file:
        shutil.copyfile(compiled_iop, iop_file)
        compiled_iop = iop_file
    print_utils.infoc(f"IOP generated at '{compiled_iop}'")

    # Generate random input data and oracle for comparision.
    lo, hi = np.iinfo(np.int16).min // 2, np.iinfo(np.int16).max // 2
    inp1 = np.random.randint(lo, hi, size=t1.shape, dtype=np.int16)
    inp2 = np.random.randint(lo, hi, size=t2.shape, dtype=np.int16)

    oracle = inp1 + inp2

    print_utils.infoc("\nRunning on HW ...")
    # Create TSP runner and pass input dictionary of "tensor name" : "tensor data".
    runner = tsp.create_tsp_runner(compiled_iop)
    inputs = {t1.name: inp1, t2.name: inp2}
    results = runner(**inputs)

    print_utils.infoc("\nComparing results with oracle ...")
    actual = results[result_mt.name]
    max_atol = max(abs(oracle.reshape(-1) - actual.reshape(-1)))
    if max_atol == 0:
        print_utils.success(f"Test PASSED with a max tolerance of {max_atol}")
    else:
        print_utils.err(
            f"Test FAILED with a max tolerance of {max_atol} (should be = 0)"
        )


if __name__ == "__main__":
    iop_file = None
    if len(sys.argv) == 2:
        iop_file = sys.argv[1]
    elif len(sys.argv) > 2:
        sys.exit(f"usage: python {sys.argv[0]} [output.iop]")

    main(iop_file)
