# Copyright (c) 2022, Groq Inc. All rights reserved.
#
# Groq, Inc. and its licensors retain all intellectual property and proprietary
# rights in and to this software, related documentation and any modifications
# thereto. Any use, reproduction, modification, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Groq, Inc. is strictly prohibited.

"""Program Package Example

This script defines 2 programs to load onto a TSP together. The programs use different memory
allocations for their input and output tensors.

Programs can share memory allocations. In this example, Program 2 uses Program 1's result as
a parameter to multiply.

The programs are compiled into a single IOP incrementally. Program 1 is compiled into an IOP.
Then Program 2 is compiled with Program 1's IOP into a combined IOP.
"""

import os
import shutil
from typing import List

import numpy as np

import groq.api as g

try:
    import groq.runtime as runtime
except ImportError:
    raise ModuleNotFoundError("groq.runtime")


def compile(shape) -> List[str]:
    """Compiles a program package with 2 programs.

    Return: (List[str]): A list of IOP files.
    """
    output_dir = "multi_program_example"
    shutil.rmtree(output_dir, ignore_errors=True)
    pgm_pkg = g.ProgramPackage("multi_program_example", output_dir)

    # Defines a program that adds two input tensors
    with pgm_pkg.create_program_context("program_1") as pgm1:
        x = g.input_tensor(shape, g.float32, name="X")
        y = g.input_tensor(shape, g.float32, name="Y")
        z = g.add(x, y, time=10).write(name="Z", program_output=True)

    # Defines a program that adds two input tensors and multiplies the sum by program 1's result
    with pgm_pkg.create_program_context("program_2") as pgm2:
        # Using a ProgramPackage ensures that program 1's persistent memory allocations are considered
        # when allocating memory for program 2.

        # Program 1's output tensor is shared with program 2.
        g.reserve_tensor(pgm1, pgm2, z)
        # Creates a shared tensor to reuse the memory allocation for program 1's output.
        zz = g.shared_memory_tensor(mem_tensor=z, name="new_z")

        a = g.input_tensor(shape, g.float32, name="A")
        b = g.input_tensor(shape, g.float32, name="B")
        c = g.add(a, b, time=10)
        # Multiplies sum `c` with sum `z` from pgm1.
        _d = g.mul(c, zz, input_streams=[g.SG4[0], g.SG4[2]]).write(
            name="D", program_output=True
        )

    iops = pgm_pkg.assemble()
    return iops


def invoke(device, iop, pgm_num, ep_num, tensors):
    """Low level interface to the device driver to access multiple programs. A higher level abstraction
    will be provided in a future release that offers access to multiple programs and entry points."""

    pgm = iop[pgm_num]
    ep = pgm.entry_points[ep_num]
    input_buffer = runtime.BufferArray(ep.input, 1)[0]
    output_buffer = runtime.BufferArray(ep.output, 1)[0]
    if ep.input.tensors:
        for input_tensor in ep.input.tensors:
            if input_tensor.name not in tensors:
                raise ValueError(f"Missing input tensor named {input_tensor.name}")
            input_tensor.from_host(tensors[input_tensor.name], input_buffer)
    device.invoke(input_buffer, output_buffer)
    outs = {}
    if ep.output.tensors:
        for output_tensor in ep.output.tensors:
            result_tensor = output_tensor.allocate_numpy_array()
            output_tensor.to_host(output_buffer, result_tensor)
            outs[output_tensor.name] = result_tensor
    return outs


def run(iop_file, shape):
    """Runs 2 programs back-to-back with the output of the 1st program used as
    the input to the 2nd program.

    This function interacts with the device driver at a lower level to show
    the control of loading 2 programs and invoking each program through
    program entry points. A higher level abstraction will be made available
    to load and invoke multiple programs in a future release.
    """
    np.set_printoptions(linewidth=1000, threshold=10000)

    if not os.path.exists(iop_file):
        raise Exception(f"IOP file does not exist: {iop_file}")

    print(f"Running programs from {iop_file}")

    iop = runtime.IOProgram(iop_file)
    driver = runtime.Driver()
    device = driver.next_available_device()
    with device:
        device.load(iop[0], unsafe_keep_entry_points=True)
        device.load(iop[1], unsafe_keep_entry_points=True)

        x = np.ones(shape, dtype=np.float32) * 1
        y = np.ones(shape, dtype=np.float32) * 2
        pgm_1_output = invoke(device, iop, 0, 0, {"X": x, "Y": y})
        z = pgm_1_output["Z"]

        a = np.ones(shape, dtype=np.float32) * 3
        b = np.ones(shape, dtype=np.float32) * 4
        pgm_2_output = invoke(device, iop, 1, 0, {"new_z": z, "A": a, "B": b})

        print("\nProgram 1 result:")
        print(pgm_1_output["Z"])
        print("\nProgram 2 result:")
        print(pgm_2_output["D"])


def main():
    """Compiles and runs the example programs."""
    shape = (2, 20)

    iop_files = compile(shape)
    print(f"Program compiled to IOP file: {iop_files}")

    run(iop_files[0], shape)


if __name__ == "__main__":
    main()
