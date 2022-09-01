"""Build, compile and run C2C broadcast collective op for fully connected A1.1 4-way topology."""

import argparse
import numpy as np
import sys

import groq.api as g
import groq.runner.tsp as tsp  # pylint: disable=import-error
from groq.common import print_utils
from groq.common.config import config
import time


def create_data(input_tensor, result_tensors):
    # Create random data for each input tensor.
    shape = input_tensor.shape
    nptype = input_tensor.dtype.to_nptype()
    input_data = (np.random.rand(*shape) * 10.0).astype(nptype)

    # Build input dictionary.
    inputs = {input_tensor.name: input_data}

    # Build oracle from src data.
    oracles = {mt.name: input_data for mt in result_tensors}

    return inputs, oracles


def check_results(results, oracles) -> bool:
    all_okay = True
    for result_key, result in results.items():
        oracle = oracles[result_key]
        try:
            np.testing.assert_allclose(result, oracle, rtol=1e-5)
        except AssertionError as exc:
            print_utils.err(f"Result mismatch for tensor '{result_key}' =>")
            print_utils.warn(f"Comparing (result, oracle): {exc}")
            all_okay = False
            continue
        print_utils.success(f"Result matched the oracle for tensor '{result_key}'")
    return all_okay


def build_program(user_config, pgm_pkg, prog_name, speed):
    # Setup multi-chip topology and create a new program context.
    topo = g.configure_topology(config=user_config, speed=speed)
    print_utils.infoc(
        f"Building C2C program '{prog_name}' with '{topo.name}' topology ..."
    )
    pg_ctx = pgm_pkg.create_program_context(prog_name, topo)

    with pg_ctx:
        shape = (2, 200)
        dtype = g.float32

        # Initialize input tensor on device 0.
        with g.device(0):
            input_tensor = g.input_tensor(shape, dtype, name="c2c_inp_d0")

        if (
            user_config == g.TopologyConfig.FC2_A11_2_CHIP
            or user_config == g.TopologyConfig.DF_A14_2_CHIP
        ):
            # Perform c2c broadcast operation to devices 1 and 2.
            dst_devices = [g.device(1)]
        elif (
            user_config == g.TopologyConfig.FC2_A11_4_CHIP
            or user_config == g.TopologyConfig.DF_A14_4_CHIP
        ):
            # Perform c2c broadcast operation to devices 1 and 3.
            dst_devices = [g.device(1), g.device(3)]
        elif user_config == g.TopologyConfig.DF_A14_8_CHIP:
            # Perform c2c broadcast operation to devices 1,3,5,7.
            dst_devices = [g.device(1), g.device(3), g.device(5), g.device(7)]
        else:
            # Default bcast it to device 1 and 3
            dst_devices = [g.device(1), g.device(3)]

        result_tensors = g.c2c_broadcast(input_tensor, devices=dst_devices, time=10)
        for result_t in result_tensors:
            result_t.set_program_output()

    return input_tensor, result_tensors


def get_config_from_topo_str(topo_str):
    if topo_str == "A11_2C":
        return g.TopologyConfig.FC2_A11_2_CHIP
    elif topo_str == "A11_4C":
        return g.TopologyConfig.FC2_A11_4_CHIP
    elif topo_str == "A14_2C":
        return g.TopologyConfig.DF_A14_2_CHIP
    elif topo_str == "A14_4C":
        return g.TopologyConfig.DF_A14_4_CHIP
    elif topo_str == "A14_8C":
        return g.TopologyConfig.DF_A14_8_CHIP
    else:
        assert False, f"Unsupported topology {topo_str}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C2C collectives example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bringup",
        action="store_true",
        help="Bringup C2C links for given topology",
    )
    parser.add_argument(
        "--topo_str",
        type=str,
        default="A11_4C",
        choices=["A11_2C", "A11_4C", "A14_2C", "A14_4C", "A14_8C"],
        help="Topology type to run this C2C program: \n",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=25.78125,
        choices=[25, 25.78125, 30],
        help="Link bringup speed: 25, 25.78125 or 30G\n",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
        help="Iterations to run the program\n",
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Topology instance on the node\n",
    )
    args = parser.parse_args()

    # Step 1: Instantiate a program package to store multi-chip (C2C)
    # or single-chip programs.
    pkg_name = "c2c_pkg"
    pkg_dir = config.get_tmp_dir(pkg_name)
    print_utils.infoc(f"Creating a program package '{pkg_name}' at '{pkg_dir}' ...")
    pgm_pkg = g.ProgramPackage(name=pkg_name, output_dir=pkg_dir)

    # Step 2: Build your multi-chip C2C program.
    prog_name = "c2c_bcast"
    user_config = get_config_from_topo_str(args.topo_str)
    input_tensor, result_tensors = build_program(
        user_config, pgm_pkg, prog_name, args.speed
    )

    # You are free to add more multi-chip or single-chip programs to the package.

    # Step 3: Assemble all programs in the multi-device package.
    print_utils.infoc(f"Assembling multi-device package '{pkg_name}' ...")
    pgm_pkg.assemble()

    # Step 4 [Optional]: Bringup the c2c links before we run the program.
    # link up procedure is now embedded in multi_tsp_runner, it will
    # check for the link status before it begins invoking the programs
    # the user can force the linkup by passing --bringup option
    if args.bringup:
        print_utils.infoc("Bringup C2C topology ...")
        try:
            tsp.bringup_topology(
                user_config=user_config,
                speed=args.speed,
                instance_id=args.instance,
            )
        except Exception as e:  # pylint: disable=broad-except
            print_utils.infoc("Aborting, " + str(e))
            sys.exit(1)

    # Step 5: Create a multi-tsp runner
    # Make sure to pass the program name to be executed.
    print_utils.infoc("Creating multi-tsp runner ...")
    try:
        runner = tsp.create_multi_tsp_runner(
            pkg_name,
            pkg_dir,
            prog_name,
            user_config=user_config,
            speed=args.speed,
            instance_id=args.instance,
        )
    except Exception as e:  # pylint: disable=broad-except
        print_utils.infoc("Aborting, " + str(e))
        sys.exit(1)

    # Step 6: Pass inputs to the runner and execute the program on HW.
    inputs, oracles = create_data(input_tensor, result_tensors)
    print_utils.infoc(f"Executing C2C program '{prog_name}' ...")
    try:
        for i in range(args.iter):
            print_utils.infoc(f"Testing for iteration: {i}")
            time.sleep(0.5)
            results = runner(**inputs)

            # Validation: Compare against oracle.
            print_utils.infoc("Validating results ...")
            assert check_results(results, oracles), "Validation failed!"
    except KeyboardInterrupt:
        print_utils.infoc("Program Interrupted... Terminating the program")
    finally:
        print_utils.infoc("Test run completed")
