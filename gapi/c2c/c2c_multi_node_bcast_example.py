"""Build, compile and run C2C broadcast collective op for fully connected A1.1 4-way topology."""

import argparse
import numpy as np
import sys

import groq.api as g
import groq.runner.tsp as tsp  # pylint: disable=import-error

from groq.common import print_utils
from groq.common.config import config

try:
    # pylint: disable=import-error
    from mpi4py import MPI
except Exception as e:  # pylint: disable=broad-except
    MPI = None

if MPI is not None:
    mpi_size = MPI.COMM_WORLD.Get_size()
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_name = MPI.Get_processor_name()
else:
    mpi_size = 1
    mpi_rank = 0
    mpi_name = "Alas simulator"


def create_data(input_tensor, result_tensors, input):
    # Create random data for each input tensor.
    shape = input_tensor.shape
    nptype = input_tensor.dtype.to_nptype()
    input_data = np.full(shape, input, dtype=nptype)

    # Build input dictionary.
    inputs = {input_tensor.name: input_data}

    # Build oracle from src data.
    oracles = {mt.name: input_data for mt in result_tensors}

    return inputs, oracles


def check_results(results, oracles) -> bool:
    all_okay = True
    for result_key, result in results.items():
        try:
            oracle = oracles[result_key]
            np.testing.assert_allclose(result, oracle, rtol=1e-5)
        except AssertionError as exc:
            print_utils.err(f"Result mismatch for tensor '{result_key}' =>")
            print_utils.warn(f"Comparing (result, oracle): {exc}")
            all_okay = False
            continue
        except KeyError as exc:
            print_utils.err(f"Key Error for tensor '{result_key}' => {exc}")
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
            user_config == g.TopologyConfig.DF_A14_8_CHIP
            or user_config == g.TopologyConfig.DF_A14_16_CHIP
            or user_config == g.TopologyConfig.DF_A14_32_CHIP
            or user_config == g.TopologyConfig.DF_A14_64_CHIP
        ):
            # Perform c2c broadcast operation to all devices on the node
            dst_devices = [g.device(i) for i in range(1, 8, 1)]
        else:
            raise Exception("Unsupported config")

        result_tensors0 = g.c2c_broadcast(input_tensor, devices=dst_devices, time=10)
        result_tensors = result_tensors0

        if (
            user_config == g.TopologyConfig.DF_A14_16_CHIP
            or user_config == g.TopologyConfig.DF_A14_32_CHIP
            or user_config == g.TopologyConfig.DF_A14_64_CHIP
        ):
            # for node1
            with g.device(1):
                dst_devices = [g.device(8)]
                input_for_node1 = result_tensors0[0]

                result_tensors1 = g.c2c_broadcast(
                    input_for_node1, devices=dst_devices, time=1000
                )
                result_tensors = result_tensors + result_tensors1
            # node1
            with g.device(8):
                node1_input = result_tensors1[0]
                # Perform c2c broadcast operation to devices 9,10,11,12,13.
                dst_devices = [g.device(i) for i in range(9, 16, 1)]

                result_tensors2 = g.c2c_broadcast(
                    node1_input, devices=dst_devices, time=2000
                )
                result_tensors = result_tensors + result_tensors2

        if (
            user_config == g.TopologyConfig.DF_A14_32_CHIP
            or user_config == g.TopologyConfig.DF_A14_64_CHIP
        ):
            # for node2
            with g.device(9):
                dst_devices = [g.device(16)]
                input_for_node2 = result_tensors2[0]

                result_tensors3 = g.c2c_broadcast(
                    input_for_node2, devices=dst_devices, time=3000
                )
                result_tensors = result_tensors + result_tensors3
            # node 2
            with g.device(16):
                dst_devices = [g.device(i) for i in range(17, 24, 1)]
                node2_input = result_tensors3[0]

                result_tensors4 = g.c2c_broadcast(
                    node2_input, devices=dst_devices, time=4000
                )
                result_tensors = result_tensors + result_tensors4
            # for node3
            with g.device(17):
                dst_devices = [g.device(24)]
                input_for_node3 = result_tensors4[0]

                result_tensors5 = g.c2c_broadcast(
                    input_for_node3, devices=dst_devices, time=5000
                )
                result_tensors = result_tensors + result_tensors5
            # node 3
            with g.device(24):
                dst_devices = [g.device(i) for i in range(25, 32, 1)]
                node3_input = result_tensors5[0]

                result_tensors6 = g.c2c_broadcast(
                    node3_input, devices=dst_devices, time=6000
                )
                result_tensors = result_tensors + result_tensors6

        if user_config == g.TopologyConfig.DF_A14_64_CHIP:
            # for node4
            with g.device(25):
                dst_devices = [g.device(32)]
                input_for_node4 = result_tensors6[0]

                result_tensors7 = g.c2c_broadcast(
                    input_for_node4, devices=dst_devices, time=7000
                )
                result_tensors = result_tensors + result_tensors7
            # node 4
            with g.device(32):
                dst_devices = [g.device(i) for i in range(33, 40, 1)]
                node4_input = result_tensors7[0]

                result_tensors8 = g.c2c_broadcast(
                    node4_input, devices=dst_devices, time=8000
                )
                result_tensors = result_tensors + result_tensors8
            # for node 5
            with g.device(33):
                dst_devices = [g.device(40)]
                input_for_node5 = result_tensors8[0]

                result_tensors9 = g.c2c_broadcast(
                    input_for_node5, devices=dst_devices, time=9000
                )
                result_tensors = result_tensors + result_tensors9
            # node 5
            with g.device(40):
                dst_devices = [g.device(i) for i in range(41, 48, 1)]
                node5_input = result_tensors9[0]
                result_tensors10 = g.c2c_broadcast(
                    node5_input, devices=dst_devices, time=10000
                )
                result_tensors = result_tensors + result_tensors10
            # for node6
            with g.device(41):
                dst_devices = [g.device(48)]
                input_for_node6 = result_tensors10[0]

                result_tensors11 = g.c2c_broadcast(
                    input_for_node6, devices=dst_devices, time=11000
                )
                result_tensors = result_tensors + result_tensors11
            # node 6
            with g.device(48):
                dst_devices = [g.device(i) for i in range(49, 56, 1)]
                node6_input = result_tensors11[0]
                result_tensors12 = g.c2c_broadcast(
                    node6_input, devices=dst_devices, time=12000
                )
                result_tensors = result_tensors + result_tensors12
            # for node7
            with g.device(49):
                dst_devices = [g.device(56)]
                input_for_node7 = result_tensors12[0]

                result_tensors13 = g.c2c_broadcast(
                    input_for_node7, devices=dst_devices, time=13000
                )
                result_tensors = result_tensors + result_tensors13
            # node 7
            with g.device(56):
                dst_devices = [g.device(i) for i in range(57, 64, 1)]
                node7_input = result_tensors13[0]
                result_tensors14 = g.c2c_broadcast(
                    node7_input, devices=dst_devices, time=14000
                )
                result_tensors = result_tensors + result_tensors14

        for result_t in result_tensors:
            result_t.set_program_output()

    return input_tensor, result_tensors


def get_config_from_topo_str(topo_str):
    if topo_str == "A14_8C":
        return g.TopologyConfig.DF_A14_8_CHIP
    elif topo_str == "A14_16C":
        if mpi_size != 2:
            print_utils.err(f"Invalid mpi_size: {mpi_size} for {topo_str}")
            sys.exit(-1)
        return g.TopologyConfig.DF_A14_16_CHIP
    elif topo_str == "A14_32C":
        if mpi_size != 4:
            print_utils.err(f"Invalid mpi_size: {mpi_size} for {topo_str}")
            sys.exit(-1)
        return g.TopologyConfig.DF_A14_32_CHIP
    elif topo_str == "A14_64C":
        if mpi_size != 8:
            print_utils.err(f"Invalid mpi_size: {mpi_size} for {topo_str}")
            sys.exit(-1)
        return g.TopologyConfig.DF_A14_64_CHIP
    else:
        print_utils.err("Unsupported config")
        sys.exit(-1)


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
        default="A14_16C",
        choices=[
            "A14_8C",
            "A14_16C",
            "A14_32C",
            "A14_64C",
        ],
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
    args = parser.parse_args()

    user_config = get_config_from_topo_str(args.topo_str)

    # Step 4 [Optional]: Bringup the c2c links before we run the program.
    # link up procedure is now embedded in multi_tsp_runner, it will
    # check for the link status before it begins invoking the programs
    # the user can force the linkup by passing --bringup option
    if args.bringup:
        print_utils.infoc(
            f"Bringup C2C topology {mpi_name} {mpi_size}, {mpi_rank}..."
        )
        try:
            tsp.bringup_topology(
                user_config=user_config,
                speed=args.speed,
            )
        except Exception as e:  # pylint: disable=broad-except
            print_utils.infoc("Aborting, " + str(e))
            sys.exit(1)

        print_utils.infoc("Bringup done")

    # Step 1: Instantiate a program package to store multi-chip (C2C)
    # or single-chip programs.
    pkg_name = "c2c_multi_hop"
    pkg_dir = config.get_tmp_dir(pkg_name)
    print_utils.infoc(f"Creating a program package '{pkg_name}' at '{pkg_dir}' ...")
    pgm_pkg = g.ProgramPackage(name=pkg_name, output_dir=pkg_dir)

    # Step 2: Build your multi-chip C2C program.
    prog_name = "c2c_multi_hop"
    input_tensor, result_tensors = build_program(
        user_config, pgm_pkg, prog_name, args.speed
    )

    # You are free to add more multi-chip or single-chip programs to the package.

    # Step 3: Assemble all programs in the multi-device package.
    print_utils.infoc(f"Assembling multi-device package '{pkg_name}' ...")
    pgm_pkg.assemble()

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
        )
    except Exception as e:  # pylint: disable=broad-except
        print_utils.infoc("Aborting, " + str(e))
        sys.exit(1)

    # Step 6: Pass inputs to the runner and execute the program on HW.
    # inputs, oracles = create_data(input_tensor, result_tensors, 0)

    # np.savez(f"{pkg_dir}/{pkg_name}.input.npz", **inputs)
    # np.savez(f"{pkg_dir}/{pkg_name}.oracle.npz", **oracles)

    print_utils.infoc(f"Executing C2C program '{prog_name}' ...")

    try:
        for i in range(args.iter):
            inputs, oracles = create_data(input_tensor, result_tensors, (i % 16))
            print_utils.infoc(f"Testing for iteration: {i}")

            results = runner(**inputs)

            # Validation: Compare against oracle.
            print_utils.infoc("Validating results ...")

            # Results can be checked at the individual nodes
            # or we can choose to send it to the root node for the
            # verification. For this example, check the results locally on each node
            check_results(results, oracles)
    except KeyboardInterrupt:
        print_utils.infoc("Program Interrupted... Terminating the program")
    finally:
        print_utils.infoc("Test run completed")
    config.rm_tmp_dir(pkg_name)
