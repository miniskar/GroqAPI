"""Build, compile and run C2C broadcast collective op for fully connected A1.1 4-way topology."""

import sys

import argparse
import numpy as np

import groq.api as g
import groq.runner.tsp as tsp  # pylint: disable=import-error

from groq.common import print_utils
from groq.common.config import config

try:
    # pylint: disable=import-error
    from mpi4py import MPI
except ImportError as e:
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
    # input_data = (np.random.rand(*shape) * 10.0).astype(nptype)
    input_data = np.full(shape, input, dtype=nptype)
    output_data = np.full(shape, input * 2 * 2, dtype=nptype)

    # Build input dictionary.
    inputs = {input_tensor.name: input_data}

    # Build oracle from src data.
    oracles = {result_tensors.name: output_data}

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

        # send vectors over the inter-node links
        if user_config == g.TopologyConfig.DF_A14_2_CHIP:
            from_dev = g.device(0)
            to_dev = g.device(1)
        elif user_config == g.TopologyConfig.DF_A14_4_CHIP:
            from_dev = g.device(0)
            to_dev = g.device(1)
        elif user_config == g.TopologyConfig.DF_A14_8_CHIP:
            from_dev = g.device(0)
            to_dev = g.device(3)
        elif user_config == g.TopologyConfig.DF_A14_16_CHIP:
            from_dev = g.device(1)
            to_dev = g.device(8)
        elif user_config == g.TopologyConfig.DF_A14_32_CHIP:
            from_dev = g.device(17)
            to_dev = g.device(24)
        elif user_config == g.TopologyConfig.DF_A14_64_CHIP:
            from_dev = g.device(49)
            to_dev = g.device(56)
        else:
            print(f"Invalid topology type, aborting {user_config}")
            sys.exit(-1)

        with from_dev:
            input_tensor = g.input_tensor(
                name="c2c_inp", shape=(4, 320), dtype=g.int8
            )

            transferred_st = g.transmit(
                input=input_tensor, device_tx=None, device_rx=to_dev, time=100
            )

        with to_dev:
            transferred_mt = transferred_st.write(layout="H1(E),-1,S1")

            added_st = transferred_mt.add(
                alus=[12], input=transferred_mt, output_streams=g.SG4[6], time=10000
            )

            added_mt = added_st.write(layout="H1(W),-1,S1")

            final_st = g.transmit(
                added_mt, device_tx=None, device_rx=from_dev, time=20000
            )

        with from_dev:

            result_tensors = final_st.write()

            added_new_st = result_tensors.add(
                alus=[12], input=result_tensors, output_streams=g.SG4[6], time=30000
            )

            added_new_mt = added_new_st.write(layout="H1(W),-1,S2")

            # result_tensors.set_program_output()
            added_new_mt.set_program_output()

    return input_tensor, added_new_mt


def get_config_from_topo_str(topo_str):
    if topo_str == "A14_2C":
        return g.TopologyConfig.DF_A14_2_CHIP
    if topo_str == "A14_4C":
        return g.TopologyConfig.DF_A14_4_CHIP
    if topo_str == "A14_8C":
        return g.TopologyConfig.DF_A14_8_CHIP
    if topo_str == "A14_16C":
        return g.TopologyConfig.DF_A14_16_CHIP
    elif topo_str == "A14_32C":
        return g.TopologyConfig.DF_A14_32_CHIP
    elif topo_str == "A14_64C":
        return g.TopologyConfig.DF_A14_64_CHIP
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
        default="A14_16C",
        choices=[
            "A14_2C",
            "A14_4C",
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
    parser.add_argument(
        "--instance",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Topology instance on the node\n",
    )
    args = parser.parse_args()

    user_config = get_config_from_topo_str(args.topo_str)

    # [Optional]: Bringup the c2c links before we run the program.
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
                instance_id=args.instance,
            )
        except Exception as e:  # pylint: disable=broad-except
            print_utils.infoc("Aborting, " + str(e))
            sys.exit(1)

        print_utils.infoc("Bringup done")

    # Instantiate a program package to store multi-chip (C2C)
    # or single-chip programs.
    pkg_name = "c2c_multi_hop"
    pkg_dir = config.get_tmp_dir(pkg_name)
    print_utils.infoc(f"Creating a program package '{pkg_name}' at '{pkg_dir}' ...")
    pgm_pkg = g.ProgramPackage(name=pkg_name, output_dir=pkg_dir)

    # Build your multi-chip C2C program.
    prog_name = "c2c_multi_hop"
    input_tensor, result_tensors = build_program(
        user_config, pgm_pkg, prog_name, args.speed
    )

    # Assemble all programs in the multi-device package.
    print_utils.infoc(f"Assembling multi-device package '{pkg_name}' ...")
    pgm_pkg.assemble()

    # Create a multi-tsp runner
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
        print_utils.err("Aborting, " + str(e))
        sys.exit(1)

    # Pass inputs to the runner and execute the program on HW.
    # inputs, oracles = create_data(input_tensor, result_tensors, 0)

    print_utils.infoc(f"Executing C2C program '{prog_name}' ...")

    try:
        for i in range(args.iter):
            inputs, oracles = create_data(input_tensor, result_tensors, (i % 16))

            print_utils.infoc(f"Testing for iteration: {i}")
            results = runner(**inputs)

            # Validation: Compare against oracle.
            print_utils.infoc("Validating results ...")
            check_results(results, oracles)
    except KeyboardInterrupt:
        print_utils.infoc("Program Interrupted... Terminating the program")
    finally:
        print_utils.infoc("Test run completed")
    config.rm_tmp_dir(pkg_name)
