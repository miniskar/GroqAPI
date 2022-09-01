"""Build, compile and run C2C broadcast collective op for fully connected A1.1 4-way topology."""

import argparse
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


def build_program(user_config, pgm_pkg, prog_name, speed):
    # Setup multi-chip topology and create a new program context.
    topo = g.configure_topology(config=user_config, speed=speed)
    print_utils.infoc(
        f"{mpi_name}: Building C2C program '{prog_name}' with '{topo.name}' topology ..."
    )

    # create init-sync-notify program using create_program_context
    pgm_pkg.create_program_context(prog_name, topo)

    return


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
            f"{mpi_name}: Bringup C2C topology {mpi_name} {mpi_size}, {mpi_rank}..."
        )
        try:
            tsp.bringup_topology(
                user_config=user_config, speed=args.speed, instance_id=args.instance
            )
        except Exception as e:  # pylint: disable=broad-except
            print_utils.err(f"{mpi_name}: Aborting, " + str(e))
            sys.exit(1)

        print_utils.infoc(f"{mpi_name}: Bringup done")

    # Instantiate a program package to store multi-chip (C2C)
    # or single-chip programs.
    pkg_name = "c2c_sync_notify"
    pkg_dir = config.get_tmp_dir(pkg_name)
    print_utils.infoc(
        f"{mpi_name}: Creating a program package '{pkg_name}' at '{pkg_dir}' ..."
    )
    pgm_pkg = g.ProgramPackage(name=pkg_name, output_dir=pkg_dir)

    # Build multi-chip C2C program.
    prog_name = "c2c_sync_notify"
    build_program(user_config, pgm_pkg, prog_name, args.speed)

    # Assemble all programs in the multi-device package.
    print_utils.infoc(f"{mpi_name}: Assembling multi-device package '{pkg_name}' ...")
    pgm_pkg.assemble()

    # Create a multi-tsp runner
    # Make sure to pass the program name to be executed.
    print_utils.infoc(f"{mpi_name}: Creating multi-tsp runner ...")
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
        print_utils.err(f"{mpi_name}: Aborting, " + str(e))
        sys.exit(1)

    print_utils.infoc(f"{mpi_name}: Executing C2C program '{prog_name}' ...")
    try:
        for i in range(args.iter):
            print_utils.infoc(f"{mpi_name}: Testing for iteration: {i}")
            results = runner()
            # Validation:
            print_utils.infoc(f"{mpi_name}: Sync-Notify completed, iteration {i} ...")
    except KeyboardInterrupt:
        print_utils.infoc(
            f"{mpi_name}: Program Interrupted... Terminating the program"
        )
    finally:
        print_utils.infoc(f"{mpi_name}: Test run completed")
    config.rm_tmp_dir(pkg_name)
