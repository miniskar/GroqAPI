# Groq RealScale™ Chip-to-Chip (C2C) Examples

Groq API supports multi-device programs using RealScale (C2C) as the communication
between devices. The following examples demonstrate this capability.

There is also a Multi-Chip Jupyter Notebook example included in the notebook folder.

Before running the examples, it is important to ensure that the
cards are named such that they match the expected topology. See GroqWare Getting
Started Guide for more information on naming the Groq devices. As well,
if running the C2C example for the first time, use the `--bringup` option
to set up the C2C config.

**Note:** For A1.4b-based GroqNode servers, default board speed is set as 25 (implicitly
the Ethernet Standard of 25.78125GHz) and cannot be overridden by the user. For
A1.1-based GroqNode servers, board speed should be explicitly set to 30GHz by the user
with the `--speed=30` flag.

1. `c2c_bcast_example.py` :: Example design that builds, compiles and runs C2C
broadcast collective op for fully connected n-way topology, where 2<= n <= 8.

    (First Time)
    > python c2c_bcast_example.py --topo_str=A14_8C --bringup

    (Subsequent)
    > python c2c_bcast_example.py --topo_str=A14_8C

2. `c2c_gather_example.py` :: Example design that builds, compiles and runs C2C
gather collective op for fully connected n-way topology, where 2<= n <= 8.

    (First Time)
    > python c2c_gather_example.py --topo_str=A14_8C --bringup

    (Subsequent)
    > python c2c_gather_example.py --topo_str=A14_8C

3. `c2c_reduce_example.py` :: Example design that builds, compiles and runs C2C
reduce collective op for fully connected n-way topology, where 2<= n <= 8.

    (First Time)
    > python c2c_reduce_example.py --topo_str=A14_8C --bringup

    (Subsequent)
    > python c2c_reduce_example.py --topo_str=A14_8C

4. `c2c_scatter_example.py` :: Example design that builds, compiles and runs C2C
scatter collective op for fully connected n-way topology, where 2<= n <= 8.

    (First Time)
    > python c2c_scatter_example.py --topo_str=A14_8C --bringup

    (Subsequent)
    > python c2c_scatter_example.py --topo_str=A14_8C

5. `c2c_fault_monitoring_example.py` :: Example design that builds, compiles and runs C2C
scatter collective op for supported topologies. With an option to enable and monitor faults
on the C2C links on each device associated in the topology.

    > python c2c_fault_monitoring_example.py --topo_str=A14_8C --poll_interval=0.5 --file="/tmp/c2c_faults.log"

or
    > python c2c_fault_monitoring_example.py --topo_str=A14_8C --poll_interval=0.5 --file="/tmp/c2c_faults.log"

6. `c2c_all_reduce_example.py` :: Example design that builds, compiles and runs C2C
all-reduce collective operation for fully connected n-way topology, where 2<= n <= 8.

    (First Time)
    > python c2c_all_reduce_example.py --topo_str=A14_8C --bringup

    (Subsequent)
    > python c2c_all_reduce_example.py --topo_str=A14_8C

7. `c2c_multi_node_bcast_example.py` :: Example design that builds, compile and runs C2C
broadcast collective operation for multi-node topology, 1 node (8-chip topology), 2 node (16-chip topology)
4-node (32-chip topology) and 8-node (64-chip topology)
    (First Time)
    > /nix/store/s6mdw66684fwr5lf3kkbzhqp2yfl7z91-openmpi-4.1.2/bin/mpirun --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include eno1 -np <number_of_nodes> --hostfile hostfile.txt python c2c_multi_node_bcast_example.py --topo_str=A14_16C --bringup

    (Subsequent)
    > /nix/store/s6mdw66684fwr5lf3kkbzhqp2yfl7z91-openmpi-4.1.2/bin/mpirun --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include eno1 -np <number_of_nodes> --hostfile hostfile.txt python c2c_multi_node_bcast_example.py --topo_str=A14_16C

8. You may override the default topology and run the above 6 examples for different topologies of your choice.

   The list of available topologies that can be mapped to are as follows (note that 'C' stands for 'Card'):
   *A11_2C
   *A11_4C
   *A14_2C
   *A14_4C
   *A14_8C
   *A14_16C
   *A14_32C

When n == 2 or n == 4, the program can be deployed onto any of the instances of its
occurrence on the GroqNode (as described in the above examples) OR can be deployed onto
a specific instance within the GroqNode.

For example, if you wanted to bringup and deploy a DragonFly 2-Card topology-based program
on instance 0, the following commands would be used:

(First Time)
> python c2c_bcast_example.py --topo_str=A14_2C --bringup --instance=0

(Subsequent)
> python c2c_bcast_example.py --topo_str=A14_2C --instance=0

And if you wanted to bringup and deploy an A1.1 fully connected topology-based program,
the following commands would be used (note how `--speed` must be set to 30GHz for A1.1):

(First Time)
> python c2c_bcast_example.py --bringup --topo_str=A11_4C --speed=30

(Subsequent)
> python c2c_bcast_example.py --topo_str=A11_4C --speed=30

[2-Way DF has 4 instances within a fully connected DragonFly topology]
[4-Way DF has 2 instances within a fully connected DragonFly topology]
[8-Way DF has 1 instances within a fully connected DragonFly topology]
