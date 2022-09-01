# README for Groq API Tutorials
This directory contains tutorials for using Groq API. If you're a new
user to Groq API, this is a great place to started!

## Installing Jupyter Lab

To use Jupyter Notebooks, be sure you've installed Jupyter Lab:

`pip install jupyter_client==7.3.4`

Note: If you have issues running Jupyter Lab, the following commands
may be helpful.

`find ~ -name jupyter-lab`

Output:
> /home/user/.local/bin/jupyter-lab

Either add the output of the command to your PATH or launch Jupyter Lab
using the path:

`/home/user/.local/bin/jupyter lab`

If you're remotely accessing your server, don't forget to setup your tunnel, where
server_ip is the IP address of your server.

`ssh -L 8888:localhost:8888 server_ip`

## Tutorials

It is also recommended that you read the [Groq API Programming
Abstraction Guide](https://support.groq.com/#/downloads/groqapi-abstraction)
and the [Groq API Tutorial Guide](https://support.groq.com/#/downloads/groqapi-tutorial)
prior to using these tutorials.

While each tutorial is sufficient on its own, we recommend going through
each tutorial in the following order:

1. Memory Copy - 1_memcopy.ipynb
    - Tensors
    - StreamGroups

2. Adding Tensors - 2_adding_tensors.ipynb
    - Time
    - Components
    - VXM

3. Buffered Scopes - 3_buffered_scopes.ipynb
    - Buffered Resource Scopes

4. Matrix Multiplication - 4_matmul.ipynb
    - MXM

5. Multiple Matrix Multiplications - 5_multi_matmul.ipynb
    - MEM
    - Tensor Layouts in Memory

6. Linear Layer - 6_linear.ipynb (advanced)
    - VXM Chaining
    - Stream Conflicts

7. Fibonacci - 7_fibonacci.ipynb (advanced)
    - Program Contexts
    - Shared Tensors / Resource Requests
    - Input / Output (IOP) Files

8. Multi Chip Design - 8_multichip.ipynb (advanced)
    - Requires more than 1 card to be installed.
    - Program Contexts
    - Groq RealScale™ interconnect between GroqCard accelerators
    - Multi-Program Packages

