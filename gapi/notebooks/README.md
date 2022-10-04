# Jupyter Notebook Examples

The following examples are in the format of a Jupyter Notebook.

## Installing Jupyter Lab

To use Jupyter Notebooks, be sure you've installed Jupyter Lab and
the requirements needed for Groq API examples:

`pip install -r requirements.txt`

You can then launch Jupyter Lab using the following command:

`jupyter lab`

If you're remotely accessing your server, don't forget to setup your tunnel, where
server_ip is the IP address of your server.

`ssh -L 8888:localhost:8888 server_ip`

<br>

<b> Note:</b> If you have issues running Jupyter Lab, the following commands
may be helpful.

`find ~ -name jupyter-lab`

Output:
> /home/user/.local/bin/jupyter-lab

Either add the output of the command to your PATH or launch Jupyter Lab
using the path:

`/home/user/.local/bin/jupyter lab`


