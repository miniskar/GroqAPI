# About Examples

The following examples are python files. Below includes a short description of
the example as well as the command(s) to run the program. These are examples that
use Groq API at the Tensor API level.

1. `add_tensors.py` :: A standalone python module that adds two tensors
together using Groq API.

    > python add_tensors.py

2. `matmul_i8.py` :: A standalone matrix multiplication python module in
Groq API using API matmul function with INT8 input tensors.

    > python matmul_i8.py

3. `mem_constraints_example.py` :: An example on how/when to use the memory
constraints that dictate a relationship between the memory allocated to memory tensors.

    > python mem_constraints_example.py

4. `multi_program_example.py` :: This script defines 2 programs to load onto a GroqChip
together. The programs use different memory allocations for their input and output tensors.
Programs can share memory allocations. In this example, Program 2 uses Program 1's result as
a parameter to multiply.

    > python multi_program_example.py

5. `buffered_scopes_example.py` :: This is an example on how to use buffered scopes
to automatically handle RAW dependencies. The example has three stages: a transpose, a matmul
and an addition. Each stage writes its output to the memory so that it can be consumed by the
subsequent stage. By marking each stage as buffered, we don't need to specify the time for
the second and third stages.

    > python buffered_scopes_example.py

