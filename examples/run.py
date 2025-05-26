"""VQITE Runner Script.

This script performs Variational Quantum Imaginary Time Evolution (VQITE) using Quimb
and MPI parallelization. It provides a command-line interface to configure and run
VQITE simulations with various optimization strategies and parameters.

Command-line Arguments:
    -f, --filename : str
        Filename specifying parameters in the format, e.g., 'N12g0.1'
        (default: 'N12g0.1')
    -i, --init_params : str
        Initial parameters strategy, e.g., 'random', 'zeros' (default: 'random')
    -om, --optimize_m : str
        Optimizer to use when computing matrix M (default: 'greedy')
    -ov, --optimize_v : str
        Optimizer to use when computing vector V (default: 'greedy')
    -s, --simplify_sequence : str
        Simplification sequence to use by quimb (default: 'ADCRS')

The script uses MPI for parallel computation, with rank 0 process handling
initialization and final output, while other ranks participate in the VQITE computation.

Output:
    Creates output files in the 'outputs' directory with naming convention:
    output{filename}n{num_processes}_{init_params}_om{optimize_m}_ov{optimize_v}
    _s{simplify_sequence}.txt
"""

import argparse
import os
import time

from vqite import vqite_quimb

try:
    from mpi4py import MPI

    mpi_available = True
    # Initialize MPI communication
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except ImportError:
    # Fallback: Serial mode
    mpi_available = False
    rank = 0
    size = 1

# Set up command-line argument parser
parser = argparse.ArgumentParser(
    description="Performs VQITE using Quimb and MPI parallelization"
)
parser.add_argument(
    "-f",
    "--filename",
    type=str,
    default="N12g0.1",
    metavar="\b",
    help="filename specifying parameters in the format, e.g., N12g0.1",
)
parser.add_argument(
    "-i",
    "--init_params",
    type=str,
    default="random",
    metavar="\b",
    help="initial parameters",
)
parser.add_argument(
    "-om",
    "--optimize_m",
    type=str,
    default="greedy",
    metavar="\b",
    help="optimizer to use when computing matrix M",
)
parser.add_argument(
    "-ov",
    "--optimize_v",
    type=str,
    default="greedy",
    metavar="\b",
    help="optimizer to use when computing vector V",
)
parser.add_argument(
    "-s",
    "--simplify_sequence",
    type=str,
    default="ADCRS",
    metavar="\b",
    help="simplification sequence to use by quimb.",
)
args = parser.parse_args()

# Extract command line arguments
filename = args.filename
init_params = args.init_params
optimize_m = args.optimize_m
optimize_v = args.optimize_v
simplify_sequence = args.simplify_sequence

# Set up input and output file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
incar_file = os.path.join(script_dir, "incars", f"incar{filename}")
ansatz_file = os.path.join(script_dir, "data_adaptvqite", filename, "ansatz_inp.pkle")

outputs_dir = os.path.join(script_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)
output_file = os.path.join(
    outputs_dir,
    f"output{filename}n{size}_{init_params}_om{optimize_m}_ov{optimize_v}_"
    f"s{simplify_sequence}.txt",
)

if mpi_available:
    with open(output_file, "w") as f:
        print(f"Running in parallel mode with {size} processes", file=f)
else:
    with open(output_file, "w") as f:
        print("MPI not available, running in serial mode", file=f)


def log(message: str) -> None:
    """Log a message to the output file."""
    global output_file
    with open(output_file, "a") as f:
        print(message, file=f)


# Initialize VQITE on rank 0 and broadcast parameters to other ranks
# (this is for the case of random initial parameters such that they are the same on all
# ranks)
if rank == 0:
    if mpi_available:
        start_time = MPI.Wtime()
    else:
        start_time = time.time()
    vqite_quimb_obj = vqite_quimb.QuimbVqite(
        incar_file=incar_file,
        ansatz_file=ansatz_file,
        output_file=output_file,
        init_params=init_params,
    )
    init_params = vqite_quimb_obj.params
    if mpi_available:
        end_time = MPI.Wtime()
    else:
        end_time = time.time()
    log(f"rank={rank}, initialization time: {end_time - start_time}")
else:
    init_params = None

# Broadcast initial parameters from rank 0 to all other ranks
if mpi_available:
    init_params = comm.bcast(init_params, root=0)

# Initialize VQITE on other ranks with broadcasted parameters
if rank != 0:
    vqite_quimb_obj = vqite_quimb.QuimbVqite(
        incar_file=incar_file,
        ansatz_file=ansatz_file,
        output_file=output_file,
        init_params=init_params,
    )

# with open("adaptvqite/adaptvqite/data/N12g0.5/M_V.pkle", 'rb') as inp:
#     data_inp = pickle.load(inp)
#     M_adaptvqite = data_inp[0]
#     V_adaptvqite = data_inp[1]

# start_time = MPI.Wtime()
# vqite_quimb_obj.compute_m(which_nonzero=None,optimize='greedy',simplify_sequence = '',
# backend=None)
# end_time = MPI.Wtime()
# Mdiff = M_adaptvqite - vqite_quimb_obj._m
# if rank==0:
#     print("time: ",end_time-start_time)
#     print(np.where((Mdiff>1e-14) == True))

# start_time = MPI.Wtime()
# vqite_quimb_obj.compute_v(optimize='greedy',simplify_sequence = '',backend=None)
# end_time = MPI.Wtime()
# Vdiff = V_adaptvqite - vqite_quimb_obj._v
# if rank==0:
#     print("time: ",end_time-start_time)
#     print(np.where((Vdiff>1e-14) == True))

# Record initial parameters
if rank == 0:
    log(f"{vqite_quimb_obj.params}")

# Run VQITE
vqite_quimb_obj.vqite(
    optimize_m=optimize_m,
    optimize_v=optimize_v,
    simplify_sequence=simplify_sequence,
    backend=None,
)

# Record final results
log(f"Final energy: {vqite_quimb_obj._e}, {vqite_quimb_obj.params}")
