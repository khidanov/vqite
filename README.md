# vqite
Tensor-network implementation of the Variational Quantum Imaginary-Time Evolution (VQITE) algorithm using Quimb. The code is parallelized using MPI.


## Prerequisites

Before installing the Python packages, ensure you have the following prerequisites:

It's recommended to use a **virtual environment**:

1. Using **venv** (built-in):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Using **conda**

   - Install Anaconda or Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html)

   ```bash
   conda create -n myenv python=3.10
   conda activate myenv
   ```

### MPI Installation (*in progress, check macOS installation*)

This package can optionally use **MPI** for parallel execution.


- **Linux/macOS**: Install OpenMPI or MPICH
  ```bash
  # Ubuntu/Debian
  sudo apt-get install openmpi-bin libopenmpi-dev

  # macOS (using Homebrew)
  brew install open-mpi
  ```
- **Windows**: Install Microsoft MPI (MS-MPI)
  1. Download and install both the MS-MPI SDK and runtime from the [Microsoft website](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
  2. Add the MPI installation directory to your system PATH

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/khidanov/vqite.git
   ```

2. Create and activate a new environment (optional):

3. Install **project dependencies** (required):
   ```bash
   pip install .
   ```

4. Install **optional dependecies**: [IN PROGRESS. ADD MPI INSTALLATION FOR MACOS]
   ```bash
   # For Windows users
   pip install .[cotengra_advanced,mpi]

   # For Zsh users
   pip install ".[cotengra_advanced,kahypar]"
   ```

### Important Notes

- **Windows Users**: The `kahypar` package (used by some tensor network contractions) is not supported on Windows. You may need to:
  - Use Windows Subsystem for Linux (WSL) for full functionality
- **MacOS Users**: Installing `mpi4py` via `pip install .[mpi]` may fail

## Running the Code

1. Activate the Conda environment:
   ```bash
   conda activate myenv
   ```

2. Execution:
   - For execution **with MPI** (single-node):
      ```bash
      mpiexec -n <number_of_processes> python run.py
      ```
   - For execution **with MPI** (multi-node / cluster):
      ```bash
      mpiexec -n <number_of_processes> --hostfile hostfile python run.py
      ```
   - For execution **without MPI**:
      ``` bash
      python run.py
      ```
