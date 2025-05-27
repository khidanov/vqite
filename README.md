# vqite
Tensor-network implementation of the Variational Quantum Imaginary-Time Evolution (VQITE) algorithm using Quimb. The code is parallelized using MPI.


## Prerequisites

Before installing the Python packages, ensure you have the following prerequisites:

It's recommended to use a **virtual environment**:

1. Using **venv** (built-in):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  
   
   # On Windows: .venv\Scripts\activate
   ```
2. Using **conda**

   - Install Anaconda or Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html)

   ```bash
   conda create -n myenv python=3.10
   conda activate myenv
   ```

### MPI (mpi4py) Installation

This package can optionally use **MPI (mpi4py)** for parallel execution. See official **mpi4py** [installation page](https://mpi4py.readthedocs.io/en/4.0.3/install.html).


- **Linux/macOS**: It is recommended to use **conda-forge** with the desired MPI implementation, for example MPICH:
  ```bash
  conda install -c conda-forge mpi4py mpich
  ```
  Alternatively, one can try installing via **pip**, but installation problems can be encountered in this case:
  
  ```bash
  python -m pip install mpi4py
  ```

- **Windows**: Install Microsoft MPI (MS-MPI)
  1. Download and install both the MS-MPI SDK and runtime from the [Microsoft website](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
  2. Add the MPI installation directory to your system PATH

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/khidanov/vqite.git
   ```

2. Create and activate a new environment (optional)

3. Install **project dependencies** (required):
   ```bash
   pip install .
   ```

4. Install **optional dependecies**:
   ```bash
   # For Windows users 
   pip install .[cotengra_advanced,mpi]

   # For Linux/macOS users
   pip install .[cotengra_advanced,kahypar]

   # For Zsh shell users
   pip install ".[cotengra_advanced,kahypar]"
   ```

### Important Notes

- **Windows Users**: The `kahypar` package (which is an optional dependency for `cotengra` used to perform optimized tensor network contractions) is not supported on Windows
- **MacOS Users**: Installing `mpi4py` via `pip install .[mpi]` may fail

## Running the Code

1. Activate environment:
   ```bash
   conda activate myenv
   ```

2. Execution:
   - For execution **with MPI** (single-node):
      ```bash
      mpiexec -n <number_of_processes> python run.py
      ```
   - For execution **without MPI**:
      ``` bash
      python run.py
      ```
