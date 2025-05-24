# avqite-quimb
Tensor-network implementation of the Variational Quantum Imaginary-Time Evolution (VQITE) algorithm using Quimb. The code is parallelized using MPI.


## Prerequisites

Before installing the Python packages, ensure you have the following prerequisites:

### Conda Installation

1. Install Anaconda or Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html)
2. Verify the installation:
   ```bash
   conda --version
   ```

### MPI Installation

Before installing Python packages, make sure MPI is installed on your system.

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
   git clone https://github.com/khidanov/vqite-quimb.git
   ```

2. Create and activate a new Conda environment:
   ```bash
   # Create a new environment with Python 3.12
   conda create -n vqite python=3.12
   
   # Activate the environment
   conda activate vqite
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
### Important Notes

- **Always activate the Conda environment before running the code**:
  ```bash
  conda activate vqite
  ```

- **Windows Users**: The `kahypar` package (used by some tensor network contractions) is not supported on Windows. You may need to:
  - Use Windows Subsystem for Linux (WSL) for full functionality

## Running the Code

1. Activate the Conda environment:
   ```bash
   conda activate vqite
   ```

2. For single-node execution:
   ```bash
   mpirun -n <number_of_processes> python run.py
   ```

3. For multi-node execution (cluster):
   ```bash
   mpirun -n <number_of_processes> --hostfile hostfile python run.py
   ```