# QuantumBeast

QuantumBeast (formerly Min_ex) is a high-performance C++ framework for simulating the Variational Quantum Eigensolver (VQE). It is built on top of [QuEST](https://github.com/QuEST-Kit/QuEST) and provides a powerful, modular architecture capable of running on both personal laptops (via a sleek graphical interface) and High-Performance Computing (HPC) clusters (via a Command-Line Interface and MPI support).

## Features

- **Dual Modes of Operation:**
  - **GUI Mode (Default):** An interactive, real-time graphical interface built with ImGui and ImPlot to visualize optimizations, configure the ansatz, and monitor energy convergence and shot noise simulations.
  - **CLI Mode:** A headless command-line application designed for maximum performance, built using CLI11, and supporting MPI for distributed environments.
- **Advanced VQE Pipeline:**
  - Support for the UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz and other variational forms.
  - Seamless integration with Python scripts (Qiskit/PySCF) to dynamically generate molecular Hamiltonians and UCCSD structures.
  - Highly parallelized gradient calculation using the Parameter Shift Rule via OpenMP.
  - Configurable stochastic energy estimation with shot noise simulation.
- **State-of-the-Art Optimization:**
  - Integrated with NLopt to support various optimization algorithms, including L-BFGS, SLSQP, COBYLA, and Nelder-Mead.
- **High-Performance Backend:**
  - Built-in multi-threading (OpenMP) for all core calculations.
  - Optional support for MPI distribution.
  - Hardware Acceleration-Ready: Automatic detection and activation for NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs through QuEST.

## Project Structure

- `src/core/`: The core library handling quantum simulations, ansatz definitions, physics models, and energy evaluations.
- `src/apps/CLI/`: Entry points and routines for the Headless CLI application.
- `src/apps/GUI/`: Entry points and rendering code for the interactive GUI application.
- `src/scripts/`: Python backend scripts utilized for complex physics calculations (`generate_hamiltonian.py`, `generate_uccsd.py`).

## Prerequisites

- **CMake** (v3.15 or higher)
- **C++20** compatible compiler (GCC, Clang, or MSVC)
- **Python 3** (with `qiskit` and `pyscf` installed for Hamiltonian and Ansatz generation)
- *(Optional)* **CUDA Toolkit** or **ROCm** for GPU acceleration
- *(Optional)* **MPI** implementation for cluster deployment

## Build Instructions

QuantumBeast uses CMake's `FetchContent` to download and configure its extensive list of C++ dependencies (QuEST, CLI11, spdlog, NLopt, Eigen, json, ImGui, ImPlot, GLFW, etc.) automatically.

### Windows

1. Open PowerShell or the Developer Command Prompt.
2. Create a build directory and navigate into it:

   ```powershell
   mkdir build
   cd build
   ```

3. Generate the build files (by default, it configures for GUI mode):

   ```powershell
   cmake ..
   ```

   *Note: To build without the GUI, append `-DBUILD_GUI=OFF` (e.g., `cmake .. -DBUILD_GUI=OFF`).*
4. Compile the project:

   ```powershell
   cmake --build . --config Release
   ```

5. The executable `quantum_gui.exe` (or `quantum_cli.exe`) will be located in `build\bin\Release\` (or `build\bin\`, depending on the generator). The build process will automatically copy the required Python scripts into the output directory.

### Linux

1. Open a terminal.
2. Create a build directory and navigate into it:

   ```bash
   mkdir build && cd build
   ```

3. Generate the build files:

   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

   *Note: Use `-DUSE_MPI=ON` to enable MPI support, or `-DBUILD_GUI=OFF` to build only the CLI mode.*
4. Compile using all available CPU cores:

   ```bash
   make -j$(nproc)
   ```

5. The compiled executables will be available in the `build/bin/` directory.

## Usage

### GUI Mode (Laptop)

To start the graphical interface, run the compiled executable:

- **Windows:** `.\build\bin\Release\quantum_gui.exe`
- **Linux:** `./build/bin/quantum_gui`

**Inside the GUI:**

1. **Configure Molecules:** Define your molecular structure, basis sets, and spatial metrics.
2. **Select Ansatz:** Choose your desired Ansatz (like UCCSD) and precisely configure the number of qubits and electrons.
3. **Setup Optimizer:** Select the optimization algorithm (e.g., L-BFGS, SLSQP) and set hyperparameters like learning rates, parameter constraints, and the number of shots for noise simulation.
4. **Run Simulation:** Launch the VQE simulation and observe the real-time ImPlot graphs tracking energy convergence and optimization steps.

### CLI Mode (HPC/Cluster)

For continuous deployment and cluster runs, compile the project with `-DBUILD_GUI=OFF` (or `-DBUILD_CLI=ON`):

- Run `./build/bin/quantum_cli --help` to display all available command-line arguments and configuration options.
- The CLI is designed to seamlessly consume configuration files or explicit arguments to start VQE runs and output metrics logically.
- For **MPI runs**, initiate the executable with `mpirun`:

  ```bash
  mpirun -np <number_of_processes> ./build/bin/quantum_cli [options]
  ```
