# Computational Physics Projects

This repository contains code associated with written papers in computational physics.  
Each project is implemented in C++ with supporting Python scripts for analysis and plotting.

---

## Repository Structure
 
 – Simulation of 2D Ising model  
 – 2D time-dependent Schrödinger equation

---

## 2D Ising Model

We study the thermodynamical properties of the two-dimensional Ising model on a square lattice using Metropolis Monte Carlo sampling and OpenMP parallelisation. We investigate finite-size effects, equilibration behaviour, energy distributions, and estimate the critical temperature \(T_c\) in the thermodynamic limit.

### Objectives

- Implement a **Metropolis Monte Carlo** solver with periodic boundary conditions.  
- Validate numerical results for `L=2` against exact analytical results.  
- Study equilibration and energy/magnetisation distributions.  
- Compute thermodynamic observables near the critical temperature.  
- Perform **finite-size scaling** to estimate \(T_c(L \to \infty)\).  
- Benchmark OpenMP speed-up across independent Monte Carlo walkers.

### Files

- **`ising.cpp`** – Full Monte Carlo solver, OpenMP parallelisation, file output.

### Compilation

```bash
g++ -O3 -std=c++17 ising.cpp -o ising -fopenmp
```

---

## Time-Dependent Schrödinger Equation in Two Dimensions

We solve the two-dimensional time-dependent Schrödinger equation numerically using a Crank–Nicolson discretisation in space and time. The project focuses on wavepacket propagation, norm preservation, and quantum interference phenomena arising from slit potentials. A C++ implementation is used for time evolution, while Python is used for post-processing and visualisation.

### Objectives

- Implement the **Crank–Nicolson method** for the 2D Schrödinger equation with Dirichlet boundary conditions.  
- Construct the required sparse matrices and index-mapping strategy for efficient linear algebra operations.  
- Simulate Gaussian wavepacket propagation in free space and in **single-, double-, and triple-slit potentials**.  
- Verify numerical accuracy through strict monitoring of **probability conservation**.  
- Extract **1D detection profiles** and compare them against **analytical Fresnel diffraction predictions**.  
- Visualise the evolution of the **probability density**, **real part**, and **imaginary part** of the wavefunction.

### Files

- **`solver.cpp`** – Main solver implementing:
  - Grid construction and 2D → 1D index mapping,
  - Sparse CN matrices `A` and `B`,
  - Time evolution using `spsolve()`,
  - Wavepacket initialisation and normalisation,
  - Output of:
    - full wavefunction time series,
    - total probability vs. time.

- **`v_gen.cpp`** – Generates **single-, double-, and triple-slit potentials** on a uniform grid and writes them to `.txt` files.

- **`parameters.txt`** – Controls grid spacing, time step, potential file, wavepacket parameters, and output names.

### Compilation

Compile solver:

```bash
g++ -O2 -std=c++17 solver.cpp -o solver -larmadillo
```

Compile potential generator:

```bash
g++ -O2 -std=c++17 v_gen.cpp -o v_gen
```

### Dependencies

- Armadillo  
- BLAS & LAPACK  

Install Armadillo:

```bash
sudo apt-get install libarmadillo-dev libopenblas-dev liblapack-dev
```

---

## Authors

Lars Campsteijn-Høiby – Computational Physics Projects
