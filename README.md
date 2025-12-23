# physics_697_fall_2025_Max_Aradh_project
Project repository for PHYSIC 697 at UMass Boston for Fall 2025

# 1+1 QCD and QED Hamiltonian Simulations

This repository contains Julia code for simulating 1+1 dimensional Lattice Gauge Theories (QCD and QED) using Matrix Product States (MPS) and the DMRG algorithm via the [ITensors.jl](https://github.com/ITensor/ITensors.jl) library.

## Files Overview

This project consists of three main simulation files:

### 1. 'Hamiltonian_auto.jl' (QCD Only)

* **Physics:** Simulates SU(N) gauge theory with staggered fermions.
* **Execution:** Runs serially or with multi-threaded block-sparse operations.

### 2. 'Hamiltonian_auto_QED.jl' (QCD + QED)

* **Physics:** Adds U(1) electromagnetic coupling to the SU(N) model. Includes the full electric field term .
  
### 3. 'Hamiltonian_auto_QED_parallel.jl' (Distributed QCD + QED)

* **Physics:** Identical to the QED file but architected for **Distributed Computing**.
* **Architecture:** Uses 'pmap' (Parallel Map) to distribute phase diagram pixels across separate worker processes.
* **Trade-off:**
* **Pros:** Scales linearly with CPU cores; Garbage Collection pauses are limited to a single process.
* **Cons:** **High Memory Usage.** Every worker constructs and stores its own copy of the Hamiltonian operators (MPOs). Ensure your machine has sufficient RAM (reccomended at least 32GB and a 64GB Swap file).



---

## Plots & Visualization

The scripts generate '.jld2' data files and '.png' plots for the following observables:

* **Phase Diagram (Energy Gap):**
* A heatmap of the gap  between the ground and first excited states.
* Dark regions (gap  0) indicate phase transitions or critical lines.


* **Chiral Condensate :**
* A heatmap showing the order parameter for chiral symmetry breaking.
*  Maps the boundaries between massive (symmetry broken) and massless (symmetric) phases.


* **Total Flux Squared (Charge) :**
* A heatmap of the background electric flux/charge accumulation.
* Useful for identifying screening phases or vacuum realignment.


* **Local Magnetization (Observables):**
* Line plots of spin expectation values  at every lattice site, resolved by Color and Flavor indices.
* visualizes the local charge/matter density distribution.


* **Entanglement Entropy:**
* Von Neumann entropy  calculated across every bond cut.
* Peaks indicate high quantum correlation; scaling can reveal the Central Charge () of the underlying CFT.


* **Spin-Spin Correlations:**
* Heatmap of  correlations
* Reveals magnetic ordering (antiferromagnetic/ferromagnetic) across the lattice.



---

## Environment Setup

To run these simulations efficiently, you must configure Julia's threading and garbage collection to avoid "oversubscription" (where too many threads fight for CPU time).

### 1. Critical: "Ghost Thread" Fix

Julia 1.10+ spawns a Garbage Collection (GC) thread for every compute thread, often causing performance stuttering. You should limit this via an environment variable.

* **Linux / macOS:**
'''bash
export JULIA_NUM_GC_THREADS=1

'''


* **Windows (PowerShell):**
'''powershell
$env:JULIA_NUM_GC_THREADS="1"

'''



### 2. Running 'Hamiltonian_auto.jl' or 'Hamiltonian_auto_QED.jl'

These files use shared-memory threading.

* **Recommended:** Run with 'auto' threads, but force GC threads to 1.
'''bash
julia -t auto --gcthreads=1 Hamiltonian_auto_QED.jl

'''



### 3. Running 'Hamiltonian_auto_QED_parallel.jl'

This file uses distinct processes. You do not need the '-t' flag.

* **Recommended:** Let the script handle worker allocation (it auto-detects cores), or launch manually with '-p'.
'''bash
julia -p 4 Hamiltonian_auto_QED_parallel.jl

'''



**Note on Crashes:**
If the parallel script crashes, worker processes may remain in memory.

* **Cleanup Command (Linux/Mac):** 'pkill -f julia'
* **Cleanup Command (Windows):** 'taskkill /F /IM julia.exe'

## Dependencies

Ensure the following packages are installed in your Julia environment:

'''julia
using Pkg
Pkg.add([
    "ITensors",
    "ITensorMPS",
    "Plots",
    "JLD2",
    "ProgressMeter",
    "Distributed",
    "PlotlyJS"
])

'''
