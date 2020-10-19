# musr-decoherence
Calculator for predicting the dipolar coupling between muons and nuclei in MuSR experiments, and to analyse the 'decoherence' caused by multiple nuclei's interactions with the muon.

## Installation

Install the package with pip -- clone this repository, cd to this folder and run:
```bash
pip install ./
```
for the CPU version, or
```bash
pip install ./[gpu]
```
to enable GPU-parallelisation (do this AFTER you have installed and configured the CUDA toolkit)

In the examples folder, there are CaF2 polarisation and fitting scripts, and a NaF quadrupole polarisation script.
