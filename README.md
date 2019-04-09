# Quantum Simulator
1-Dimensional Simulation of the Schrodinger Equation.

![Screenshot of main window.](https://raw.githubusercontent.com/wgxli/quantum-simulator/master/screenshots/main.png)

## Features
* Stable simulation of unitary evolution.
* Stable time acceleration.
* View evolution of wavefunction in different bases.
* Measure observables and simulate the collapse of the wavefunction.

## Usage
Simply run `main.py`.

* The real and imaginary parts of the wave function are plotted in green and blue, respectively.
* The magnitude of the wavefunction is plotted in white.
* The potential is only visible in the default (position) basis, and is plotted in yellow.

## Requirements
Requires the following libraries on Python 3:
* PyQt5
* Numpy
* Scipy
* pyqtgraph

## Limitations
* Does not yet support time-varying potentials.
* Currently supports only one-dimensional wavefunctions.
* Components with very high energy are discarded to improve numerical stability.

## Troubleshooting
* If the simulation runs slowly, try decreasing `basis_size` in `main.py` to 256.
  If the problem persists, decrease `discretization_size` to 512.
  Note, however, that this will decrease the simulation accuracy.
