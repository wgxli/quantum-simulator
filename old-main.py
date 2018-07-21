from functools import lru_cache

import numpy as np
from scipy.stats import norm
from scipy.linalg import expm as matrix_exponential

import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout)

from gui import Slider, LogarithmicSlider, Measurer


# Graphics setup
app = QtGui.QApplication([])


# Simulation Options
bounds = (-10, 10) # Simulation bounds
N = 256 # Discretization fineness

time_multiplier = LogarithmicSlider(1e-4, 1e4, default=0.1, label_format='Simulation Speed: {:.3g}') # Time speed-up factor

h_bar = 1
m = 1

n_components = N # Number of frequency components to use

# Display options
frame_rate = 30
potential_scale = 8e-4 # Scale with which to plot potential

x = np.linspace(*bounds, N)

# Initial Wavefunction
psi = norm.pdf(x, loc=3, scale=0.2).astype(complex)
psi *= np.exp(-3j * x)

# Potential
constant_potential = True # Faster algorithm for constant potential
V=10*x*x
V[x > 9.5] = 1000
V[x < -9.5] = 1000
#V[(x > -0.1) & (x < 0.1)] = 500 #50
#V = 500 * np.cos(np.pi * x)



simulation_size = bounds[1] - bounds[0]

component_frequencies = np.fft.fftfreq(len(psi), simulation_size / N)
frequency_magnitudes = np.abs(component_frequencies)
frequency_basis = np.argsort(frequency_magnitudes)[:n_components] # Frequency components to use for simulation

print('Using {} of {} frequency components.'.format(
    len(frequency_basis),
    len(component_frequencies)))

derivative_multiplier = 2j * np.pi * component_frequencies[frequency_basis]

# Spectral representation conversion utilities
def to_spectral(psi):
    return np.fft.fft(psi)[frequency_basis]

def from_spectral(psi_hat):
    spectrum = np.zeros(N, dtype=complex)
    spectrum[frequency_basis] = psi_hat
    return np.fft.ifft(spectrum)

def derivative(psi_hat, n=1):
    return psi_hat * derivative_multiplier**n

# Compute time derivative of function, expressed in the reduced Fourier basis
def time_derivative(psi_hat, potential=V):
    kinetic_component = 1j * h_bar * derivative(psi_hat, 2) / (2 * m)
    potential_component = -1j * potential * from_spectral(psi_hat) / h_bar
    return kinetic_component + to_spectral(potential_component)


# Get operator matrix from given function (operator is a function which accepts a reduced Fourier basis)
def to_matrix(operator):
    print('Generating matrix for operator...')
    return np.array([operator(row) for row in np.eye(n_components)]).T

# Work in spectral space for speedup
psi_hat = np.zeros(n_components, dtype=complex)

if constant_potential:
    print('Pre-computing energy operator...')
    energy_operator = to_matrix(time_derivative)

    # Compute eigenbasis for energy operator
    print('Computing eigenbasis...')
    eigenvalues, eigenvectors = np.linalg.eig(energy_operator)


    # Sort eigenvalues/eigenvectors by energy
    energies = -eigenvalues.imag # Energies should be positive, as energy operator is physical
    order = np.argsort(energies)

    energies[:] = energies[order]
    eigenvalues = -1j * energies # Eigenvalues should be pure imaginary (energy operator is unitary, operator is physical)
    eigenvectors[:, :] = eigenvectors[:, order]

    assert np.abs(energy_operator @ eigenvectors - eigenvalues * eigenvectors).max() < 1e-3

    # Basis conversion matrix
    projector = eigenvectors.conj().T

    # Compute spectral representation in this basis
    print('Initializing wavefunction...')
    psi_hat = projector @ to_spectral(psi)
    psi_hat *= N / (simulation_size * np.linalg.norm(psi_hat))

    # Utilities and visualization
    stationary_states = np.array([from_spectral(eigenvector) for eigenvector in eigenvectors.T])
    total_power = np.square(np.linalg.norm(psi_hat))
    for i in np.argsort(np.abs(psi_hat))[:-10:-1]:
        print('{:5.2f}% | {:7.3f} Hz'.format(
            100*np.square(np.abs(psi_hat[i]))/total_power,
            energies[i]/(2 * np.pi)
            ))
else:
    psi_hat = to_spectral(psi)
    psi_hat *= N / (simulation_size * np.linalg.norm(psi_hat))



# Evolve system using spectral method
def evolve(t, max_dt=0.001):
    global psi_hat

    if constant_potential:
        psi_hat *= np.exp(eigenvalues*t)
        psi[:] = from_spectral(eigenvectors @ psi_hat)
    else:
        dt = min(max_dt, t)
        iterations = round(t / dt)
        for i in range(iterations):
            # Normalization
            psi_hat *= N / (simulation_size * np.linalg.norm(psi_hat))

            # Runge-Kutta 4th Order Integrator
            k_1 = dt * energy(psi_hat)
            k_2 = dt * energy(psi_hat + k_1 / 2)
            k_3 = dt * energy(psi_hat + k_2 / 2)
            k_4 = dt * energy(psi_hat + k_3)

            step = (k_1 + 2*k_2 + 2*k_3 + k_4) / 6
            psi_hat += step
        psi[:] = from_spectral(psi_hat)


# Observables
if constant_potential:
    position_observable = lambda psi_hat: projector @ to_spectral(x * from_spectral(eigenvectors @ psi_hat))
    momentum_observable = lambda psi_hat: h_bar/1j * (projector @ derivative(eigenvectors @ psi_hat))
    def energy_observable(psi_hat):
        fourier_state = eigenvectors @ psi_hat
        kinetic_energy = -h_bar**2 * derivative(fourier_state, 2) / (2*m)
        potential_energy = to_spectral(V * from_spectral(fourier_state))
        return projector @ (kinetic_energy + potential_energy)
else:
    raise ValueError

# Main Window
control_win = QWidget()
control_win.setWindowTitle('Quantum Simulation')

layout = QVBoxLayout(control_win)

# Animation Window
win = pg.GraphicsWindow()
layout.addWidget(win)

# Sliders
layout.addWidget(time_multiplier)

# Measurement
measurement_panel = QHBoxLayout()
measurement_panel.addWidget(Measurer('Position', to_matrix(position_observable), psi_hat))
measurement_panel.addWidget(Measurer('Momentum', to_matrix(momentum_observable), psi_hat))
measurement_panel.addWidget(Measurer('Energy', to_matrix(energy_observable), psi_hat))

measurement_widget = QWidget()
measurement_widget.setLayout(measurement_panel)
layout.addWidget(measurement_widget)

# Animation
pg.setConfigOptions(antialias=True)

plot = win.addPlot()
plot.setLabel('left', 'Amplitude')
plot.setLabel('bottom', 'Position', units='m')
plot.setXRange(*bounds)
plot.setYRange(-0.2, 0.4)

plot_functions = [
        lambda: psi.real,
        lambda: psi.imag,
        lambda: np.abs(psi),
        lambda: potential_scale * V]
colors=['g', 'b', 'w', 'y']
plots = [plot.plot(pen=color) for color in colors]

def update_animation():
    evolve(float(time_multiplier) / frame_rate)
    for plot, plot_function in zip(plots, plot_functions):
        plot.setData(x, plot_function())
    return plots

timer = QtCore.QTimer()
timer.timeout.connect(update_animation)
timer.start(1000 // frame_rate)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            control_win.show()
            QtGui.QApplication.instance().exec_()
