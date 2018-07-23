import numpy as np

from scipy.stats import norm

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QHBoxLayout

from base import State
from observables import (
    PositionObservable,
    MomentumObservable,
    EnergyObservable)

from gui import Window, Plot, LogarithmicSlider, ObservableWidget


app = QtGui.QApplication([])

# Graphics options
"""
frame_rate : integer
    Target frame rate (FPS) of simulation.

simulation_speed : Slider
    Slider to control simulation speed.

potential_scale : Slider
    Slider to control vertical scale at which energy potential is plotted.
"""
frame_rate = 30

simulation_speed = LogarithmicSlider(
    default=1,
    minimum=1e-4, maximum=1e4,
    label_format='Simulation Speed: {:.3g}')

potential_scale = LogarithmicSlider(
    default=1e-1,
    minimum=1e-5, maximum=1e1,
    label_format='Potential Scale: {:.2g}')


# Simulation options
"""
bounds : tuple of 2 real numbers
    Position of the physical bounds of the simulation space.

discretization_size : integer
    Number of samples used to discretize simulation space.
    Higher numbers increase visual resolution at expense of speed.
    Must be greater than basis_size.
    Setting equal to basis_size may cause unwanted artifacts
    due to momentum wrap-around.

basis_size : integer
    Number of energy components to use.
    Higher numbers increase simulation accuracy at expense of speed.
    Must be less than discretization_size.
    Setting equal to discretization_size may cause unwanted artifacts
    due to momentum wrap-around.
"""
bounds = (-10, 10)
discretization_size = 1024
basis_size = 512


# Physical parameters
"""
mass : float
    Mass of simulated quantum object.

h_bar : float
    Reduced Planck constant.
    Lower numbers reduce dispersion of wavefunction.
    Artifacts may appear at extremely low values due to wrap-around
    in momentum space.
"""
mass = 1
h_bar = 0.1


# Define space and initial conditions
"""
space : array_like
    One-dimensional array with length discretization_size.
    Represents the simulation space in the position basis.
    Each entry should be set to the value of its corresponding position
    in the simulation.

potential : array_like
    One-dimensional array with length discretization_size.
    Values represent energy potential at each point in space.

psi : array_like
    One-dimensional array with length discretization_size.
    Values represent initial wavefunction at each point in space.
    Will later be normalized automatically.
"""
space = np.linspace(*bounds, discretization_size)

potential = 0.1 * space**2  # Harmonic oscillator potential
potential[space < 0] = -2  # Create well on left side
potential[(space < -9.5) | (space > 9.5)] = 1e5  # Create walls
potential[(space > -0.1) & (space < 0.1)] = 2  # Create central barrier

# r = (space-bounds[0]) / 8 + 0.3
# potential = 1 * (r**-12 - 2 * r**-6) # Atomic potential

psi = norm.pdf(space, loc=3, scale=0.1).astype(complex)  # Gaussian wavepacket

# ----- End of User-Defined Settings ----- #


# Define observables
position = PositionObservable(space)
momentum = MomentumObservable(space, h_bar=h_bar)
energy = EnergyObservable(
        momentum, potential,
        basis_size=basis_size,
        mass=mass)


# Initialize simulation state
simulation_size = bounds[1] - bounds[0]
target_norm = np.sqrt(discretization_size) / simulation_size

state = State(psi * target_norm / np.linalg.norm(psi)).to_basis(energy)


def evolve(time):
    """Evolve the current state forward in time by given duration."""
    state_multiplier = np.exp((-1j/h_bar * time) * energy.eigenvalues)
    state.vector *= state_multiplier


# Graphics
plot = Plot(
    state, basis=position,
    potential=potential, potential_scale=potential_scale)

main_window = Window('Quantum Simulation')
main_window.add_widget(plot)
main_window.add_widget(simulation_speed)
main_window.add_widget(potential_scale)

measurement_panel = QWidget()
measurement_layout = QHBoxLayout(measurement_panel)
measurement_layout.addWidget(ObservableWidget(position, state, plot))
measurement_layout.addWidget(ObservableWidget(momentum, state, plot))
measurement_layout.addWidget(ObservableWidget(energy, state, plot))

main_window.add_widget(measurement_panel)


# Animation
def update_animation():
    evolve(simulation_speed.value() / frame_rate)
    plot.update()


timer = QtCore.QTimer()
timer.timeout.connect(update_animation)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        timer.start(1000 // frame_rate)
        main_window.show()
        QtGui.QApplication.instance().exec_()
