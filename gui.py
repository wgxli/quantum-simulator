import numpy as np

import pyqtgraph as pg

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QLabel,
    QPushButton, QSlider,
    QHBoxLayout, QVBoxLayout)

from observables import PositionObservable


class Slider(QWidget):
    def __init__(
            self,
            minimum, maximum, default=None,
            ticks=1000,
            label_format='{:.3f}',
            label_size=200,
            parent=None):
        super().__init__(parent=parent)

        # Qt Layout
        self.label_format = label_format
        self.label = QLabel(self)
        self.label.setMinimumWidth(label_size)

        self.slider = QSlider(self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(ticks)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.valueChanged.connect(self.update_label)

        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.label)
        horizontal_layout.addWidget(self.slider)

        self.setLayout(horizontal_layout)
        self.resize(self.sizeHint())

        # Internal variables
        self.minimum = minimum
        self.maximum = maximum
        self.ticks = ticks

        self.range = self.maximum - self.minimum
        assert self.range > 0

        if default is not None:
            self.set_value(default)
            self.update_label()

    def set_value(self, value):
        tick = self.ticks * (value - self.minimum) / self.range
        self.slider.setValue(round(tick))

    def fractional_position(self):
        return self.slider.value() / self.ticks

    def value(self):
        return self.minimum + self.fractional_position() * self.range

    def update_label(self):
        self.label.setText(self.label_format.format(self.value()))


class LogarithmicSlider(Slider):
    def __init__(
            self,
            minimum, maximum, *args,
            default=None, **kwargs):
        super().__init__(minimum, maximum, *args, **kwargs)
        assert self.minimum > 0

        self.log_range = np.log(self.maximum) - np.log(self.minimum)

        if default is not None:
            self.set_value(default)

    def set_value(self, value):
        log_position = np.log(value) - np.log(self.minimum)
        fractional_position = log_position / self.log_range
        self.slider.setValue(round(fractional_position * self.ticks))

    def value(self):
        return self.minimum * np.exp(self.fractional_position()*self.log_range)


class Window(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent=parent)
        vertical_layout = QVBoxLayout()
        self.setLayout(vertical_layout)
        self.setWindowTitle(title)

    def add_widget(self, widget):
        self.layout().addWidget(widget)


class Plot(pg.GraphicsWindow):
    def __init__(
            self,
            state, basis,
            potential, potential_scale=1,
            parent=None):
        super().__init__(parent=parent)
        self.plot = self.addPlot()
        pg.setConfigOptions(antialias=True)

        self.plot.setLabel('left', 'Amplitude')

        self.initialize_plots()

        self.state = state
        self.update_basis(basis)

        self.potential = potential
        self.potential_scale = potential_scale

    def initialize_plots(self):
        plot_functions = [np.real, np.imag, np.abs]
        colors = ['g', 'b', 'w']
        self.plots = [
            (self.plot.plot(pen=color), plot_function)
            for color, plot_function in zip(colors, plot_functions)]
        self.potential_plot = None

    def update_basis(self, basis):
        self.basis = basis

        self.order = np.argsort(basis.eigenvalues)
        self.x_data = basis.eigenvalues[self.order]

        self.plot.setLabel('bottom', basis.name.title(), units=basis.unit)

        self.plot.setXRange(self.x_data.min(), self.x_data.max())
        self.plot.setYRange(-0.3, 0.3)

        # Only plot potential in position basis
        self.plot.removeItem(self.potential_plot)
        if isinstance(basis, PositionObservable) or basis is None:
            self.potential_plot = self.plot.plot(pen='y')
        else:
            self.potential_plot = None

    def update(self):
        self.setVisible(False)
        psi = self.state.to_basis(self.basis).vector[self.order]
        for plot, plot_function in self.plots:
            plot.setData(self.x_data, plot_function(psi))

        if self.potential_plot is not None:
            y_data = self.potential_scale.value() * self.potential
            self.potential_plot.setData(self.x_data, y_data)

        self.setVisible(True)
        return self.plots


class ObservableWidget(QWidget):
    def __init__(self, observable, state, plot, parent=None):
        super().__init__(parent=parent)

        self.header_label = QLabel()
        self.header_label.setText(observable.name.title())

        self.basis_button = QPushButton()
        self.basis_button.setText('Use as Basis')
        self.basis_button.clicked.connect(self.change_basis)

        self.measure_button = QPushButton()
        self.measure_button.setText('Measure')
        self.measure_button.clicked.connect(self.measure)

        self.result_label = QLabel()
        self.result_label.setText('Result: Not Measured')

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.header_label)
        vertical_layout.addWidget(self.basis_button)
        vertical_layout.addWidget(self.measure_button)
        vertical_layout.addWidget(self.result_label)
        self.setLayout(vertical_layout)

        self.observable = observable
        self.state = state
        self.plot = plot

    def measure(self):
        # Measure observable
        measurement_result = self.state.measure(self.observable)

        # Update result label
        self.result_label.setText('Result: {:.3f} {}'.format(
            measurement_result,
            self.observable.unit))

    def change_basis(self):
        self.plot.update_basis(self.observable)
