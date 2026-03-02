"""Test custom legend location (x, y) tuple."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .helpers import assert_equality

mpl.use("Agg")


def plot() -> Figure:
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="Sine Wave", color="blue")

    # 3. Add legend using a 2-tuple for coordinates
    # loc=(0.5, 0.5) places the 'lower left' corner of the legend
    # at the center of the axes (50% width, 50% height).
    plt.legend(loc=(0.5, 0.5))

    plt.title("Plot with Coordinate-Based Legend")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.grid(visible=True)

    return fig


def test() -> None:
    assert_equality(plot, __file__[:-3] + "_reference.tex")
