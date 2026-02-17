"""Test that extra_axis_parameters override default axis parameters."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .helpers import assert_equality

mpl.use("Agg")


def plot() -> Figure:
    """Create a simple line plot."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 2 * np.pi, 50)
    ax.plot(x, np.sin(x))
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    return fig


def test_extra_parameters_override_defaults() -> None:
    """Extra axis params (height, width) should override axis_width/axis_height."""
    assert_equality(
        plot,
        "test_extra_parameters_reference.tex",
        axis_width="6cm",
        axis_height="4cm",
        extra_axis_parameters=["height=5cm", "width=8cm"],
    )
