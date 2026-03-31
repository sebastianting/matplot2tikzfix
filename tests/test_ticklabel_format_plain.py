"""Test plain scalar tick formatting exports explicit labels."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .helpers import assert_equality

mpl.use("Agg")


def plot() -> Figure:
    years = np.arange(1920, 2001, 20)
    control_pop_bil = np.array([8000000000, 10000000000, 13000000000, 18000000000, 24000000000])
    treatment_pop_bil = np.array([7000000000, 9500000000, 14000000000, 21000000000, 29000000000])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(years, control_pop_bil, marker="o", label="control")
    ax.plot(years, treatment_pop_bil, marker="o", label="treatment")
    ax.set_xticks(years)
    ax.ticklabel_format(useOffset=False, style="plain")

    return fig


def test() -> None:
    assert_equality(plot, __file__[:-3] + "_reference.tex")
