import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .common import RANDOM_STATE

def setup_environment():
    # pandas config
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", 20)
    pd.set_option("display.max_columns", 20)
    
    # numpy config
    np.random.seed(RANDOM_STATE)

    # matplotlib config
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True
    })
    
    # seaborn config
    sns.set_palette("husl")
    sns.set_context("notebook", font_scale=1.1)
    