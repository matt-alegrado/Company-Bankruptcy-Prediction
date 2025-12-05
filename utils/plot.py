import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .common import OUTPUT_DIR

# generic functions
def plot_hist(df: pd.DataFrame,
              col: str,
              df_name: str | None = None,
              title: str | None = None, 
              xlabel: str | None = None, 
              ylabel: str | None = "Count",
              figsize: tuple | None = (8,5), 
              dpi: int | None = 200, 
              save: bool | None = True,
              show: bool | None = True,
              show_pct: bool | None = False):
    """
    Plot a histogram / count plot of a categorical or discrete column.
    
    Args:
        df          : input dataframe
        col         : column name to plot
        df_name     : output file prefix
        title       : plot title
        xlabel      : x-axis label
        ylabel      : y-axis label
        output_path : path to save the figure, if None, figure is not saved
        figsize     : figure size
        dpi         : figure dpi
        save        : save the figure or not
        show        : display the figure or not
        show_pct    : display percentage or not
    """
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=col, data=df, color='#4C72B0')
    plt.xlabel(xlabel or col)
    plt.ylabel(ylabel)
    plt.title(title or f"Distribution of {df_name}_{col}" if df_name else f"Distribution of {col}")

    if show_pct:
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            pct = height / total * 100
            ax.annotate(f'{pct:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10)

    if save:      
        fname = f"{df_name}_{col}.png" if df_name else f"{col}.png"
        plt.savefig(Path(OUTPUT_DIR)  / fname, dpi=dpi)

    if show:
        plt.show()
    
    plt.close()
    
def plot_bin_rate(
        df: pd.DataFrame,
        col: str,
        target_col: str = "bankrupt",
        bins: list | int = 5,
        df_name: str | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = "Bankruptcy Rate",
        figsize: tuple = (8,5),
        dpi: int = 200,
        save: bool = True,
        show: bool = True
    ):
    """
    Plot the bankruptcy rate per bin of a continuous variable.

    Args:
        df          : input dataframe
        col         : continuous variable to bin
        target_col  : binary target column
        bins        : int number of bins or list of bin edges
        df_name     : output file prefix
        title       : plot title
        xlabel      : x-axis label
        ylabel      : y-axis label
        figsize     : figure size
        dpi         : figure dpi
        save        : save the figure or not
        show        : display the figure or not
    """
    df = df.copy()
    
    # Create bins
    if isinstance(bins, int):
        df['bin'] = pd.cut(df[col], bins=bins)
    else:
        df['bin'] = pd.cut(df[col], bins=bins)
    
    # Calculate bankruptcy rate per bin
    rate = df.groupby('bin', observed=True)[target_col].mean().reset_index()
    
    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='bin', y=target_col, data=rate, color='#DD8452')
    plt.xticks(rotation=45)
    plt.xlabel(xlabel or col)
    plt.ylabel(ylabel)
    plt.title(title or f"Bankruptcy Rate by {col}" if df_name is None else f"{df_name}_{col}")
    
    # Annotate percentage
    for p, r in zip(ax.patches, rate[target_col]):
        ax.annotate(f"{r*100:.1f}%", 
                    (p.get_x() + p.get_width() / 2., r),
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save:
        fname = f"{df_name}_{col}.png" if df_name else f"{col}.png"
        plt.savefig(Path(OUTPUT_DIR) / fname, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_corr_with_target(
        df: pd.DataFrame,
        target_col: str = "bankrupt",
        figsize: tuple = (8,5),
        dpi: int = 100,
        save: bool = True,
        show: bool = True,
        df_name: str | None = None,
        title: str | None = None
    ):
    """
    Plot all features' correlation with a binary target, sorted by correlation.

    Args:
        df          : input dataframe
        target_col  : target column name
        figsize     : figure size
        dpi         : figure dpi
        save        : save the figure or not
        show        : display the figure or not
        df_name     : file name prefix
        title       : plot title
    """
    corr = df.corr()[target_col].drop(target_col)
    top_corr = corr.sort_values()  # sort from negative to positive

    plt.figure(figsize=figsize, dpi=dpi)
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_corr.values]

    # Use hue=top_corr.index to avoid warning, legend=False hides legend
    sns.barplot(
        x=top_corr.values, 
        y=top_corr.index, 
        palette=colors, 
        hue=top_corr.index, 
        dodge=False, 
        legend=False
    )

    plt.axvline(0, color="black", lw=0.8, ls="--")
    plt.title(title or f"Correlation of Features with {target_col}")
    plt.xlabel("Pearson correlation")
    plt.ylabel("Feature")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save:
        fname = f"{df_name}_{target_col}_corr.png" if df_name else f"{target_col}_corr.png"
        plt.savefig(Path(OUTPUT_DIR)/fname, dpi=dpi)
    if show:
        plt.show()
    plt.close()











def plot_bankrupt_features(
        df: pd.DataFrame,
        target_col: str = "bankrupt",
        top_k: int = 3,
        figsize: tuple = (12, 5),
        dpi: int = 200,
        save: bool = True,
        show: bool = True,
        df_name: str | None = None
    ):
    """
    Compare distributions between bankrupt vs non-bankrupt companies.
    Plots features according to mean difference.

    Args:
        df              : input dataframe
        target_col      : binary target column
        top_k           : number of features to plot
        figsize         : figure size
        dpi             : dpi for saving
        save            : save figures or not
        show            : display figures
        df_name         : prefix for saved file names
    """

    df_b = df[df[target_col] == 1]
    df_a = df[df[target_col] == 0]

    # select numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != target_col]

    stats = []
    for col in num_cols:
        mean_b = df_b[col].mean()
        mean_a = df_a[col].mean()
        diff = mean_b - mean_a
        stats.append((col, mean_b, mean_a, diff))

    stats_df = pd.DataFrame(stats, columns=["feature", "mean_bankrupt", "mean_alive", "mean_diff"])
    stats_df["abs_diff"] = stats_df["mean_diff"].abs()

    # top-k by absolute difference
    selected = stats_df.sort_values("abs_diff", ascending=False).head(top_k)

    alive_color = "#4C72B0"
    bank_color = "#DD8452"

    for _, row in selected.iterrows():
        col = row["feature"]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # left: histogram
        axes[0].hist(df_a[col].dropna(), bins=30, alpha=0.65, color=alive_color, label="Alive")
        axes[0].hist(df_b[col].dropna(), bins=30, alpha=0.65, color=bank_color, label="Bankrupt")
        axes[0].set_title(f"Distribution of {col}")
        axes[0].legend()

        # right: violin plot
        sns.violinplot(
            data=df[[col, target_col]],
            x=target_col, y=col,
            hue=target_col,
            palette={0: alive_color, 1: bank_color},
            ax=axes[1],
            inner="quartile",
            legend=False
        )
        axes[1].set_title(f"Violin plot of {col}")
        axes[1].set_yscale('symlog', linthresh=0.1)

        plt.tight_layout()

        if save:
            fname = f"{df_name}_{col}.png" if df_name else f"{col}.png"
            plt.savefig(Path(OUTPUT_DIR) / fname, dpi=dpi)

        if show:
            plt.show()

        plt.close()


def plot_trend(df: pd.DataFrame,
               year_col: str,
               count_col: str = None,
               df_name: str | None = None,
               title: str | None = None,
               xlabel: str | None = "Year",
               ylabel: str | None = "Count",
               figsize: tuple | None = (10,6),
               dpi: int | None = 200,
               save: bool | None = True,
               show: bool | None = True,
               show_bar : bool | None = False,
               show_avg : bool | None = False):
    """
    Plot a trend of counts over time (e.g., number of bankrupt companies per year).
    
    Args:
        df          : input dataframe
        year_col    : column name representing years
        count_col   : optional column to count (if None, counts rows per year)
        df_name     : output file prefix
        title       : plot title
        xlabel      : x-axis label
        ylabel      : y-axis label
        figsize     : figure size
        dpi         : figure dpi
        save        : save figure or not
        show        : display figure or not
        show_bar    : display background bar chart
        show_avg    : display avg line
    """
    # Aggregate counts per year
    if count_col:
        trend = df.groupby(year_col)[count_col].count().reset_index(name="count")
    else:
        trend = df.groupby(year_col).size().reset_index(name="count")

    trend[year_col] = trend[year_col].astype(str)

    # Plot
    plt.figure(figsize=figsize)
    
    # bar chart
    if show_bar:
        plt.bar(trend[year_col], trend["count"], alpha=0.3, color='#b9b9b9')
    
    # trend line
    sns.lineplot(data=trend, x=year_col, y="count", marker="o", color='#08519c')
    
    # avg line
    if show_avg:
        avg = trend["count"].mean()
        plt.axhline(avg, color='#e6550d', linestyle='--', label=f"Average: {avg:.0f}")
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title or f"Trend of {df_name or count_col or 'count'} over years")

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save figure
    if save:
        fname = f"{df_name}_trend.png" if df_name else "trend.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=dpi)

    # Show figure
    if show:
        plt.show()
    plt.close()

def plot_corr_matrix(df: pd.DataFrame,
                     method: str = 'spearman',
                     df_name: str | None = None,
                     title: str | None = None,
                     figsize: tuple | None = (12,10),
                     dpi: int | None = 200,
                     save: bool | None = True,
                     show: bool | None = True,):
    """
    Plot a correlation heatmap (upper triangle) of a DataFrame.
    
    Args:
        df          : input dataframe
        method      : correlation method ('spearman', 'pearson', 'kendall')
        df_name     : output file prefix
        title       : plot title
        figsize     : figure size
        dpi         : figure dpi
        save        : save the figure or not
        show        : display the figure or not
    """
    # Compute correlation matrix
    corr_matrix = df.corr(method=method)
    
    # Figure setup
    fig, ax = plt.subplots(figsize=figsize)
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Diverging color palette
    palette = sns.diverging_palette(250, 15, s=85, l=35, as_cmap=True)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=palette,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.4,
        cbar_kws={"fraction": 0.045, "pad": 0.03},
        ax=ax
    )
    
    ax.set_title(title or f"{method.capitalize()} Correlation Heatmap", fontsize=16, pad=20)
    
    plt.tight_layout()
    
    if save:
        fname = f"{df_name}_correlation_heatmap.png" if df_name else "correlation_heatmap.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=dpi)
    
    if show:
        plt.show()
    
    plt.close()

def plot_pca(
        df: pd.DataFrame,
        target_col: str | None = "bankrupt",
        n_components: int | None = 2,
        figsize: tuple | None = (8,6),
        dpi: int | None = 200,
        save: bool | None = True,
        show: bool | None = True,
        df_name: str | None = None,
        title: str | None  = None,
        alpha: float | None = 0.5
    ):
    """
    Perform PCA on features and plot a scatter plot of the first two principal components.

    Args:
        df          : input dataframe
        target_col  : binary target column
        n_components: number of PCA components to compute
        figsize     : figure size
        dpi         : figure dpi
        save        : whether to save the figure
        show        : whether to display the figure
        df_name     : prefix for saved file name
        title       : plot title
        alpha       : scatter point transparency
    """
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale features
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca_result = PCA(n_components=n_components).fit_transform(X_scaled)

    unique_classes = sorted(y.unique())
    
    colors = ["#4C72B0", "#DD8452"][:len(unique_classes)]
    cmap = ListedColormap(colors)

    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(pca_result[:,0], pca_result[:,1], c=y, cmap=cmap, alpha=alpha)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title or f"PCA Projection of {df_name}_{target_col}" if df_name else "PCA Projection")

    if save:
        fname = f"{df_name}_pca.png" if df_name else "pca_projection.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=dpi)

    if show:
        plt.show()
    plt.close()
