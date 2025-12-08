import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from .common import OUTPUT_DIR
import shap

def plot_pie(
        df: pd.DataFrame,
        col: str,
        df_name: str | None = None,
        title: str | None = None,
        figsize: tuple | None = (5,5),
        dpi: int | None = 135,
        save: bool | None = True,
        show: bool | None = True,
        label_map: dict | None = None,
        colors: list | None = None
    ):
    """
    Plot a clean, professional pie chart for categorical data.

    Args:
        df          : input dataframe
        col         : column name to plot
        df_name     : output file prefix for saving
        title       : title of the plot
        figsize     : figure size (width, height)
        dpi         : figure resolution
        save        : whether to save the figure
        show        : whether to display the figure
        label_map   : optional dictionary to map values to labels
        colors      : optional list of colors for the pie segments
    """
    counts = df[col].value_counts().sort_index()
    labels = counts.index.tolist()
    if label_map:
        labels = [label_map.get(l, l) for l in labels]
    values = counts.values.tolist()

    # highlight small categories with explode
    explode = [0.05 if v/np.sum(values) < 0.15 else 0 for v in values]

    plt.figure(figsize=figsize, dpi=dpi)
    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
        startangle=90,
        counterclock=False,
        explode=explode,
        colors=colors or ["#4C72B0", "#DD8452"],
        wedgeprops=dict(edgecolor='w')
    )

    for t in texts + autotexts:
        t.set_fontsize(10)

    plt.title(title or f"{df_name}_{col}" if df_name else f"{col}", fontsize=12)
    plt.axis('equal')

    if save:
        fname = f"{df_name}_{col}.png" if df_name else f"{col}.png"
        plt.savefig(Path(OUTPUT_DIR)/fname, dpi=dpi)
    if show:
        plt.show()
    plt.close()

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
        label_map: dict | None = None,  
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
        label_map   : optional dict mapping bin intervals to labels
        figsize     : figure size
        dpi         : figure dpi
        save        : save the figure or not
        show        : display the figure or not
    """
    df = df.copy()
    
    # Create bins
    df['bin'] = pd.cut(df[col], bins=bins)
    
    # Calculate bankruptcy rate per bin
    rate = df.groupby('bin', observed=True)[target_col].mean().reset_index()
    
    # Apply label mapping if provided
    if label_map:
        rate['bin_label'] = rate['bin'].map(label_map)
    else:
        rate['bin_label'] = rate['bin'].astype(str)
    
    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='bin_label', y=target_col, data=rate, color='#DD8452')
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
    
    colors = ["#491D7D", "#EDED74"][:len(unique_classes)]
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

def plot_classification_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    title: str | None = None,
    figsize: tuple[int,int] = (7,5),
    dpi: int | None = 100,
    save: bool | None = True,
    show: bool | None = True,
    labels_map: dict | None = None,
    palette: dict | None = None
) -> None:
    """
    Plot per-class precision, recall, f1-score as a horizontal bar chart.

    Args:
        y_true    : true labels
        y_pred    : predicted labels
        title     : plot title
        figsize   : figure size
        dpi       : figure dpi
        save      : whether to save figure
        show      : whether to display figure
        labels_map: dict to rename class labels
        palette   : dict of colors for precision/recall/f1
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        digits=3,
        labels=[0,1],
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T

    for c in ['0','1']:
        if c not in report_df.index:
            report_df.loc[c] = {'precision':0,'recall':0,'f1-score':0,'support':0}
    report_df = report_df.loc[['0','1']]

    plot_df = report_df.reset_index().melt(
        id_vars='index',
        value_vars=['precision','recall','f1-score'],
        var_name='Metric',
        value_name='Score'
    )

    plot_df.loc[
        plot_df['index'].isin(['0', '1']),
        'index'
    ] = plot_df.loc[
        plot_df['index'].isin(['0', '1']),
        'index'
    ].astype(int)

    if labels_map:
        plot_df['index'] = plot_df['index'].map(labels_map).fillna(plot_df['index'])

    default_palette = {'precision':'#3498db', 'recall':'#9b59b6', 'f1-score':'#f39c12'}
    palette = palette or default_palette

    sns.set_style("whitegrid")
    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.barplot(data=plot_df, x='Score', y='index', hue='Metric', palette=palette, dodge=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Class")
    ax.set_xlim(0,1)
    ax.set_title(title or "Classification Report")

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.2f}", (width, p.get_y() + p.get_height()/2),
                    ha='left', va='center', fontsize=9)

    if save:
        fname = f"{title}_classification_report.png" if title else "classification_report.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    title: str | None = None,
    figsize: tuple[int,int] = (6,5),
    dpi: int | None = 130,
    save: bool | None = True,
    show: bool | None = True,
    labels_map: dict | None = None,
    cmap: str = "Reds"
) -> None:
    """
    Plot confusion matrix 

    Args:
        y_true    : true labels
        y_pred    : predicted labels
        title     : plot title
        figsize   : figure size
        dpi       : figure dpi
        save      : whether to save figure
        show      : whether to display figure
        labels_map: dict to rename class labels
        cmap      : colormap for heatmap
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm_rates = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    labels = [labels_map.get(i, i) for i in [0,1]] if labels_map else [0,1]

    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(cm_rates, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title or "Confusion Matrix")
    plt.tight_layout()

    if save:
        fname = f"{title}_confusion_matrix.png" if title else "confusion_matrix.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()

def plot_grouped(
    df: pd.DataFrame,
    group_col: str,
    count_col: str,
    df_name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = "Count",
    figsize: tuple[int,int] | None = (8,5),
    dpi: int | None = 200,
    save: bool | None = True,
    show: bool | None = True,
    show_pct: bool | None = False,
    colors: list[str] | None = None
) -> None:
    """
    Plot a grouped bar chart for two categorical columns, each group in group_col is split by count_col.

    Args:
        df        : input dataframe
        group_col : column used as the main group (e.g., Altman prediction)
        count_col : column used to split bars within each group (e.g., actual bankrupt)
        df_name   : file prefix when saving
        title     : plot title
        xlabel    : x-axis label
        ylabel    : y-axis label
        figsize   : figure size
        dpi       : figure resolution
        save      : whether to save figure
        show      : whether to display figure
        show_pct  : display percentage instead of raw counts
        colors    : optional list of colors for the bars
    """
    plt.figure(figsize=figsize)
    
    # create a crosstab
    ct = pd.crosstab(df[group_col], df[count_col])
    
    # convert to proportion
    if show_pct:
        ct = ct.div(ct.sum(axis=1), axis=0) * 100
        ylabel = "Percentage (%)"
    
    ct.plot(kind="bar", stacked=False, color=colors, figsize=figsize, width=0.7)
    
    plt.title(title or (f"{df_name}_{group_col}_vs_{count_col}" if df_name else f"{group_col} vs {count_col}"))
    plt.xlabel(xlabel or group_col)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    
    if save:
        fname = f"{df_name}_{group_col}_vs_{count_col}.png" if df_name else f"{group_col}_vs_{count_col}.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_shap(model, X_train: pd.DataFrame):
    """
    Global SHAP feature importance bar plot.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap_abs_mean = np.abs(shap_values[:, :, 1]).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": shap_abs_mean
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 10))
    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )
    plt.gca().invert_yaxis()  
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global SHAP Feature Importance (Class 1)")

    plt.tight_layout()
    plt.show()
    plt.savefig(OUTPUT_DIR / "shap_global_importance.png", dpi=200)
    plt.close()


def plot_shap_summary_class(model, X_train: pd.DataFrame):
    """
    SHAP summary plot for class 1 (Bankrupt)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    shap_matrix = shap_values[:, :, 1]
    
    plt.figure(figsize=(8,6))

    shap.summary_plot(shap_matrix, X_train, show=True)
    plt.savefig(OUTPUT_DIR / "shap_summary_class1.png", dpi=200)
    plt.close()

def plot_decision_tree(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 200
) -> None:
    """
    Plot and save a trained Decision Tree.
    """
    plt.figure(figsize=figsize)
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=12
    )
    plt.savefig(OUTPUT_DIR / "decision_tree.png", dpi=200)
    plt.show()
    plt.close()

