import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from math import ceil
from textwrap import wrap
from typing import Dict, Optional
from numpy.typing import ArrayLike


def boxplots(
    feature_matrix: pd.DataFrame,
    target: ArrayLike,
    order: Optional[ArrayLike] = None,
    colors: Optional[Dict] = None,
    box_to_legend_ratio: float = 1 / 5,
    ncol_subplots: Optional[int] = None,
    ncol_legend: Optional[int] = None,
    box_title_line_width: int = 20,
    box_title_font_size: int = 10,
) -> None:
    """For each column in feature_matrix create a boxplot w.r.t. target.
    Each boxplot has its own y-axis.

    If the feature_matrix has four columns, the plot could look like this:

        <boxplot of feature 1>      <boxplot of feature 2>

        <boxplot of feature 3>      <boxplot of feature 4>

                            <legend>

    The legend contains information of the coloring of the boxplots.

    Keyword arguments:
        feature_matrix:
            pd.DataFrame of numeric columns.
        target:
            Vector of classes.
            Its length should be equal to the number of rows of feature_matrix.
            Example: ["group 1", "group 2", "group 1"].
        order
            Order of appearance of the classes in the boxplots.
            Should contain precisely the unique elements of target.
            Example: ["group 2", "group 1"].
        colors
            Color of the boxplot for the class.
            Keys should be precisely the unique elements of target.
            Example: {"group 1": "blue", "group 2": "green"}.
        box_to_legend_ratio:
            "Height of a single boxplot" / "Height of the legend".
            Used to adjust the space of the legend if there are many classes.
        ncol_subplots:
            The boxplots are plotted on a grid-matrix.
            ncol_subplots is the number of columns of the grid-matrix.
        ncol_legend:
            Number of columns of the legend.
        box_title_line_width:
            Line with of the title of each boxplot.
        box_title_font_size:
            Font size of the title of each boxplot.
    """

    if order is None:
        order = np.sort(np.unique(np.array(target)))

    if colors is None:
        colors = {k: v for (k, v) in zip(order, sns.color_palette("Set1", len(order)))}

    if ncol_subplots is None:
        ncol_subplots = min(feature_matrix.shape[1], 10)

    if ncol_legend is None:
        ncol_legend = len(order)

    df = feature_matrix.copy()
    df["target"] = target

    nrow_subplots = ceil(feature_matrix.shape[1] / ncol_subplots)
    # The boxplots are plotted on a grid-matrix of dimension (nrow_subplots + 1, ncol_subplots)
    # The last row is used for the legend of the plots
    gridspec_kw = dict(
        width_ratios=[1 for i in range(ncol_subplots)],
        height_ratios=[1 for i in range(nrow_subplots)] + [box_to_legend_ratio],
    )

    fig, axes = plt.subplots(
        nrow_subplots + 1,
        ncol_subplots,
        sharey=False,
        gridspec_kw=gridspec_kw,
        tight_layout=True,
    )

    for ax in axes.flatten():
        ax.axes.get_xaxis().set_visible(False)

    for i, ax in enumerate(
        axes[
            :-1,
        ].flatten()
    ):
        if i <= df.shape[1] - 2:
            sns.boxplot(
                ax=ax,
                x="target",
                y=df.columns[i],
                data=df,
                order=order,
                palette=colors,
                hue="target",  # Add legend to each subplot (not shown, but used for overall legend)
                dodge=False,  # Don't change the position of the plot elements when using hue
            )
            ax.yaxis.set_label_text("")
            ax.set_title(
                "\n".join(
                    wrap(text=feature_matrix.columns[i], width=box_title_line_width)
                ),
                fontsize=box_title_font_size,
            )
            ax.legend().set_visible(False)  # Don't show legend of subplots
        else:
            # Remove all visible elements from the remaining subplots in the next-to-last row
            ax.axes.get_yaxis().set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

    # Remove all visible elements from the subplots in the last row
    for i in range(ncol_subplots):
        axes[nrow_subplots, i].axis("off")

    # Create legend using one of the subplot legends
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order_indices = [labels.index(str(o)) for o in order]
    handles_ordered = [handles[index] for index in order_indices]
    labels_ordered = [labels[index] for index in order_indices]
    gs = axes[nrow_subplots, 0].get_gridspec()
    ax_legend = fig.add_subplot(gs[nrow_subplots, :])
    ax_legend.axis("off")
    plt.legend(
        handles=handles_ordered, labels=labels_ordered, loc="center", ncol=ncol_legend
    )
