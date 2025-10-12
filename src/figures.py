import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from my_utils import set_mystyle
from results_utils import get_divercity_at_k, get_divercity_vs_radius_at_k, filter_by_p_eps

from matplotlib.patches import PathPatch
from matplotlib import path

from scipy.stats import spearmanr, pearsonr


def create_fig1c(dict_df_results, list_cities, k, p, eps, cities_to_annotate=[], 
                 save=False, savepath=None):
    """
    Create Figure 1c: KDE of city-level median DiverCity values across all cities.
    
    Parameters
    ----------
    dict_df_results : dict
        Dictionary mapping city names to their corresponding results DataFrame.
    list_cities : list
        List of city names to include in the plot.
    k, p, eps : int or float
        Parameters used in DiverCity computation.
    cities_to_annotate : list, optional
        Cities to annotate in the plot.
    save : bool, optional
        Whether to save the figure (default: False).
    savepath : str, optional
        Path (including filename and extension) where the figure should be saved.
        Example: "figures/Fig1c_k10_p01_eps30.png"
    """

    medians = []
    city_to_median_dc = {}
    city_to_iqr_dc = {}
    
    # Compute median and IQR DiverCity for each city
    for city in list_cities:
        df = dict_df_results[city]
        y_vector = get_divercity_at_k(df, k, p, eps)
        
        median_dc = np.median(y_vector)
        iqr_dc = np.percentile(y_vector, 75) - np.percentile(y_vector, 25)
        
        medians.append(median_dc)
        city_to_median_dc[city] = median_dc
        city_to_iqr_dc[city] = iqr_dc

    # Plot KDE of median DiverCity across cities
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    kde_line = sns.kdeplot(
        [city_to_median_dc[c] for c in list_cities], 
        alpha=0.8, color="k", linewidth=1, ax=ax
    ).get_lines()[0].get_data()
    
    x, y = kde_line
    kde_path = path.Path(np.column_stack([x, y]))
    patch = PathPatch(kde_path, facecolor="none", edgecolor="grey", hatch='//', alpha=0.1)
    ax.add_patch(patch)

    # Annotate selected cities
    for city in cities_to_annotate:
        x_annotate = city_to_median_dc[city]
        y_annotate = np.interp(x_annotate, x, y)
        ax.scatter(x_annotate, y_annotate, s=40, marker="o", c="k", zorder=3)
        ax.scatter(x_annotate, y_annotate, s=15, marker="o", c="white", zorder=4)  
        ax.annotate(
            city.replace("_", " ").title(), 
            (x_annotate * 1.01, y_annotate),
            weight=600, ha="left", va="center", fontsize=8
        )
    
    # Style adjustments
    ax.set_ylabel("Density", fontsize=10, fontweight=400)
    ax.set_xlabel(r"$\mathcal{D}_C$", fontsize=10, fontweight=400)
    set_mystyle(ax, size_text_ticks=8)

    plt.tight_layout()

    # Optionally save the figure
    if save:
        if savepath is None:
            savepath = f"Fig1c_k{k}_p{p}_eps{eps}.png"
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax



def create_fig1d(dict_df_results, list_cities, k, p, eps, 
                 save=False, savepath=None, show=True):
    """
    Create Figure 1d: DiverCity as a function of distance from the city center.
    
    Parameters
    ----------
    dict_df_results : dict
        Dictionary mapping city names to their corresponding results DataFrame.
    list_cities : list
        List of city names to include in the plot.
    k, p, eps : int or float
        Parameters used in DiverCity computation.
    save : bool, optional
        Whether to save the figure (default: False).
    savepath : str, optional
        Path (including filename and extension) where the figure should be saved.
        Example: "figures/Fig1d_k10_p01_eps30.png"
    show : bool, optional
        Whether to display the figure (default: True).
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    df_average_all_cities = pd.DataFrame()

    # Plot each city's DiverCity vs radius
    for city in list_cities:
        city_norm = city.lower().replace(" ", "_")
        df = dict_df_results[city_norm]
        y_vector = get_divercity_vs_radius_at_k(df, k, p, eps)
        
        ax.plot(
            np.arange(1, len(y_vector) + 1, 1),
            y_vector,
            color="black",
            alpha=1,
            linestyle=":",
            linewidth=0.2
        )

        df_average_all_cities[city_norm] = y_vector

    # Compute median and interquartile range across cities
    y_all_cities = df_average_all_cities.T.median().values
    percentile_25 = df_average_all_cities.T.apply(lambda col: np.percentile(col, 25))
    percentile_75 = df_average_all_cities.T.apply(lambda col: np.percentile(col, 75))

    # Global average curve
    ax.plot(
        np.arange(1, len(y_vector) + 1, 1),
        y_all_cities,
        color="black",
        alpha=1,
        linestyle="-",
        linewidth=1,
        label="Global average"
    )

    # Interquartile shaded area
    ax.fill_between(
        np.arange(1, len(y_vector) + 1, 1),
        percentile_25,
        percentile_75,
        color='lightblue',
        alpha=0.6,
        label="Interquartile range",
        linewidth=0
    )

    # Labels and style
    ax.set_ylabel(r"${\mathcal{D}(u, v)}$", fontsize=10, fontweight=400)
    ax.set_xlabel("radius [km]", fontsize=10, fontweight=400)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    set_mystyle(ax, size_text_ticks=8)

    plt.tight_layout()

    # Optionally save figure
    if save:
        if savepath is None:
            savepath = f"Fig1d_k{k}_p{p}_eps{eps}.png"
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    # Optionally show
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def create_fig2c(df_node_dcity, list_percentiles, at_k, save=False, savepath=None, show=True):
    """Plot Figure 2c: Average attractor distance vs node DiverCity percentile."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    list_distances, prev_value = [], 0

    for pc in list_percentiles:
        if pc == 10:
            avg_dist = df_node_dcity[df_node_dcity[f"div_k_{at_k}"] < df_node_dcity[f"pct_{pc}"]].mean()["dist_attr"]
        else:
            avg_dist = df_node_dcity[
                (df_node_dcity[f"div_k_{at_k}"] > df_node_dcity[f"pct_{prev_value}"]) &
                (df_node_dcity[f"div_k_{at_k}"] < df_node_dcity[f"pct_{pc}"])
            ].mean()["dist_attr"]
        list_distances.append(avg_dist)
        prev_value = pc

    ax.axhline(df_node_dcity["dist_attr"].mean(), color="grey", linestyle="--", linewidth=0.5, label="global average")
    ax.plot(list_percentiles, list_distances, markersize=7, c="k", marker=".", linewidth=0.5, linestyle=":")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax.set_ylabel("avg. attractor distance [km]", fontsize=10, fontweight=400)
    ax.set_xlabel(r"percentile range of $\mathcal{D}(i)$", fontsize=10, fontweight=400)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_xticklabels(["10-20", "30-40", "50-60", "70-80", "90-100"], fontsize=7)
    set_mystyle(ax, size_text_ticks=8)

    plt.tight_layout()
    if save:
        savepath = savepath or "Fig2c_distance_vs_divercity.png"
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def create_fig2d(city_to_attr_density, city_to_median_dc, city_to_attr_homo,
                 list_cities, save=False, savepath=None, show=True):
    """Plot Figure 2d: DiverCity vs attractor density, colored by spatial dispersion H."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    x_list = [city_to_attr_density[c] for c in list_cities]
    y_list = [city_to_median_dc[c] for c in list_cities]
    colors = [city_to_attr_homo[c] for c in list_cities]

    pearson_corr, _ = pearsonr(x_list, y_list)
    spearman_corr, _ = spearmanr(x_list, y_list)

    scatter = ax.scatter(x_list, y_list, s=23, edgecolors="black",
                         c=colors, alpha=.7, linewidths=0, cmap="Greys", vmin=-1)

    ax.set_ylabel(r"$\mathcal{D}_C$", fontsize=10, fontweight=400)
    ax.set_xlabel("attractor density", fontsize=10, fontweight=400)
    ax.text(0.75, 0.12, rf"$r=${round(pearson_corr,3)}", transform=ax.transAxes, fontsize=8)
    ax.text(0.75, 0.05, rf"$\rho=${round(spearman_corr,3)}", transform=ax.transAxes, fontsize=8)

    set_mystyle(ax, size_text_ticks=8)

    colorbar = fig.colorbar(scatter, ax=ax, label="H", fraction=0.1, pad=0.01)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.ax.yaxis.label.set_size(10)
    colorbar.outline.set_linewidth(0.3)
    colorbar.outline.set_edgecolor("lightgray")

    plt.tight_layout()
    if save:
        savepath = savepath or "Fig2d_density_vs_divercity.png"
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax



def create_fig3a(dict_gains_DiverCity_speed, speeds_to_load, 
                 show_cities=[], save=False, savepath=None, show=True):
    """
    Plot Figure 3a: DiverCity gain vs speed reduction (%)
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    df_gains = pd.DataFrame()

    for city, results in dict_gains_DiverCity_speed.items():
        y_gains = results["median"] + [0]
        df_gains[city] = y_gains[::-1]

        if city in show_cities:
            line, = ax.plot(
                [0] + speeds_to_load,
                [0] + results["median"][::-1],
                color="black", linewidth=1, alpha=0.5, linestyle="--"
            )
            line.set_dashes([12, 12])
            ax.annotate(city.replace('_', ' ').title(),
                        (speeds_to_load[-1], results["median"][0]),
                        fontsize=8, ha="right", va="center", fontweight=600)

    avg_line = df_gains.T.mean()
    p25 = df_gains.T.apply(lambda col: np.percentile(col, 25))
    p75 = df_gains.T.apply(lambda col: np.percentile(col, 75))

    ax.plot([0] + speeds_to_load, avg_line, color="black", linewidth=1, marker=".")
    ax.fill_between([0] + speeds_to_load, p25, p75, color="lightblue", alpha=0.8,
                    label="Interquartile range", linewidth=0)

    ax.set_xlabel("Speed reduction (%)", fontsize=10, weight=400)
    ax.set_ylabel(r"$\mathcal{D}_C$ increase", fontsize=10, weight=400)
    set_mystyle(ax, size_text_ticks=8)

    plt.tight_layout()
    if save:
        savepath = savepath or "Fig3a_speed_vs_divercity.png"
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def create_fig3b(dict_tt_difference, speeds_to_load, 
                 show_cities=[], save=False, savepath=None, show=True):
    """
    Plot Figure 3b: Travel time increase vs speed reduction (%)
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    df_tt_gains = pd.DataFrame()

    for city, results in dict_tt_difference.items():
        tt_in_minutes = list(np.array(results["trip"]) / 60) + [0]
        tt_in_minutes = tt_in_minutes[::-1]
        df_tt_gains[city] = tt_in_minutes

        if city in show_cities:
            line, = ax.plot([0] + speeds_to_load, tt_in_minutes,
                            color="black", linewidth=1, alpha=0.5, linestyle="--")
            line.set_dashes([12, 12])
            ax.annotate(city.replace('_', ' ').title(),
                        (speeds_to_load[-1], tt_in_minutes[-1]),
                        fontsize=8, ha="right", va="center", fontweight=600)

    avg_line = df_tt_gains.T.mean()
    p25 = df_tt_gains.T.apply(lambda col: np.percentile(col, 25))
    p75 = df_tt_gains.T.apply(lambda col: np.percentile(col, 75))

    ax.plot([0] + speeds_to_load, avg_line, color="black", linewidth=1, marker=".")
    ax.fill_between([0] + speeds_to_load, p25, p75, color="lightblue", alpha=0.8,
                    label="Interquartile range", linewidth=0)

    ax.set_xlabel("Speed reduction (%)", fontsize=10, weight=400)
    ax.set_ylabel("Travel time increase (min)", fontsize=10, weight=400)
    set_mystyle(ax, size_text_ticks=8)

    plt.tight_layout()
    if save:
        savepath = savepath or "Fig3b_speed_vs_traveltime.png"
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def create_fig3c(city, dict_df_results, dict_df_speed, speeds_to_plot, at_k, p, eps,
                  palette=None, save=False, savepath=None, show=True):
    """
    Create Figure 3c: DiverCity vs radius under multiple speed-reduction scenarios.

    Parameters
    ----------
    city : str
        City name (e.g. 'rome').
    dict_df_results : dict
        Baseline DiverCity dataframes for each city.
    dict_df_speed : dict
        Nested dict of {speed_reduction: {city_name: dataframe}}.
    speeds_to_plot : list[int]
        Speed reductions (%) to include, e.g. [10, 50, 90].
    at_k, p, eps : numeric
        DiverCity computation parameters.
    palette : list or None
        List of colors (len = len(speeds_to_plot)+1). Defaults to sns.color_palette("Reds_r", ...).
    save : bool
        Whether to save the figure.
    savepath : str or None
        Path (incl. filename) where to save the figure.
    show : bool
        Whether to display the figure.

    Returns
    -------
    fig, ax : Matplotlib Figure and Axes
    """

    # Palette
    if palette is None:
        palette = sns.color_palette("Reds_r", len(speeds_to_plot) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Baseline DiverCity vs radius
    df_base = dict_df_results[city]
    y_baseline = get_divercity_vs_radius_at_k(df_base, at_k, p, eps)

    # Speed reduction scenarios
    for i, s in enumerate(speeds_to_plot):
        df_speed = dict_df_speed[s][f"{city}_speed{s}"]
        y_speed = get_divercity_vs_radius_at_k(df_speed, at_k, p, eps)

        ax.plot(
            np.arange(1, len(y_speed) + 1),
            y_speed,
            color=palette[i],
            alpha=1,
            linestyle="-",
            linewidth=0.75,
            label=f"{100 - s}%",
        )

    # Baseline (0% reduction)
    line, = ax.plot(
        np.arange(1, len(y_baseline) + 1),
        y_baseline,
        color="black",
        alpha=1,
        linestyle="--",
        linewidth=1,
        label="0%",
        zorder=10
    )
    line.set_dashes([3, 3])

    # Axis labels and style
    ax.set_ylabel(r"${\mathcal{D}(u, v)}$", fontsize=10, fontweight=400)
    ax.set_xlabel("radius [km]", fontsize=10, fontweight=400)
    set_mystyle(ax, size_text_ticks=8)

    # Legend
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower right",
        title="Speed Reductions",
        title_fontsize=8
    )

    plt.tight_layout()

    # Save / show
    if save:
        savepath = savepath or f"Fig3_{city}_speed_profiles.png"
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
