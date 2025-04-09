import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy
import seaborn as sbn
from esda.moran import Moran, Moran_BV, Moran_Local, Moran_Local_BV
from libpysal.weights.spatial_lag import lag_spatial
from matplotlib import colors, patches
from spreg import OLS

# from ._viz_utils import mask_local_auto, moran_hot_cold_spots, splot_colors

"""
Lightweight visualizations for esda using Matplotlib and Geopandas

TODO
* geopandas plotting, change round shapes in legends to boxes
* prototype moran_facet using `seaborn.FacetGrid`
"""

__author__ = "Stefanie Lumnitz <stefanie.lumitz@gmail.com>"


def _create_moran_fig_ax(ax, figsize, aspect_equal):
    """
    Creates matplotlib figure and axes instances
    for plotting moran visualizations. Adds common viz design.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    ax.spines["left"].set_position(("axes", -0.05))
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.spines["top"].set_color("none")
    if aspect_equal is True:
        ax.set_aspect("equal")
    return fig, ax


def moran_scatterplot_mm(
    moran,
    zstandard=True,
    p=None,
    aspect_equal=True,
    ax=None,
    scatter_kwds=None,
    fitline_kwds=None,
    axis_kwds=None,
):
    """
    Moran Scatterplot

    Parameters
    ----------
    moran : esda.moran instance
        Values of Moran's I Global, Bivariate and Local
        Autocorrelation Statistics
    zstandard : bool, optional
        If True, Moran Scatterplot will show z-standardized attribute and
        spatial lag values. Default =True.
    p : float, optional
        If given, the p-value threshold for significance
        for Local Autocorrelation analysis. Points will be colored by
        significance. By default it will not be colored.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes will show the same aspect or visual proportions
        for Moran Scatterplot.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import (Moran, Moran_BV,
    ...                         Moran_Local, Moran_Local_BV)
    >>> from splot.esda import moran_scatterplot

    Load data and calculate weights

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'

    Calculate esda.moran Objects

    >>> moran = Moran(y, w)
    >>> moran_bv = Moran_BV(y, x, w)
    >>> moran_loc = Moran_Local(y, w)
    >>> moran_loc_bv = Moran_Local_BV(y, x, w)

    Plot

    >>> fig, axs = plt.subplots(2, 2, figsize=(10,10),
    ...                         subplot_kw={'aspect': 'equal'})
    >>> moran_scatterplot(moran, p=0.05, ax=axs[0,0])
    >>> moran_scatterplot(moran_loc, p=0.05, ax=axs[1,0])
    >>> moran_scatterplot(moran_bv, p=0.05, ax=axs[0,1])
    >>> moran_scatterplot(moran_loc_bv, p=0.05, ax=axs[1,1])
    >>> plt.show()

    """
    if isinstance(moran, Moran):
        if p is not None:
            warnings.warn(
                "`p` is only used for plotting `esda.moran.Moran_Local`\n"
                "or `Moran_Local_BV` objects"
            )
        fig, ax = _moran_global_scatterplot(
            moran=moran,
            zstandard=zstandard,
            ax=ax,
            aspect_equal=aspect_equal,
            scatter_kwds=scatter_kwds,
            fitline_kwds=fitline_kwds,
        )
    elif isinstance(moran, Moran_BV):
        if p is not None:
            warnings.warn(
                "`p` is only used for plotting `esda.moran.Moran_Local` "
                "or `Moran_Local_BV` objects."
            )
        fig, ax = _moran_bv_scatterplot(
            moran_bv=moran,
            ax=ax,
            aspect_equal=aspect_equal,
            scatter_kwds=scatter_kwds,
            fitline_kwds=fitline_kwds,
        )
    elif isinstance(moran, Moran_Local):
        fig, ax = _moran_loc_scatterplot(
            moran_loc=moran,
            zstandard=zstandard,
            ax=ax,
            p=p,
            aspect_equal=aspect_equal,
            scatter_kwds=scatter_kwds,
            fitline_kwds=fitline_kwds,
        )
    elif isinstance(moran, Moran_Local_BV):
        fig, ax = _moran_loc_bv_scatterplot(
            moran_loc_bv=moran,
            ax=ax,
            p=p,
            aspect_equal=aspect_equal,
            scatter_kwds=scatter_kwds,
            fitline_kwds=fitline_kwds,
        )
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    return fig, ax


def _moran_global_scatterplot(
    moran,
    zstandard=True,
    aspect_equal=True,
    ax=None,
    scatter_kwds=None,
    fitline_kwds=None,
    axis_kwds=None,
):
    """
    Global Moran's I Scatterplot.

    Parameters
    ----------
    moran : esda.moran.Moran instance
        Values of Moran's I Global Autocorrelation Statistics
    zstandard : bool, optional
        If True, Moran Scatterplot will show z-standardized attribute and
        spatial lag values. Default =True.
    aspect_equal : bool, optional
        If True, Axes will show the same aspect or visual proportions.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran
    >>> from splot.esda import moran_scatterplot

    Load data and calculate weights

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'

    Calculate Global Moran

    >>> moran = Moran(y, w)

    plot

    >>> moran_scatterplot(moran)
    >>> plt.show()

    customize plot

    >>> fig, ax = moran_scatterplot(moran, zstandard=False,
    ...                             fitline_kwds=dict(color='#4393c3'))
    >>> ax.set_xlabel('Donations')
    >>> plt.show()

    """
    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()
    if fitline_kwds is None:
        axis_kwds = dict()

    # define customization defaults
    scatter_kwds.setdefault("alpha", 0.6)
    scatter_kwds.setdefault("color", splot_colors["moran_base"])
    scatter_kwds.setdefault("s", 40)

    fitline_kwds.setdefault("alpha", 0.9)
    fitline_kwds.setdefault("color", splot_colors["moran_fit"])

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize=(7, 7), aspect_equal=aspect_equal)

    # set labels
    ax.set_xlabel("Attribute")
    ax.set_ylabel("Spatial Lag")
    ax.set_title("Moran Scatterplot" + " (" + str(round(moran.I, 2)) + ")")

    # plot and set standards
    if zstandard is True:
        lag = lag_spatial(moran.w, moran.z)
        fit = OLS(lag[:, None], moran.z[:, None])
        # plot
        ax.scatter(moran.z, lag, **scatter_kwds)
        ax.plot(moran.z, fit.predy, **fitline_kwds)
        # v- and hlines
        ax.axvline(0, alpha=0.5, color="k", linestyle="--")
        ax.axhline(0, alpha=0.5, color="k", linestyle="--")
    else:
        lag = lag_spatial(moran.w, moran.y)
        b, a = numpy.polyfit(moran.y, lag, 1)
        # plot
        ax.scatter(moran.y, lag, **scatter_kwds)
        ax.plot(moran.y, a + b * moran.y, **fitline_kwds)
        # dashed vert at mean of the attribute
        ax.vlines(moran.y.mean(), lag.min(), lag.max(), alpha=0.5, linestyle="--")
        # dashed horizontal at mean of lagged attribute
        ax.hlines(lag.mean(), moran.y.min(), moran.y.max(), alpha=0.5, linestyle="--")
    return fig, ax


def plot_moran_simulation(
    moran, aspect_equal=True, ax=None, fitline_kwds=None, **kwargs
):
    """
    Global Moran's I simulated reference distribution.

    Parameters
    ----------
    moran : esda.moran.Moran instance
        Values of Moran's I Global Autocorrelation Statistics
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the
        vertical moran fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborn.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Simulated reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran
    >>> from splot.esda import plot_moran_simulation

    Load data and calculate weights

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'

    Calculate Global Moran

    >>> moran = Moran(y, w)

    plot

    >>> plot_moran_simulation(moran)
    >>> plt.show()

    customize plot

    >>> plot_moran_simulation(moran, fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()

    """
    # to set default as an empty dictionary that is later filled with defaults
    if fitline_kwds is None:
        fitline_kwds = dict()

    figsize = kwargs.pop("figsize", (7, 7))

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize, aspect_equal=aspect_equal)

    # plot distribution
    shade = kwargs.pop("shade", True)
    color = kwargs.pop("color", splot_colors["moran_base"])
    sbn.kdeplot(moran.sim, fill=shade, color=color, ax=ax, **kwargs)

    # customize plot
    fitline_kwds.setdefault("color", splot_colors["moran_fit"])
    ax.vlines(moran.I, 0, 1, **fitline_kwds)
    ax.vlines(moran.EI, 0, 1)
    ax.set_title("Reference Distribution")
    ax.set_xlabel("Moran I: " + str(round(moran.I, 2)))
    return fig, ax


def plot_moran(
    moran,
    zstandard=True,
    aspect_equal=True,
    scatter_kwds=None,
    fitline_kwds=None,
    **kwargs,
):
    """
    Global Moran's I simulated reference distribution and scatterplot.

    Parameters
    ----------
    moran : esda.moran.Moran instance
        Values of Moran's I Global Autocorrelation Statistics
    zstandard : bool, optional
        If True, Moran Scatterplot will show z-standardized attribute and
        spatial lag values. Default =True.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline
        and vertical fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborne.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran scatterplot and reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran
    >>> from splot.esda import plot_moran

    Load data and calculate weights

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'

    Calculate Global Moran

    >>> moran = Moran(y, w)

    plot

    >>> plot_moran(moran)
    >>> plt.show()

    customize plot

    >>> plot_moran(moran, zstandard=False,
    ...            fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()

    """
    figsize = kwargs.pop("figsize", (10, 4))
    fig, axs = plt.subplots(1, 2, figsize=figsize, subplot_kw={"aspect": "equal"})
    plot_moran_simulation(moran, ax=axs[0], fitline_kwds=fitline_kwds, **kwargs)
    moran_scatterplot(
        moran,
        zstandard=zstandard,
        ax=axs[1],
        scatter_kwds=scatter_kwds,
        fitline_kwds=fitline_kwds,
    )
    axs[0].set(aspect="auto")
    if aspect_equal is True:
        axs[1].set_aspect("equal", "datalim")
    else:
        axs[1].set_aspect("auto")
    return fig, axs


def _moran_bv_scatterplot(
    moran_bv, ax=None, aspect_equal=True, scatter_kwds=None, fitline_kwds=None
):
    """
    Bivariate Moran Scatterplot.

    Parameters
    ----------
    moran_bv : esda.moran.Moran_BV instance
        Values of Bivariate Moran's I Autocorrelation Statistics
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate moran scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran_BV
    >>> from splot.esda import moran_scatterplot

    Load data and calculate weights

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'

    Calculate Bivariate Moran

    >>> moran_bv = Moran_BV(x, y, w)

    plot

    >>> moran_scatterplot(moran_bv)
    >>> plt.show()

    customize plot

    >>> moran_scatterplot(moran_bv,
    ...                      fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()

    """
    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()

    # define customization
    scatter_kwds.setdefault("alpha", 0.6)
    scatter_kwds.setdefault("color", splot_colors["moran_base"])
    scatter_kwds.setdefault("s", 40)

    fitline_kwds.setdefault("alpha", 0.9)
    fitline_kwds.setdefault("color", splot_colors["moran_fit"])

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize=(7, 7), aspect_equal=aspect_equal)

    # set labels
    ax.set_xlabel("Attribute X")
    ax.set_ylabel("Spatial Lag of Y")
    ax.set_title("Bivariate Moran Scatterplot" + " (" + str(round(moran_bv.I, 2)) + ")")

    # plot and set standards
    lag = lag_spatial(moran_bv.w, moran_bv.zy)
    fit = OLS(lag[:, None], moran_bv.zx[:, None])
    # plot
    ax.scatter(moran_bv.zx, lag, **scatter_kwds)
    ax.plot(moran_bv.zx, fit.predy, **fitline_kwds)
    # v- and hlines
    ax.axvline(0, alpha=0.5, color="k", linestyle="--")
    ax.axhline(0, alpha=0.5, color="k", linestyle="--")
    return fig, ax


def plot_moran_bv_simulation(
    moran_bv, ax=None, aspect_equal=True, fitline_kwds=None, **kwargs
):
    """
    Bivariate Moran's I simulated reference distribution.

    Parameters
    ----------
    moran_bv : esda.moran.Moran_BV instance
        Values of Bivariate Moran's I Autocorrelation Statistics
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the
        vertical moran fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborne.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate moran reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran_BV
    >>> from splot.esda import plot_moran_bv_simulation

    Load data and calculate weights

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'

    Calculate Bivariate Moran

    >>> moran_bv = Moran_BV(x, y, w)

    plot

    >>> plot_moran_bv_simulation(moran_bv)
    >>> plt.show()

    customize plot

    >>> plot_moran_bv_simulation(moran_bv,
    ...                          fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()

    """
    # to set default as an empty dictionary that is later filled with defaults
    if fitline_kwds is None:
        fitline_kwds = dict()

    figsize = kwargs.pop("figsize", (7, 7))

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize, aspect_equal=aspect_equal)

    # plot distribution
    shade = kwargs.pop("shade", True)
    color = kwargs.pop("color", splot_colors["moran_base"])
    sbn.kdeplot(moran_bv.sim, fill=shade, color=color, ax=ax, **kwargs)

    # customize plot
    fitline_kwds.setdefault("color", splot_colors["moran_fit"])
    ax.vlines(moran_bv.I, 0, 1, **fitline_kwds)
    ax.vlines(moran_bv.EI_sim, 0, 1)
    ax.set_title("Reference Distribution")
    ax.set_xlabel("Bivariate Moran I: " + str(round(moran_bv.I, 2)))
    return fig, ax


def plot_moran_bv(
    moran_bv, aspect_equal=True, scatter_kwds=None, fitline_kwds=None, **kwargs
):
    """
    Bivariate Moran's I simulated reference distribution and scatterplot.

    Parameters
    ----------
    moran_bv : esda.moran.Moran_BV instance
        Values of Bivariate Moran's I Autocorrelation Statistics
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline
        and vertical fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborne.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate moran scatterplot and reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran_BV
    >>> from splot.esda import plot_moran_bv

    Load data and calculate weights

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'

    Calculate Bivariate Moran

    >>> moran_bv = Moran_BV(x, y, w)

    plot

    >>> plot_moran_bv(moran_bv)
    >>> plt.show()

    customize plot

    >>> plot_moran_bv(moran_bv, fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()

    """
    figsize = kwargs.pop("figsize", (10, 4))
    fig, axs = plt.subplots(1, 2, figsize=figsize, subplot_kw={"aspect": "equal"})
    plot_moran_bv_simulation(moran_bv, ax=axs[0], fitline_kwds=fitline_kwds, **kwargs)
    moran_scatterplot(
        moran_bv,
        ax=axs[1],
        aspect_equal=aspect_equal,
        scatter_kwds=scatter_kwds,
        fitline_kwds=fitline_kwds,
    )
    axs[0].set(aspect="auto")
    if aspect_equal is True:
        axs[1].set_aspect("equal", "datalim")
    else:
        axs[1].set(aspect="auto")
    return fig, axs


def _moran_loc_scatterplot(
    moran_loc,
    zstandard=True,
    p=None,
    aspect_equal=True,
    ax=None,
    scatter_kwds=None,
    fitline_kwds=None,
):
    """
    Moran Scatterplot with option of coloring of Local Moran Statistics

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local instance
        Values of Moran's I Local Autocorrelation Statistics
    p : float, optional
        If given, the p-value threshold for significance. Points will
        be colored by significance. By default it will not be colored.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran Local scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> import geopandas as gpd
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> from esda.moran import Moran_Local
    >>> from splot.esda import moran_scatterplot

    Load data and calculate Moran Local statistics

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> m = Moran_Local(y, w)

    plot

    >>> moran_scatterplot(m)
    >>> plt.show()

    customize plot

    >>> moran_scatterplot(m, p=0.05,
    ...                   fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()

    """
    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()

    if p is not None:
        if not isinstance(moran_loc, Moran_Local):
            raise ValueError(
                "`moran_loc` is not a\n " + "esda.moran.Moran_Local instance"
            )
        if "color" in scatter_kwds or "c" in scatter_kwds or "cmap" in scatter_kwds:
            warnings.warn(
                "To change the color use cmap with a colormap of 5,\n"
                + " color defines the LISA category"
            )

        # colors
        spots = moran_hot_cold_spots(moran_loc, p)
        color_all = numpy.array(["#bababa", "#d7191c", "#abd9e9", "#2c7bb6", "#fdae61"])
        hmap = colors.ListedColormap(color_all[list(numpy.unique(spots))])

    # define customization
    scatter_kwds.setdefault("alpha", 0.6)
    scatter_kwds.setdefault("s", 40)
    fitline_kwds.setdefault("alpha", 0.9)

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize=(7, 7), aspect_equal=aspect_equal)

    # set labels
    ax.set_xlabel("Attribute")
    ax.set_ylabel("Spatial Lag")
    ax.set_title("Moran Local Scatterplot")

    # plot and set standards
    if zstandard is True:
        lag = lag_spatial(moran_loc.w, moran_loc.z)
        fit = OLS(
            lag[:, None],
            moran_loc.z[:, None],
        )
        # v- and hlines
        ax.axvline(0, alpha=0.5, color="k", linestyle="--")
        ax.axhline(0, alpha=0.5, color="k", linestyle="--")
        if p is not None:
            fitline_kwds.setdefault("color", "k")
            scatter_kwds.setdefault("cmap", hmap)
            scatter_kwds.setdefault("c", numpy.sort(spots))
            ax.plot(moran_loc.z, fit.predy, **fitline_kwds)
            ax.scatter(
                moran_loc.z[spots.argsort()], lag[spots.argsort()], **scatter_kwds
            )
        else:
            scatter_kwds.setdefault("color", splot_colors["moran_base"])
            fitline_kwds.setdefault("color", splot_colors["moran_fit"])
            ax.plot(moran_loc.z, fit.predy, **fitline_kwds)
            ax.scatter(moran_loc.z, lag, **scatter_kwds)
    else:
        lag = lag_spatial(moran_loc.w, moran_loc.y)
        b, a = numpy.polyfit(moran_loc.y, lag, 1)
        # dashed vert at mean of the attribute
        ax.vlines(moran_loc.y.mean(), lag.min(), lag.max(), alpha=0.5, linestyle="--")
        # dashed horizontal at mean of lagged attribute
        ax.hlines(
            lag.mean(), moran_loc.y.min(), moran_loc.y.max(), alpha=0.5, linestyle="--"
        )
        if p is not None:
            fitline_kwds.setdefault("color", "k")
            scatter_kwds.setdefault("cmap", hmap)
            scatter_kwds.setdefault("c", numpy.sort(spots))
            ax.plot(moran_loc.y, a + b * moran_loc.y, **fitline_kwds)
            ax.scatter(
                moran_loc.y[spots.argsort()], lag[spots.argsort()], **scatter_kwds
            )
        else:
            scatter_kwds.setdefault("c", splot_colors["moran_base"])
            fitline_kwds.setdefault("color", splot_colors["moran_fit"])
            ax.plot(moran_loc.y, a + b * moran_loc.y, **fitline_kwds)
            ax.scatter(moran_loc.y, lag, **scatter_kwds)
    return fig, ax


def lisa_cluster(
    moran_loc, gdf, p=0.05, ax=None, legend=True, legend_kwds=None, **kwargs
):
    """
    Create a LISA Cluster map

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local or Moran_Local_BV instance
        Values of Moran's Local Autocorrelation Statistic
    gdf : geopandas dataframe instance
        The Dataframe containing information to plot. Note that `gdf` will be
        modified, so calling functions should use a copy of the user
        provided `gdf`. (either using gdf.assign() or gdf.copy())
    p : float, optional
        The p-value threshold for significance. Points will
        be colored by significance.
    ax : matplotlib Axes instance, optional
        Axes in which to plot the figure in multiple Axes layout.
        Default = None
    legend : boolean, optional
        If True, legend for maps will be depicted. Default = True
    legend_kwds : dict, optional
        Dictionary to control legend formatting options. Example:
        ``legend_kwds={'loc': 'upper left', 'bbox_to_anchor': (0.92, 1.05)}``
        Default = None
    **kwargs : keyword arguments, optional
        Keywords designing and passed to geopandas.GeoDataFrame.plot().

    Returns
    -------
    fig : matplotlip Figure instance
        Figure of LISA cluster map
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran_Local
    >>> from splot.esda import lisa_cluster

    Data preparation and statistical analysis

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> moran_loc = Moran_Local(y, w)

    Plotting

    >>> fig = lisa_cluster(moran_loc, gdf)
    >>> plt.show()

    """
    # retrieve colors5 and labels from mask_local_auto
    _, colors5, _, labels = mask_local_auto(moran_loc, p=p)

    # define ListedColormap
    hmap = colors.ListedColormap(colors5)

    if ax is None:
        figsize = kwargs.pop("figsize", None)
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # check for Polygon, else no edgecolor
    if gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        gdf.assign(cl=labels).plot(
            column="cl",
            categorical=True,
            k=2,
            cmap=hmap,
            linewidth=0.1,
            ax=ax,
            edgecolor="white",
            legend=legend,
            legend_kwds=legend_kwds,
            **kwargs,
        )
    else:
        gdf.assign(cl=labels).plot(
            column="cl",
            categorical=True,
            k=2,
            cmap=hmap,
            linewidth=1.5,
            ax=ax,
            legend=legend,
            legend_kwds=legend_kwds,
            **kwargs,
        )
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def plot_local_autocorrelation(
    moran_loc,
    gdf,
    attribute,
    p=0.05,
    region_column=None,
    mask=None,
    mask_color="#636363",
    quadrant=None,
    aspect_equal=True,
    legend=True,
    scheme="Quantiles",
    cmap="YlGnBu",
    figsize=(15, 4),
    scatter_kwds=None,
    fitline_kwds=None,
):
    """
    Produce three-plot visualisation of Moran Scatteprlot, LISA cluster
    and Choropleth maps, with Local Moran region and quadrant masking

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local or Moran_Local_BV instance
        Values of Moran's Local Autocorrelation Statistic
    gdf : geopandas dataframe
        The Dataframe containing information to plot the two maps.
    attribute : str
        Column name of attribute which should be depicted in Choropleth map.
    p : float, optional
        The p-value threshold for significance. Points and polygons will
        be colored by significance. Default = 0.05.
    region_column: string, optional
        Column name containing mask region of interest. Default = None
    mask: str, float, int, optional
        Identifier or name of the region to highlight. Default = None
        Use the same dtype to specifiy as in original dataset.
    mask_color: str, optional
        Color of mask. Default = '#636363'
    quadrant : int, optional
        Quadrant 1-4 in scatterplot masking values in LISA cluster and
        Choropleth maps. Default = None
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    figsize: tuple, optional
        W, h of figure. Default = (15,4)
    legend: boolean, optional
        If True, legend for maps will be depicted. Default = True
    scheme: str, optional
        Name of PySAL classifier to be used. Default = 'Quantiles'
    cmap: str, optional
        Name of matplotlib colormap used for plotting the Choropleth.
        Default = 'YlGnBu'
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline
        in the scatterplot. Default =None.

    Returns
    -------
    fig : Matplotlib figure instance
        Moran Scatterplot, LISA cluster map and Choropleth.
    axs : list of Matplotlib axes
        Lisat of Matplotlib axes plotted.

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> import geopandas as gpd
    >>> from esda.moran import Moran_Local
    >>> from splot.esda import plot_local_autocorrelation

    Data preparation and analysis

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> moran_loc = Moran_Local(y, w)

    Plotting with quadrant mask and region mask

    >>> fig = plot_local_autocorrelation(moran_loc, gdf, 'Donatns', p=0.05,
    ...                                  region_column='Dprtmnt',
    ...                                  mask=['Ain'], quadrant=1)
    >>> plt.show()

    """
    fig, axs = plt.subplots(
        1, 3, figsize=figsize, subplot_kw={"aspect": "equal", "adjustable": "datalim"}
    )
    # Moran Scatterplot
    moran_scatterplot(
        moran_loc, p=p, ax=axs[0], scatter_kwds=scatter_kwds, fitline_kwds=fitline_kwds
    )
    if aspect_equal is True:
        axs[0].set_aspect("equal", "datalim")
    else:
        axs[0].set_aspect("auto")

    # Lisa cluster map
    # TODO: Fix legend_kwds: display boxes instead of points
    lisa_cluster(
        moran_loc,
        gdf,
        p=p,
        ax=axs[1],
        legend=legend,
        legend_kwds={"loc": "upper left", "bbox_to_anchor": (0.92, 1.05)},
    )
    axs[1].set_aspect("equal")

    # Choropleth for attribute
    gdf.plot(
        column=attribute,
        scheme=scheme,
        cmap=cmap,
        legend=legend,
        legend_kwds={"loc": "upper left", "bbox_to_anchor": (0.92, 1.05)},
        ax=axs[2],
        alpha=1,
    )
    axs[2].set_axis_off()
    axs[2].set_aspect("equal")

    # MASKING QUADRANT VALUES
    if quadrant is not None:
        # Quadrant masking in Scatterplot
        mask_angles = {1: 0, 2: 90, 3: 180, 4: 270}  # rectangle angles
        # We don't want to change the axis data limits, so use the current ones
        xmin, xmax = axs[0].get_xlim()
        ymin, ymax = axs[0].get_ylim()
        # We are rotating, so we start from 0 degrees and
        # figured out the right dimensions for the rectangles for other angles
        mask_width = {1: abs(xmax), 2: abs(ymax), 3: abs(xmin), 4: abs(ymin)}
        mask_height = {1: abs(ymax), 2: abs(xmin), 3: abs(ymin), 4: abs(xmax)}
        axs[0].add_patch(
            patches.Rectangle(
                (0, 0),
                width=mask_width[quadrant],
                height=mask_height[quadrant],
                angle=mask_angles[quadrant],
                color="#E5E5E5",
                zorder=-1,
                alpha=0.8,
            )
        )
        # quadrant selection in maps
        non_quadrant = ~(moran_loc.q == quadrant)
        mask_quadrant = gdf[non_quadrant]
        df_quadrant = gdf.iloc[~non_quadrant]
        union2 = df_quadrant.unary_union.boundary

        # LISA Cluster mask and cluster boundary
        with warnings.catch_warnings():  # temorarily surpress geopandas warning
            warnings.filterwarnings("ignore", category=UserWarning)
            mask_quadrant.plot(
                column=attribute,
                scheme=scheme,
                color="white",
                ax=axs[1],
                alpha=0.7,
                zorder=1,
            )
        gpd.GeoSeries([union2]).plot(linewidth=1, ax=axs[1], color="#E5E5E5")

        # CHOROPLETH MASK
        with warnings.catch_warnings():  # temorarily surpress geopandas warning
            warnings.filterwarnings("ignore", category=UserWarning)
            mask_quadrant.plot(
                column=attribute,
                scheme=scheme,
                color="white",
                ax=axs[2],
                alpha=0.7,
                zorder=1,
            )
        gpd.GeoSeries([union2]).plot(linewidth=1, ax=axs[2], color="#E5E5E5")

    # REGION MASKING
    if region_column is not None:
        # masking inside axs[0] or Moran Scatterplot
        # enforce the same dtype of list and mask
        if not isinstance(mask[0], type(gdf[region_column].iloc[0])):
            warnings.warn(
                "Values in `mask` are not the same dtype as"
                + " values in `region_column`. Converting `mask` values"
                + " to dtype of first observation in region_column."
            )
            data_type = type(gdf[region_column][0].item())
            mask = list(map(data_type, mask))

        ix = gdf[region_column].isin(mask)

        if not ix.any():
            raise ValueError(
                "Specified values {} in `mask` not in `region_column`".format(mask)
            )

        df_mask = gdf[ix]
        x_mask = moran_loc.z[ix]
        y_mask = lag_spatial(moran_loc.w, moran_loc.z)[ix]
        axs[0].plot(
            x_mask,
            y_mask,
            color=mask_color,
            marker="o",
            markersize=14,
            alpha=0.8,
            linestyle="None",
            zorder=-1,
        )

        # masking inside axs[1] or Lisa cluster map
        union = df_mask.unary_union.boundary
        gpd.GeoSeries([union]).plot(linewidth=2, ax=axs[1], color=mask_color)

        # masking inside axs[2] or Chloropleth
        gpd.GeoSeries([union]).plot(linewidth=2, ax=axs[2], color=mask_color)
    return fig, axs


def _moran_loc_bv_scatterplot(
    moran_loc_bv,
    p=None,
    aspect_equal=True,
    ax=None,
    scatter_kwds=None,
    fitline_kwds=None,
):
    """
    Moran Bivariate Scatterplot with option of coloring of Local Moran Statistics

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local_BV instance
        Values of Moran's I Local Autocorrelation Statistics
    p : float, optional
        If given, the p-value threshold for significance. Points will
        be colored by significance. By default it will not be colored.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate Moran Local scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> import geopandas as gpd
    >>> from libpysal.weights.contiguity import Queen
    >>> from libpysal import examples
    >>> from esda.moran import Moran_Local_BV
    >>> from splot.esda import moran_scatterplot

    Load data and calculate Moran Local statistics

    >>> guerry = examples.load_example('Guerry')
    >>> link_to_data = guerry.get_path('guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> m = Moran_Local_BV(x, y, w)

    Plot

    >>> moran_scatterplot(m)
    >>> plt.show()

    Customize plot

    >>> moran_scatterplot(m, p=0.05,
    ...                          fitline_kwds=dict(color='#4393c3')))
    >>> plt.show()

    """
    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()

    if p is not None:
        if not isinstance(moran_loc_bv, Moran_Local_BV):
            raise ValueError(
                "`moran_loc_bv` is not a\n" + "esda.moran.Moran_Local_BV instance"
            )
        if "color" in scatter_kwds or "c" in scatter_kwds or "cmap" in scatter_kwds:
            warnings.warn(
                "To change the color use cmap with a colormap of 5,\n"
                + "c defines the LISA category, color will interfere with c"
            )

        # colors
        spots_bv = moran_hot_cold_spots(moran_loc_bv, p)
        hmap = colors.ListedColormap(
            ["#bababa", "#d7191c", "#abd9e9", "#2c7bb6", "#fdae61"]
        )

    # define customization
    scatter_kwds.setdefault("alpha", 0.6)
    scatter_kwds.setdefault("s", 40)
    fitline_kwds.setdefault("alpha", 0.9)

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize=(7, 7), aspect_equal=aspect_equal)

    # set labels
    ax.set_xlabel("Attribute")
    ax.set_ylabel("Spatial Lag")
    ax.set_title("Moran BV Local Scatterplot")

    # plot and set standards
    lag = lag_spatial(moran_loc_bv.w, moran_loc_bv.zy)
    fit = OLS(lag[:, None], moran_loc_bv.zx[:, None])
    # v- and hlines
    ax.axvline(0, alpha=0.5, color="k", linestyle="--")
    ax.axhline(0, alpha=0.5, color="k", linestyle="--")
    if p is not None:
        fitline_kwds.setdefault("color", "k")
        scatter_kwds.setdefault("cmap", hmap)
        scatter_kwds.setdefault("c", spots_bv)
        ax.plot(moran_loc_bv.zx, fit.predy, **fitline_kwds)
        ax.scatter(moran_loc_bv.zx, lag, **scatter_kwds)
    else:
        scatter_kwds.setdefault("color", splot_colors["moran_base"])
        fitline_kwds.setdefault("color", splot_colors["moran_fit"])
        ax.plot(moran_loc_bv.zx, fit.predy, **fitline_kwds)
        ax.scatter(moran_loc_bv.zx, lag, **scatter_kwds)
    return fig, ax


def moran_facet(
    moran_matrix,
    figsize=(16, 12),
    scatter_bv_kwds=None,
    fitline_bv_kwds=None,
    scatter_glob_kwds=dict(color="#737373"),
    fitline_glob_kwds=None,
):
    """
    Moran Facet visualization.
    Includes BV Morans and Global Morans on the diagonal.

    Parameters
    ----------
    moran_matrix : esda.moran.Moran_BV_matrix instance
        Dictionary of Moran_BV objects
    figsize : tuple, optional
        W, h of figure. Default =(16,12)
    scatter_bv_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points of
        off-diagonal Moran_BV plots.
        Default =None.
    fitline_bv_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline of
        off-diagonal Moran_BV plots.
        Default =None.
    scatter_glob_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points of
        diagonal Moran plots.
        Default =None.
    fitline_glob_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline of
        diagonal Moran plots.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate Moran Local scatterplot figure
    axarr : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports

    >>> import matplotlib.pyplot as plt
    >>> import libpysal as lp
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> from esda.moran import Moran_BV_matrix
    >>> from splot.esda import moran_facet

    Load data and calculate Moran Local statistics

    >>> f = gpd.read_file(lp.examples.get_path("sids2.dbf"))
    >>> varnames = ['SIDR74',  'SIDR79',  'NWR74',  'NWR79']
    >>> vars = [numpy.array(f[var]) for var in varnames]
    >>> w = lp.io.open(lp.examples.get_path("sids2.gal")).read()
    >>> moran_matrix = Moran_BV_matrix(vars,  w,  varnames = varnames)

    Plot

    >>> fig, axarr = moran_facet(moran_matrix)
    >>> plt.show()

    Customize plot

    >>> fig, axarr = moran_facet(moran_matrix,
    ...                          fitline_bv_kwds=dict(color='#4393c3'))
    >>> plt.show()

    """
    nrows = int(numpy.sqrt(len(moran_matrix))) + 1
    ncols = nrows

    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, sharex=True)
    fig.suptitle("Moran Facet")

    for row in range(nrows):
        for col in range(ncols):
            if row == col:
                global_m = Moran(
                    moran_matrix[row, (row + 1) % 4].zy,
                    moran_matrix[row, (row + 1) % 4].w,
                )
                _moran_global_scatterplot(
                    global_m,
                    ax=axarr[row, col],
                    scatter_kwds=scatter_glob_kwds,
                    fitline_kwds=fitline_glob_kwds,
                )
                axarr[row, col].set_facecolor("#d9d9d9")
            else:
                _moran_bv_scatterplot(
                    moran_matrix[row, col],
                    ax=axarr[row, col],
                    scatter_kwds=scatter_bv_kwds,
                    fitline_kwds=fitline_bv_kwds,
                )

            axarr[row, col].spines["bottom"].set_visible(False)
            axarr[row, col].spines["left"].set_visible(False)
            if row == nrows - 1:
                axarr[row, col].set_xlabel(
                    str(moran_matrix[(col + 1) % 4, col].varnames["x"]).format(col)
                )
                axarr[row, col].spines["bottom"].set_visible(True)
            else:
                axarr[row, col].set_xlabel("")

            if col == 0:
                axarr[row, col].set_ylabel(
                    (
                        "Spatial Lag of "
                        + str(moran_matrix[row, (row + 1) % 4].varnames["y"])
                    ).format(row)
                )
                axarr[row, col].spines["left"].set_visible(True)
            else:
                axarr[row, col].set_ylabel("")

            axarr[row, col].set_title("")
    plt.tight_layout()
    return fig, axarr


import mapclassify as classify
import matplotlib
import matplotlib as mpl
import numpy as np
from packaging.version import Version

# isolate MPL version - GH#162
MPL_36 = Version(matplotlib.__version__) >= Version("3.6")
if MPL_36:
    from matplotlib import colormaps as cm
else:
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt


"""
Utility functions for lightweight visualizations in splot
"""

__author__ = "Stefanie Lumnitz <stefanie.lumitz@gmail.com>"


def moran_hot_cold_spots(moran_loc, p=0.05):
    sig = 1 * (moran_loc.p_sim < p)
    HH = 1 * (sig * moran_loc.q == 1)
    LL = 3 * (sig * moran_loc.q == 3)
    LH = 2 * (sig * moran_loc.q == 2)
    HL = 4 * (sig * moran_loc.q == 4)
    cluster = HH + LL + LH + HL
    return cluster


def mask_local_auto(moran_loc, p=0.5):
    """
    Create Mask for coloration and labeling of local spatial autocorrelation

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local instance
        values of Moran's I Global Autocorrelation Statistic
    p : float
        The p-value threshold for significance. Points will
        be colored by significance.

    Returns
    -------
    cluster_labels : list of str
        List of labels - ['ns', 'HH', 'LH', 'LL', 'HL']
    colors5 : list of str
        List of colours - ['#d7191c', '#fdae61', '#abd9e9',
        '#2c7bb6', 'lightgrey']
    colors : array of str
        Array containing coloration for each input value/ shape.
    labels : list of str
        List of label for each attribute value/ polygon.
    """
    # create a mask for local spatial autocorrelation
    cluster = moran_hot_cold_spots(moran_loc, p)

    cluster_labels = ["ns", "HH", "LH", "LL", "HL"]
    labels = [cluster_labels[i] for i in cluster]

    colors5 = {0: "lightgrey", 1: "#d7191c", 2: "#abd9e9", 3: "#2c7bb6", 4: "#fdae61"}
    colors = [colors5[i] for i in cluster]  # for Bokeh
    # for MPL, keeps colors even if clusters are missing:
    x = np.array(labels)
    y = np.unique(x)
    colors5_mpl = {
        "HH": "#d7191c",
        "LH": "#abd9e9",
        "LL": "#2c7bb6",
        "HL": "#fdae61",
        "ns": "lightgrey",
    }
    colors5 = [colors5_mpl[i] for i in y]  # for mpl

    # HACK need this, because MPL sorts these labels while Bokeh does not
    cluster_labels.sort()
    return cluster_labels, colors5, colors, labels


_classifiers = {
    "box_plot": classify.BoxPlot,
    "equal_interval": classify.EqualInterval,
    "fisher_jenks": classify.FisherJenks,
    "headtail_breaks": classify.HeadTailBreaks,
    "jenks_caspall": classify.JenksCaspall,
    "jenks_caspall_forced": classify.JenksCaspallForced,
    "max_p_classifier": classify.MaxP,
    "maximum_breaks": classify.MaximumBreaks,
    "natural_breaks": classify.NaturalBreaks,
    "quantiles": classify.Quantiles,
    "percentiles": classify.Percentiles,
    "std_mean": classify.StdMean,
    "user_defined": classify.UserDefined,
}


def bin_values_choropleth(attribute_values, method="quantiles", k=5):
    """
    Create bins based on different classification methods.
    Needed for legend labels and Choropleth coloring.

    Parameters
    ----------
    attribute_values : array or geopandas.series instance
        Array containing relevant attribute values.
    method : str
        Classification method to be used. Options supported:
        * 'quantiles' (default)
        * 'fisher-jenks'
        * 'equal-interval'
    k : int
        Number of bins, assigning values to. Default k=5

    Returns
    -------
    bin_values : mapclassify instance
        Object containing bin ids for each observation (.yb),
        upper bounds of each class (.bins), number of classes (.k)
        and number of onservations falling in each class (.counts)
    """
    if method not in ["quantiles", "fisher_jenks", "equal_interval"]:
        raise ValueError("Method {} not supported".format(method))

    bin_values = _classifiers[method](attribute_values, k)
    return bin_values


def bin_labels_choropleth(gdf, attribute_values, method="quantiles", k=5):
    """
    Create labels for each bin in the legend

    Parameters
    ----------
    gdf : Geopandas dataframe
        Dataframe containign relevant shapes and attribute values.
    attribute_values : array or geopandas.series instance
        Array containing relevant attribute values.
    method : str, optional
        Classification method to be used. Options supported:
        * 'quantiles' (default)
        * 'fisher-jenks'
        * 'equal-interval'
    k : int, optional
        Number of bins, assigning values to. Default k=5

    Returns
    -------
    bin_labels : list of str
        List of label for each bin.
    """
    # Retrieve bin values from bin_values_choropleth()
    bin_values = bin_values_choropleth(attribute_values, method=method, k=k)

    # Extract bin ids (.yb) and upper bounds for each class (.bins)
    yb = bin_values.yb
    bins = bin_values.bins

    # Create bin labels (smaller version)
    bin_edges = bins.tolist()
    bin_labels = []
    for i in range(k):
        bin_labels.append("<{:1.1f}".format(bin_edges[i]))

    # Add labels (which are the labels printed in the legend) to each row of gdf
    labels = np.array([bin_labels[c] for c in yb])
    gdf["labels_choro"] = [str(l_) for l_ in labels]
    return bin_labels


def add_legend(fig, labels, colors):
    """
    Add a legend to a figure given legend labels & colors.

    Parameters
    ----------
    fig : Bokeh Figure instance
        Figure instance labels should be generated for.
    labels : list of str
        Labels to use as legend entries.
    colors : Bokeh Palette instance
        Palette instance containing colours of choice.
    """
    from bokeh.models import Legend

    # add labels to figure (workaround,
    # legend with geojsondatasource doesn't work,
    # see https://github.com/bokeh/bokeh/issues/5904)
    items = []
    for label, color in zip(labels, colors):
        patch = fig.patches(xs=[], ys=[], fill_color=color)
        items.append((label, [patch]))

    legend = Legend(
        items=items, location="top_left", margin=0, orientation="horizontal"
    )
    # possibility to define glyph_width=10, glyph_height=10)
    legend.label_text_font_size = "8pt"
    fig.add_layout(legend, "below")
    return legend


def format_legend(values):
    """
    Helper to return sensible legend values

    Parameters
    ----------
    values: array
        Values plotted in legend.
    """
    in_thousand = False
    if np.any(values > 1000):
        in_thousand = True
        values = values / 1000
    return values, in_thousand


def calc_data_aspect(plot_height, plot_width, bounds):
    # Deal with data ranges in Bokeh:
    # make a meter in x and a meter in y the same in pixel lengths
    aspect_box = plot_height / plot_width  # 2 / 1 = 2
    xmin, ymin, xmax, ymax = bounds
    x_range = xmax - xmin  # 1 = 1 - 0
    y_range = ymax - ymin  # 3 = 3 - 0
    aspect_data = y_range / x_range  # 3 / 1 = 3
    if aspect_data > aspect_box:
        # we need to increase x_range,
        # such that aspect_data becomes equal to aspect_box
        halfrange = 0.5 * x_range * (aspect_data / aspect_box - 1)
        # 0.5 * 1 * (3 / 2 - 1) = 0.25
        xmin -= halfrange  # 0 - 0.25 = -0.25
        xmax += halfrange  # 1 + 0.25 = 1.25
    else:
        # we need to increase y_range
        halfrange = 0.5 * y_range * (aspect_box / aspect_data - 1)
        ymin -= halfrange
        ymax += halfrange

    # Add a bit of margin to both x and y
    margin = 0.03
    xmin -= (xmax - xmin) / 2 * margin
    xmax += (xmax - xmin) / 2 * margin
    ymin -= (ymax - ymin) / 2 * margin
    ymax += (ymax - ymin) / 2 * margin
    return xmin, xmax, ymin, ymax


# Utility functions for colormaps
# Color design
splot_colors = dict(moran_base="#bababa", moran_fit="#d6604d")


# Utility function #1 - forces continuous diverging colormap to be centered at zero
def shift_colormap(  # noqa E302
    cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"
):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Parameters
    ----------
    cmap : str or matplotlib.cm instance
        colormap to be altered
    start : float, optional
        Offset from lowest point in the colormap's range.
        Should be between 0.0 and `midpoint`.
        Default =0.0 (no lower ofset).
    midpoint : float, optional
        The new center of the colormap.Should be between 0.0 and
        1.0. In general, this should be 1 - vmax/(vmax + abs(vmin)).
        For example if your data range from -15.0 to +5.0 and
        you want the center of the colormap at 0.0, `midpoint`
        should be set to  1 - 5/(5 + 15)) or 0.75.
        Default =0.5 (no shift).
    stop : float, optional
        Offset from highets point in the colormap's range.
        Should be between `midpoint` and 1.0.
        Default =1.0 (no upper ofset).
    name : str, optional
        Name of the new colormap.

    Returns
    -------
    new_cmap : A new colormap that has been shifted.
    """
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    """
    new_cmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=new_cmap)
    return new_cmap
    """

    new_cmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    if MPL_36:
        cm.register(new_cmap)
    else:
        plt.register_cmap(cmap=new_cmap)
    return new_cmap


# Utility #2 - truncate colorcap in order to grab only positive or negative portion
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Function to truncate a colormap by selecting a subset of
    the original colormap's values

    Parameters
    ----------
    cmap : str or matplotlib.cm instance
        Colormap to be altered
    minval : float, optional
        Minimum value of the original colormap to include
        in the truncated colormap. Default =0.0.
    maxval : Maximum value of the original colormap to
        include in the truncated colormap. Default =1.0.
    n : int, optional
        Number of intervals between the min and max values
        for the gradient of the truncated colormap. Default =100.

    Returns
    -------
    new_cmap : A new colormap that has been shifted.
    """

    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
