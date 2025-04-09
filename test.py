def moran_scatterplot(
    moran,
    zstandard=True,
    p=None,
    aspect_equal=True,
    ax=None,
    scatter_kwds=None,
    fitline_kwds=None,
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
