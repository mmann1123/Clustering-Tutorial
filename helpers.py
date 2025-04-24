# %%
import pandas as pd
import libpysal
import seaborn
from splot import esda as esdaplot
import esda
import matplotlib.pyplot as plt
import pandas as pd
from splot.esda import moran_scatterplot
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy.spatial import distance_matrix
from scipy.special import comb
from geopandas import GeoDataFrame
from geopy.distance import great_circle
from concurrent.futures import ThreadPoolExecutor
from esda.moran import Moran, Moran_BV, Moran_Local, Moran_Local_BV


# %%


def calculate_moran_i_report(data, variable, weights, alpha=0.05):
    """
    Calculate Moran's I statistic and provide an interpretation report.

    Parameters:
    ----------
    data : GeoDataFrame
        The spatial data containing the variable to analyze
    variable : str
        The name of the variable to calculate Moran's I for
    weights : libpysal.weights
        The spatial weights matrix
    alpha : float, optional
        The significance level (default 0.05)

    Returns:
    -------
    mi : esda.Moran
        The Moran's I result object
    """
    # Calculate Moran's I
    mi = esda.Moran(data[variable], weights)

    # Prepare interpretation based on I value
    if mi.I > 0.7:
        pattern = "strong positive spatial autocorrelation (very clustered)"
    elif mi.I > 0.3:
        pattern = "moderate positive spatial autocorrelation (clustered)"
    elif mi.I > 0.1:
        pattern = "weak positive spatial autocorrelation (slightly clustered)"
    elif mi.I > -0.1:
        pattern = "no substantial spatial autocorrelation (random pattern)"
    elif mi.I > -0.3:
        pattern = "weak negative spatial autocorrelation (slightly dispersed)"
    elif mi.I > -0.7:
        pattern = "moderate negative spatial autocorrelation (dispersed)"
    else:
        pattern = "strong negative spatial autocorrelation (very dispersed)"

    # Determine significance
    if mi.p_norm < alpha:
        significance = f"statistically significant (p={mi.p_norm:.4f})"
    else:
        significance = f"not statistically significant (p={mi.p_norm:.4f})"

    # Print report
    print(f"Moran's I Analysis for {variable}:")
    print(f"I value: {mi.I:.4f}")
    print(f"p-value: {mi.p_norm:.4f}")
    print(f"Interpretation: The data shows {pattern} and is {significance}.")
    print("-------------------------------------------------------")

    return mi


def plot_moran_scatter(df, x_name, y_name=None, w=None, title=None):
    """
    Create a Moran scatterplot of normalized data with labels in each quadrant.

    Parameters:
    -----------
    df : GeoDataFrame
        GeoDataFrame containing the data
    x_name : str
        Column name for Moran's plot values
    y_name : str, optional
        If not None, triggers Moran_Local_BV
    w : libpysal.weights
        The spatial weights matrix
    title : str, optional
        Plot title

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The plot axes

    Example:
    --------
    plot_moran_scatter(
        df=common_loans,
        x_name="loan_income_ratio_GSE",
        w=w,
        title="GSE loans"
    )
    """

    if y_name is None:
        # Calculate Moran's I
        moran = Moran_Local(df[x_name], w)
    else:
        # Calculate Bivariate Moran's I
        moran = Moran_Local_BV(df[y_name], df[x_name], w)

    # Use moran_scatterplot_mm for enhanced visualization
    fig, ax = moran_scatterplot(
        moran=moran,
        p=0.05,  # Statistical significance threshold
        zstandard=True,
        aspect_equal=True,
    )

    # Set title if provided
    if title:
        ax.set_title(title)

    fig.tight_layout()
    plt.show()

    # Print interpretation of quadrant labels
    print(
        f"Interpretation of Quadrant Labels:\n"
        f"HH: High-High (positive correlation)\n"
        f"HL: High-Low (negative correlation)\n"
        f"LH: Low-High (negative correlation)\n"
        f"LL: Low-Low (positive correlation)\n"
        f"-------------------------------------------------------\n"
        f"Where values are in XY order - so HL would be high in x-axis, low in y-axis\n"
        f"-------------------------------------------------------"
    )

    return ax


# %%


def calculate_moran_local(gdf, x_name, w, significance_level=0.05):
    """
    Calculate Moran Local (univariate) and return a DataFrame with quadrant, significance, and translated labels.

    Parameters:
    - gdf: GeoDataFrame containing the data.
    - x_name: Column name for the variable.
    - w: Spatial weights matrix.
    - significance_level: Significance threshold for p-values (default is 0.05).

    Returns:
    - GeoDataFrame with additional columns: 'quadrant', 'significance', and 'quadrant_label'.
    """
    # Calculate Moran's Local
    local = Moran_Local(
        gdf[x_name].values,
        w,
        geoda_quads=True,
    )
    # Add quadrant and significance columns
    gdf["quadrant"] = local.q
    gdf["significance"] = local.p_sim
    # Add translated quadrant labels
    gdf["quadrant_label"] = gdf.apply(
        lambda row: (
            "HH"
            if row["significance"] <= significance_level and row["quadrant"] == 1
            else (
                "LH"
                if row["significance"] <= significance_level and row["quadrant"] == 2
                else (
                    "LL"
                    if row["significance"] <= significance_level
                    and row["quadrant"] == 3
                    else (
                        "HL"
                        if row["significance"] <= significance_level
                        and row["quadrant"] == 4
                        else "NS"
                    )
                )
            )
        ),
        axis=1,
    )
    return gdf


def calculate_moran_local_bv(gdf, x_name, y_name, w, significance_level=0.05):
    """
    Calculate Moran Local Bivariate and return a DataFrame with quadrant, significance, and translated labels.

    Parameters:
    - gdf: GeoDataFrame containing the data.
    - x_name: Column name for the first variable.
    - y_name: Column name for the second variable.
    - w: Spatial weights matrix.
    - significance_level: Significance threshold for p-values (default is 0.05).

    Returns:
    - GeoDataFrame with additional columns: 'quadrant', 'significance', and 'quadrant_label'.


    """
    bv = Moran_Local_BV(
        gdf[x_name].values,
        gdf[y_name].values,
        w,
        geoda_quads=True,
    )
    gdf["quadrant"] = bv.q
    gdf["significance"] = bv.p_sim
    gdf["quadrant_label"] = gdf.apply(
        lambda row: (
            "HH"
            if row["significance"] <= significance_level and row["quadrant"] == 1
            else (
                "LH"
                if row["significance"] <= significance_level and row["quadrant"] == 2
                else (
                    "LL"
                    if row["significance"] <= significance_level
                    and row["quadrant"] == 3
                    else (
                        "HL"
                        if row["significance"] <= significance_level
                        and row["quadrant"] == 4
                        else "NS"
                    )
                )
            )
        ),
        axis=1,
    )
    return gdf


def plot_lisa_analysis(
    df,
    x_name,
    y_name=None,
    w=None,
    title_prefix="",
    legend_kwds={"fmt": "{:.4f}"},  # Show 4 decimal places
):
    """
    Performs Moran's I LISA analysis and creates a 4-panel visualization

    Parameters:
    -----------
    df : GeoDataFrame
        The spatial dataframe containing the data
    x_name : str
        The name of the variable to analyze
    y_name : str
        If not None, triggers Moran_Local_BV
    w : W
        Spatial weights matrix
    title_prefix : str, optional
        Prefix for the plot titles
    legend_kwds : dict
        Set legend arg

    Example:
    --------
    # Single local moran's
    plot_lisa_analysis(df, "loan_income_ratio_GSE", w, "Sold to GSE")

    # Bivariate local morans
    plot_lisa_analysis(df=common_loans,
                   x_name="loan_income_ratio_GSE",
                   y_name="loan_income_ratio_not_sold",
                   w=w,
                   title_prefix="Sold to GSE")
    """

    def _local_variable_plot(df, variable, axs, legend_kwds, axis=0):
        # Subplot 1 - Choropleth of local statistics
        ax = axs[axis]
        df.plot(
            column=variable,
            # cmap="plasma",
            scheme="NaturalBreaks",
            k=5,
            edgecolor="white",
            linewidth=0.1,
            alpha=0.75,
            legend=True,
            ax=ax,
            legend_kwds={
                **legend_kwds,
                "loc": "lower center",
                "bbox_to_anchor": (0.5, -0.2),
                "ncol": 2,
            },
        )

    def _local_stats_plot(df, lisa, axs, axis=0):
        # Subplot 1 - Choropleth of local statistics
        ax = axs[axis]
        df.assign(Is=lisa.Is).plot(
            column="Is",
            # cmap="plasma",
            scheme="NaturalBreaks",
            k=5,
            edgecolor="white",
            linewidth=0.1,
            alpha=0.75,
            legend=True,
            ax=ax,
        )

    def _moran_global_scatterplot(df, lisa, axs, axis=1):
        ax = axs[axis]
        moran_scatterplot(lisa, p=0.05, ax=ax)
        # Get the current axis
        ax = plt.gca()
        # Set the x-axis and y-axis labels for moran's scatter
        ax.set_xlabel("Variable (standardized)", fontsize=12)
        ax.set_ylabel("Spatial Lag of Variable (standardized)", fontsize=12)

    def _quadrant_cat_plot(df, lisa, axs, axis=1):
        # Subplot 2 - Quadrant categories
        ax = axs[axis]
        esdaplot.lisa_cluster(
            lisa,
            df,
            p=1,
            ax=ax,
            legend_kwds={
                "loc": "lower center",
                "bbox_to_anchor": (0.5, -0.2),
                "ncol": 2,
            },
        )

    def _significance_map(df, lisa, axs, axis=2):

        # Subplot 3 - Significance map
        ax = axs[axis]
        labels = pd.Series(1 * (lisa.p_sim < 0.05), index=df.index).map(
            {1: "Significant", 0: "Non-Significant"}
        )

        df.assign(cl=labels).plot(
            column="cl",
            categorical=True,
            k=2,
            cmap="Paired",
            linewidth=0.1,
            edgecolor="white",
            legend=True,
            ax=ax,
            legend_kwds={
                "loc": "lower center",
                "bbox_to_anchor": (0.5, -0.2),
                "ncol": 2,
            },
        )

    def _cluster_map(df, lisa, axs, axis=3):
        # Subplot 4 - Cluster map
        ax = axs[axis]
        esdaplot.lisa_cluster(
            lisa,
            df,
            p=0.05,
            ax=ax,
            legend_kwds={
                "loc": "lower center",
                "bbox_to_anchor": (0.5, -0.2),
                "ncol": 2,
            },
        )

    # Set up figure and axes
    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    # Make the axes accessible with single indexing
    axs = axs.flatten()

    if y_name is None:
        f.suptitle(f"{title_prefix} Moran's I LISA Analysis", fontsize=16, y=0.95)

        # Calculate local Moran's I
        lisa = esda.moran.Moran_Local(df[x_name], w)

        _local_variable_plot(df, x_name, axs, legend_kwds=legend_kwds, axis=0)
        _moran_global_scatterplot(df, lisa, axs, axis=1)
        _significance_map(df, lisa, axs, axis=2)
        _cluster_map(df, lisa, axs, axis=3)

        # Figure styling
        titles = [
            f"{title_prefix} {x_name} Values",
            f"{title_prefix} Scatterplot Quadrant",
            f"{title_prefix} Statistical Significance",
            f"{title_prefix} Moran Cluster Map",
        ]

        # Create interpretation text
        lisa_interpretation = (
            f"\nMoran's LISA Analysis interpretation:\n"
            f"1. Values Map: Shows the spatial distribution of {x_name} values.\n"
            f"2. Quadrant Map: Categorizes each location based on its {x_name} value and its neighbors:\n"
            f"   - High-High (HH): High values surrounded by high values\n"
            f"   - High-Low (HL): High values surrounded by low values\n"
            f"   - Low-High (LH): Low values surrounded by high values\n"
            f"   - Low-Low (LL): Low values surrounded by low values\n"
            f"3. Scatter Plot: Shows quadrants and statistical significance\n"
            f"4. Cluster Map: Combines quadrant types with statistical significance,\n"
            f"   highlighting only the statistically significant spatial clusters.\n"
            f"-------------------------------------------------------"
        )
        for i, ax in enumerate(axs.flatten()):
            if i != 1:
                ax.set_axis_off()
            ax.set_title(titles[i], y=0)

        # Print the interpretation

    else:
        f.suptitle(
            f"{title_prefix} Moran's I Bivariate LISA Analysis", fontsize=16, y=0.95
        )

        lisa = esda.moran.Moran_Local_BV(df[x_name], df[y_name], w)

        _local_variable_plot(df, x_name, axs, legend_kwds=legend_kwds, axis=0)
        _local_variable_plot(df, y_name, axs, legend_kwds=legend_kwds, axis=1)
        _moran_global_scatterplot(df, lisa, axs, axis=2)
        _cluster_map(df, lisa, axs, axis=3)

        titles = [
            f"{title_prefix} {x_name} Values",
            f"{title_prefix} {y_name} Values",
            f"{title_prefix} Statistical Significance",
            f"{title_prefix} Moran Cluster Map",
        ]
        # Create interpretation text for bivariate analysis
        lisa_interpretation = (
            f"\nBivariate Moran's LISA Analysis interpretation:\n"
            f"1. {x_name} Values Map: Shows the spatial distribution of first variable.\n"
            f"2. {y_name} Values Map: Shows the spatial distribution of second variable.\n"
            f"3. Scatter Plot: Shows quadrants and statistical significance\n"
            f"   bivariate spatial relationships (p < 0.05).\n"
            f"4. Cluster Map: Shows significant spatial clusters with the following patterns:\n"
            f"   - High-High (HH): High {x_name} values surrounded by high {y_name} values\n"
            f"   - High-Low (HL): High {x_name} values surrounded by low {y_name} values\n"
            f"   - Low-High (LH): Low {x_name} values surrounded by high {y_name} values\n"
            f"   - Low-Low (LL): Low {x_name} values surrounded by low {y_name} values\n"
            f"-------------------------------------------------------"
        )
        for i, ax in enumerate(axs.flatten()):
            if i != 2:
                ax.set_axis_off()
            ax.set_title(titles[i], y=0)

    f.tight_layout()
    plt.show()

    print(lisa_interpretation)

    return lisa


import geopandas as gpd
from esda.moran import Moran_Local_BV


def map_local_morans(
    gdf, x_name, y_name=None, w=None, color_mapping=None, basemap=None, **map_kwargs
):
    """
    Runs Local Moran's I Bivariate analysis and visualizes the results using explore.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the data.
        x_name (str): The name of the first variable for Moran's I analysis.
        y_name (str): The name of the second variable for Moran's I analysis.
        w (libpysal.weights): The spatial weights matrix.
        color_mapping (dict, optional): A dictionary mapping quadrant labels to colors.
        basemap (str, optional): A basemap to use in the visualization.
        **map_kwargs: Additional keyword arguments for the `explore` function.

    Returns:
        folium.Map: An interactive map visualizing the Local Moran's I results.
    """
    # Default color mapping if none is provided
    if color_mapping is None:
        color_mapping = {
            "HH": "red",
            "HL": "orange",
            "LH": "lightblue",
            "LL": "blue",
            "NS": "lightgrey",
        }

    # Calculate Local Moran's I Bivariate
    if y_name is None:
        moran = calculate_moran_local(gdf=gdf, x_name=x_name, w=w)
    else:
        moran = calculate_moran_local_bv(
            gdf=gdf,
            x_name=x_name,
            y_name=y_name,
            w=w,
        )

    # Map colors to the quadrant labels
    gdf["color"] = gdf["quadrant_label"].map(color_mapping)

    # Visualize using explore
    return gdf.explore(
        column="quadrant_label",
        cmap=list(color_mapping.values()),  # Ensure the colors match the legend order
        tooltip=False,
        popup=True,
        legend=True,  # Enable legend
        legend_kwds={"colorbar": False, "caption": "Quadrant"},
        basemap=basemap,
        **map_kwargs,
    )


# %%

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy.spatial import distance_matrix
from shapely.geometry import Point
from geopandas import GeoDataFrame


def scale_data(data, method):
    if method == "standardize":
        return StandardScaler().fit_transform(data)
    elif method == "demean":
        return data - np.mean(data, axis=0)
    elif method == "mad":
        return RobustScaler().fit_transform(data)
    elif method == "range_standardize":
        return MinMaxScaler().fit_transform(data)
    elif method == "range_adjust":
        return (data - np.min(data, axis=0)) / (
            np.max(data, axis=0) - np.min(data, axis=0)
        )
    else:  # 'raw'
        return data


def neighbor_match_test(
    df: GeoDataFrame,
    k: int,
    scale_method="standardize",
    distance_method="euclidean",
    power=1.0,
    is_inverse=False,
):
    if not isinstance(df, GeoDataFrame):
        raise TypeError("The input data needs to be a GeoDataFrame.")

    n_vars = len(df.columns) - 1  # Exclude geometry column
    data = df.iloc[:, :-1].values
    scaled_data = scale_data(data, scale_method)

    if distance_method == "manhattan":
        dists = distance_matrix(scaled_data, scaled_data, p=1)
    else:  # 'euclidean'
        dists = distance_matrix(scaled_data, scaled_data)

    if is_inverse:
        dists = np.power(dists, -power)
    else:
        dists = np.power(dists, power)

    results = []
    for i, point in enumerate(df.geometry):
        distances = dists[i]
        neighbors_idx = np.argsort(distances)[1 : k + 1]  # Skip itself (index 0)
        neighbors = df.iloc[neighbors_idx]
        results.append([len(neighbors), np.mean(distances[neighbors_idx])])

    results_df = pd.DataFrame(results, columns=["Cardinality", "Probability"])

    # Convert -1 probabilities to NaN as in the R function
    results_df.loc[results_df["Probability"] == -1, "Probability"] = np.nan

    return results_df


# %%


def scale_data(data, method):
    if method == "standardize":
        return StandardScaler().fit_transform(data)
    elif method == "demean":
        return data - np.mean(data, axis=0)
    elif method == "mad":
        return RobustScaler().fit_transform(data)
    elif method == "minmax":
        return MinMaxScaler().fit_transform(data)
    elif method == "range_adjust":
        return (data - np.min(data, axis=0)) / (
            np.max(data, axis=0) - np.min(data, axis=0)
        )
    else:  # 'raw'
        return data


def compute_knn(data, k):
    dists = distance_matrix(data, data)
    neighbors_idx = np.argsort(dists, axis=1)[:, 1 : k + 1]  # Skip itself (index 0)
    return neighbors_idx


def neighbor_match_test(
    df: GeoDataFrame,
    k: int,
    scale_method="standardize",
):
    """
    Perform a neighbor matching test to identify spatial clustering.

    This function compares the k-nearest neighbors in attribute space with
    the k-nearest neighbors in geographical space, then computes the statistical
    significance of the overlap.

    Parameters
    ----------
    df : GeoDataFrame
        Input geodataframe containing both geometries and attributes to analyze.
    k : int
        Number of nearest neighbors to consider.
    scale_method : str, default="standardize"
        Method used to scale attribute data. Options include:
        - "standardize": Z-score standardization
        - "demean": Subtract the mean
        - "mad": Robust scaling using median absolute deviation
        - "minmax": Min-max normalization to [0,1]
        - "range_adjust": Min-max normalization preserving original range
        - "raw": No scaling

    Returns
    -------
    GeoDataFrame
        Original dataframe with two additional columns:
        - Cardinality: Number of neighbors that are common in both attribute and geographic space
        - Probability: P-value indicating the statistical significance of the overlap

    Notes
    -----
    For geographic coordinates (EPSG:4326), great-circle distance is used.
    For projected coordinates, Euclidean distance is used.

    The probability is calculated using a combinatorial approach based on
    the hypergeometric distribution, indicating the likelihood of obtaining
    the observed cardinality by random chance.

    """

    if not isinstance(df, GeoDataFrame):
        raise TypeError(
            "The input data needs to be a GeoDataFrame. Make sure to pass the geometry column"
        )

    geometry_col = df.geometry.name  # Detect the name of the geometry column
    data = df.drop(columns=[geometry_col]).values  # Drop geometry column
    scaled_data = scale_data(data, scale_method)

    # Compute k-nearest neighbors in attribute space
    attribute_neighbors_idx = compute_knn(scaled_data, k)

    # Compute k-nearest neighbors in geographical space
    if df.crs and str(df.crs).upper() == "EPSG:4326":
        # Use great-circle distance for geographic coordinates (lat/lon)

        # Extract lat/lon from geometry centroids
        coords = np.array([[geom.centroid.y, geom.centroid.x] for geom in df.geometry])

        # Create distance matrix using great-circle distance
        n = len(coords)
        geo_dists = np.zeros((n, n))
        # Use parallel processing for faster distance calculation

        def calculate_distances(i):
            i_distances = np.zeros(n)
            for j in range(n):
                if i != j:
                    i_distances[j] = great_circle(coords[i], coords[j]).kilometers
                else:
                    i_distances[j] = float("inf")  # Set self-distance to infinity
            return i, i_distances

        # Use ThreadPoolExecutor since this is an I/O-bound operation
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and collect results
            results = list(executor.map(calculate_distances, range(n)))

        # Assemble the distance matrix from results
        for i, i_distances in results:
            geo_dists[i] = i_distances

        # Get indices of k-nearest neighbors
        geo_neighbors_idx = np.argsort(geo_dists, axis=1)[:, :k]
    else:
        # Use Euclidean distance for projected coordinates
        coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in df.geometry])
        geo_neighbors_idx = compute_knn(coords, k)

    results = []
    n = len(df)
    N = n - 1  # Total possible neighbors

    for i in range(n):
        attribute_neighbors = set(attribute_neighbors_idx[i])
        geo_neighbors = set(geo_neighbors_idx[i])
        common_neighbors = attribute_neighbors & geo_neighbors
        # numbero fo common nearest neighbors in geographic and multidimensional space
        v = len(common_neighbors)

        # Compute probability using combinatorial calculation
        p_value = (comb(k, v) * comb(N - k, k - v)) / comb(N, k)

        results.append([v, p_value])

    results_df = pd.DataFrame(results, columns=["Cardinality", "Probability"])
    results_df.loc[results_df["Probability"] == -1, "Probability"] = np.nan
    results_df = pd.merge(df, results_df, left_index=True, right_index=True)
    # Create a custom color array based on significance
    results_df["alpha"] = 1.0  # default alpha
    results_df.loc[results_df["Probability"] > 0.05, "alpha"] = (
        0.5  # lower alpha for non-significant areas
    )
    # Plot with variable transparency
    # Create the plot with alpha for individual geometries
    ax = results_df.plot(
        "Cardinality",
        categorical=True,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.1,
        alpha=results_df["alpha"]
        .fillna(0)
        .values,  # use the alpha column for transparency
        legend=False,  # Don't create legend yet
    )

    # Remove x and y axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Create legend handles and labels manually
    import matplotlib.patches as mpatches

    # Get unique cardinality values and their colors from the colormap
    unique_values = sorted(results_df["Cardinality"].unique())
    cmap = plt.cm.viridis  # Same colormap as in the plot
    norm = plt.Normalize(min(unique_values), max(unique_values))

    # Create legend patches
    legend_patches = []
    for value in unique_values:
        color = cmap(norm(value))
        patch = mpatches.Patch(color=color, label=str(value), alpha=0.7)
        legend_patches.append(patch)

    # Add the legend to the existing axis
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=3,
        frameon=True,
        title="Number of Common Neighbors",
    )

    ax.set_title(
        f"Spatial Overlap of Geographic & NDim Space \n{df.drop(columns=[geometry_col]).columns.to_list()}"
    )
    return results_df


# %%


# # %%
# # Add this to your script
# import folium
# import json
# import numpy as np
# import io
# import base64
# from matplotlib import pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # Add this to your script
# import folium
# import json
# import numpy as np
# import io
# import base64
# from matplotlib import pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# from IPython.display import display


# def create_folium_moran_map(data, x, weights, output_path="folium_moran_map.html"):
#     """
#     Create a Folium-based interactive Moran map with embedded scatter plot

#     Parameters:
#     -----------
#     data : GeoDataFrame
#         Spatial data containing geometries and values
#     x : str
#         Column name for the variable to analyze
#     weights : libpysal.weights
#         Spatial weights matrix
#     output_path : str, optional
#         Path to save the HTML file

#     Returns:
#     --------
#     m : folium.Map
#         The created folium map
#     """
#     import folium
#     from matplotlib.colors import LinearSegmentedColormap
#     import matplotlib.pyplot as plt
#     import io
#     import base64
#     import numpy as np
#     import pandas as pd
#     import esda

#     # Create a deep copy to avoid modifying the original data
#     df = data.copy()

#     # Handle missing values before standardization
#     df = df.dropna(subset=[x])

#     # Compute spatially lagged variable
#     df[x + "_std"] = df[x] - df[x].mean()
#     y = "w_" + x
#     df[y] = libpysal.weights.lag_spatial(weights, df[x + "_std"])
#     df[y] = df[y] - df[y].mean()

#     # Compute quadrants for LISA map
#     df["quadrant"] = "NA"
#     df.loc[(df[x + "_std"] > 0) & (df[y] > 0), "quadrant"] = "HH"
#     df.loc[(df[x + "_std"] > 0) & (df[y] < 0), "quadrant"] = "HL"
#     df.loc[(df[x + "_std"] < 0) & (df[y] > 0), "quadrant"] = "LH"
#     df.loc[(df[x + "_std"] < 0) & (df[y] < 0), "quadrant"] = "LL"

#     # Use original values for coloring
#     df["value"] = df[x]
#     vmin, vmax = df["value"].min(), df["value"].max()

#     # Convert to WGS84 for web mapping if needed
#     if df.crs and df.crs != "EPSG:4326":
#         df = df.to_crs("EPSG:4326")

#     # Create color map
#     colormap = LinearSegmentedColormap.from_list("RdBu_r", ["blue", "white", "red"])

#     # Create Folium map centered on the US
#     m = folium.Map(location=[39.8, -98.5], zoom_start=4, tiles="CartoDB positron")

#     # Create a Moran scatter plot to embed
#     def create_moran_plot():
#         fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
#         colors = {
#             "HH": "red",
#             "LL": "blue",
#             "HL": "lightcoral",
#             "LH": "lightblue",
#             "NA": "gray",
#         }

#         # Plot each quadrant separately
#         for quad in ["HH", "HL", "LH", "LL"]:
#             subset = df[df["quadrant"] == quad]
#             if not subset.empty:
#                 ax.scatter(
#                     subset[x + "_std"],
#                     subset[y],
#                     c=colors[quad],
#                     alpha=0.7,
#                     label=quad,
#                     edgecolor="k",
#                     s=30,
#                 )

#         # Add reference lines
#         ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
#         ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

#         # Add 45-degree line
#         min_val = min(df[x + "_std"].min(), df[y].min())
#         max_val = max(df[x + "_std"].max(), df[y].max())
#         ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

#         ax.set_xlabel(f"{x} (standardized)")
#         ax.set_ylabel(f"Spatial lag of {x} (standardized)")
#         ax.set_title("Moran's I Scatter Plot")
#         ax.legend(title="Quadrants")

#         plt.tight_layout()

#         # Convert plot to image
#         img = io.BytesIO()
#         plt.savefig(img, format="png", bbox_inches="tight")
#         plt.close()
#         img.seek(0)
#         return base64.b64encode(img.getvalue()).decode("utf-8")

#     # Calculate Moran's I
#     moran = esda.Moran(df[x + "_std"], weights)
#     moran_html = f"""
#     <div style='padding: 10px; background-color: white; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2);'>
#         <h4>Global Moran's I Statistics</h4>
#         <p>I: {moran.I:.4f}</p>
#         <p>p-value: {moran.p_sim:.4f}</p>
#         <hr/>
#         <h4>Moran Scatter Plot</h4>
#         <img src="data:image/png;base64,{create_moran_plot()}" width="300px">
#     </div>
#     """

#     # Add the Moran information as a control
#     moran_control = folium.Element(moran_html)
#     m.get_root().html.add_child(moran_control)

#     # Define style function with proper error handling
#     def style_function(feature):
#         try:
#             # Get value from GeoJSON properties
#             if "properties" in feature and "value" in feature["properties"]:
#                 value = feature["properties"]["value"]
#             else:
#                 # Default color if value not found
#                 return {
#                     "fillColor": "#808080",
#                     "color": "gray",
#                     "weight": 0.5,
#                     "fillOpacity": 0.5,
#                 }

#             # Check for NaN or invalid values
#             if value is None or pd.isna(value):
#                 return {
#                     "fillColor": "#808080",
#                     "color": "gray",
#                     "weight": 0.5,
#                     "fillOpacity": 0.5,
#                 }

#             # Normalize the value
#             norm_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
#             norm_value = max(0, min(1, norm_value))  # Ensure it's between 0 and 1

#             # Get RGB color from colormap
#             color = colormap(norm_value)

#             # Safely convert to hex
#             r = max(0, min(255, int(color[0] * 255)))
#             g = max(0, min(255, int(color[1] * 255)))
#             b = max(0, min(255, int(color[2] * 255)))

#             color_hex = f"#{r:02x}{g:02x}{b:02x}"

#             return {
#                 "fillColor": color_hex,
#                 "color": "gray",
#                 "weight": 0.5,
#                 "fillOpacity": 0.7,
#             }
#         except Exception as e:
#             print(f"Error in style_function: {e}")
#             # Return a default gray color if anything goes wrong
#             return {
#                 "fillColor": "#808080",
#                 "color": "gray",
#                 "weight": 0.5,
#                 "fillOpacity": 0.5,
#             }

#     # Create a hover function
#     def highlight_function(feature):
#         return {"weight": 2, "color": "black", "fillOpacity": 0.9}

#     # Add the GeoJSON layer with error handling
#     try:
#         # Convert the GeoDataFrame to a GeoJSON-like dictionary
#         geo_data = df.__geo_interface__

#         # Create tooltip fields that we're sure exist in the data
#         tooltip_fields = ["COUNTYFP", "value", "quadrant"]
#         tooltip_aliases = ["County:", f"{x}:", "Quadrant:"]

#         # Add spatial lag if available
#         if y in df.columns:
#             tooltip_fields.append(y)
#             tooltip_aliases.append("Spatial Lag:")

#         # Add the GeoJSON to the map
#         folium.GeoJson(
#             data=geo_data,
#             style_function=style_function,
#             highlight_function=highlight_function,
#             tooltip=folium.GeoJsonTooltip(
#                 fields=tooltip_fields,
#                 aliases=tooltip_aliases,
#                 style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
#             ),
#         ).add_to(m)
#     except Exception as e:
#         print(f"Error adding GeoJSON layer: {e}")
#         # Add a simple marker to show the map still works
#         folium.Marker(
#             [39.8, -98.5],
#             popup="Error loading geospatial data. See console for details.",
#         ).add_to(m)

#     # Add a legend
#     colormap_folium = folium.LinearColormap(
#         colors=["blue", "white", "red"],
#         vmin=vmin,
#         vmax=vmax,
#         caption=f"{x}",
#     )
#     m.add_child(colormap_folium)

#     # Save to HTML file
#     try:
#         # Display the map in a notebook if we're in a notebook environment
#         display(m)

#         # Save to HTML file
#         m.save(output_path)
#         print(f"Interactive map saved to {output_path}")
#         m.save(output_path)
#         print(f"Interactive map saved to {output_path}")
#     except Exception as e:
#         print(f"Error saving map: {e}")
#         # Try saving with a different name as fallback
#         fallback_path = "folium_map_fallback.html"
#         try:
#             m.save(fallback_path)
#             print(f"Map saved to fallback location: {fallback_path}")
#         except:
#             print("Could not save map to HTML file")

#     return m
