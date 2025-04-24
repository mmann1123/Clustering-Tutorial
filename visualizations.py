# %% visualize Queen contiguity matrix

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pandas as pd

# Define polygons
polys = [
    Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),  # A
    Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),  # B
    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # C
    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # D
    Polygon([(2, 1), (3, 1), (3, 2), (2, 2)]),  # E
]

ids = list("ABCDE")
colors = ["lightcoral", "lightblue", "lightgreen", "khaki", "plum"]

gdf = gpd.GeoDataFrame({"id": ids, "geometry": polys})

# Create Queen matrix
queen = gdf.geometry.apply(
    lambda g: gdf.geometry.apply(
        lambda x: int(g.touches(x) or (g.intersects(x) and not g.equals(x)))
    )
)
queen.index = queen.columns = gdf["id"]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot polygons
gdf.plot(ax=ax1, color=colors, edgecolor="black")
for idx, row in gdf.iterrows():
    ax1.annotate(
        row["id"],
        xy=(row.geometry.centroid.x, row.geometry.centroid.y),
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
ax1.set_title("Polygons A–E")
ax1.axis("off")

# Show Queen matrix as table
ax2.axis("off")
ax2.set_title("Queen Contiguity Matrix")
table = ax2.table(
    cellText=queen.values,
    rowLabels=queen.index,
    colLabels=queen.columns,
    cellLoc="center",
    loc="center",
)
table.scale(0.7, 1.6)  # shrink width, stretch height
table.auto_set_font_size(False)
table.set_fontsize(10)

plt.tight_layout()
# plt.show()
# Save the figure
!mkdir -p images
plt.savefig("images/queen_contiguity_matrix.png", bbox_inches="tight")

# %%
