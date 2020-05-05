import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy

def plot_map():
    ax = plt.gca()
    ax.set_global()
    # ax.frameon = True
    # ax.outline_patch.set_linewidth(1.5)
    # ax.outline_patch.set_zorder(100)

    # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
    # function around 180deg
    gl = ax.gridlines(draw_labels=False,
                      linewidth=1, color='lightgray', alpha=0.5,
                      linestyle='-', zorder=-1.5)
    gl.top_labels = False
    gl.left_labels = False
    gl.xlines = True

    # Add Coastline
    ax.add_feature(cartopy.feature.LAND, zorder=-2, edgecolor='black',
                   linewidth=0.5, facecolor=(0.9, 0.9, 0.9),
                   clip_on=True)


############ DOESNT WORK #############
fig = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')

g = fig.add_gridspec(1, 3)  # Gridspec(1, 3) doesnt either

# Title
ax1 = plt.subplot(g[0, :2], projection=cartopy.crs.PlateCarree())
plot_map()

# Title
ax2 = plt.subplot(g[0, 2], projection=cartopy.crs.PlateCarree())
plot_map()
ax2.set_extent([0, 45, 0, 45])

# Can't use this
plt.tight_layout()

plt.show()


############ DOESNT WORK Either #############
fig = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')

g = fig.add_gridspec(1, 3)  # Gridspec(1, 3) doesnt either

# Title
ax1 = plt.subplot(121, projection=cartopy.crs.PlateCarree())
plot_map()

# Title
ax2 = plt.subplot(122, projection=cartopy.crs.PlateCarree())
plot_map()
ax2.set_extent([0, 45, 0, 45])

# Can't use this
plt.tight_layout()

plt.show()