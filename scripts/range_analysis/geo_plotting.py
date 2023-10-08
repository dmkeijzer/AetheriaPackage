import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from geopy.distance import distance
import seaborn as sns
from cartopy import crs as ccrs
from random import sample
import matplotlib.patches as mpatches
import pandas as pd
import os
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.range_analysis.range_analysis import iso_cities
import modules.range_analysis.ShapeFileHandler as fc

#=========================================================================
# Get the required data to plot with
#=========================================================================

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
europe = world[world.continent == 'Europe']
europe = europe.to_crs(epsg=3395) # make the plot of Europe a conformal projection
plt_data_og = pd.read_csv(r"input\RangeAnalysisData\plotting_df.csv")
plt_data = plt_data_og.to_numpy()[plt_data_og.to_numpy()[:,4] >= 159.2]




#=========================================================================
# Here under are all the plotting commands
# epsg(3395) is a conformal projection (lat and lon lines remain adjacent)
#=========================================================================

fig = plt.figure(figsize=(8,4))
ax1 = plt.subplot(111, projection=ccrs.epsg(3395))
europe.plot(legend=False, cmap=matplotlib.cm.Greys, ec="black", lw=0.4,alpha=0.8,ax=ax1) 
plt.xlim([-2.26e6,3.78e6])
plt.ylim([3.7e6, 1.07e7])
# plt.title(" GDP > 159.2 ")
# ax2 = plt.subplot(222, projection=ccrs.epsg(3395)) 
# europe.plot(legend=False, cmap=matplotlib.cm.Greys, ec="black", lw=0.4,alpha=0.8,ax=ax2) 
# plt.xlim([-2.26e6,3.78e6])
# plt.ylim([3.7e6, 1.07e7])
# plt.title("152.2 > GDP > 84.9")
# ax3 = plt.subplot(223, projection=ccrs.epsg(3395)) 
# europe.plot(legend=False, cmap=matplotlib.cm.Greys, ec="black", lw=0.4,alpha=0.8,ax=ax3) 
# plt.xlim([-2.26e6,3.78e6])
# plt.ylim([3.7e6, 1.07e7])
# plt.title("84.9 > GDP > 58.8 ")
# ax4 = plt.subplot(224, projection=ccrs.epsg(3395))  
# europe.plot(legend=False, cmap=matplotlib.cm.Greys, ec="black", lw=0.4,alpha=0.8,ax=ax4) 
# plt.xlim([-2.26e6,3.78e6])
# plt.ylim([3.7e6, 1.07e7])
# plt.title("58.8 > GDP ")

def iso_cities_altered(lim):
    """Returns a list of isolated cities, i.e cities that cannot be reached by the eVTOL

    :param lim: Set the range limit on what is not reachable
    :type lim: int
    :return: A list of isolated cities in string format
    :rtype: list
    """
    iso = [] # ['Madrid', 'Lisbon', 'Warsaw', 'Bucharest'] result with 400

    for  departure in plt_data[:, 1:]:
      count = 0
      for arrival in plt_data[:, 1:]:
         dist = distance((departure[1], departure[2]), (arrival[1], arrival[2])).km
         if dist < lim and departure[0] != arrival[0]:
            count += 1
      if count == 0:
         iso.append(departure[0])
   
    return iso
         

lim = 300
a = 0.2
iso = iso_cities_altered(lim)

        
        
#-----------------------------------------------------------------------------------
# This loop plots all the cities and their surrounding circles into their subplots
# It first checks whether the city is isolated to mark it black.
# then there some if statements used to put them into the right category of gdp
# the function .tissot from cartopy is used for the circles
#---------------------------------------------------------------------------------


for idx, row in enumerate(np.delete(plt_data_og.to_numpy(), 0 , 1)):
   if row[0] in iso:
      print(f"{row[0]} isolated city")
      col = [0,0,0, a]
      edgecol = [0,0,0, 0.9]
   else:
      col = list(sns.color_palette("tab10", 50  )[idx])
      edgecol = list(col)
      col.append(a)
      edgecol.append(0.8)
   
   # ax1.tissot(rad_km=lim, lons= row[2], lats=row[1], n_samples=36 , facecolor= col,   zorder=10,  edgecolor= edgecol, lw= 1.5)

   #Create dot for the city itself
   edgecol[-1] = 1
   ax1.tissot(rad_km=40, lons= row[2], lats=row[1], n_samples=36 , facecolor= edgecol,   zorder=10)
   ax1.text(row[2] , row[1] - 1, row[0], fontsize=8, color='black',horizontalalignment='right', transform=ccrs.PlateCarree())
   # if row[3] >= 84.9 and row[3] < 159.2:
   #    ax2.tissot(rad_km=lim, lons= row[2], lats=row[1], n_samples=36 , ec= "black",  zorder=10, alpha= a, color= col)
   # if row[3] >= 58.8 and row[3] < 84.9:
   #    ax3.tissot(rad_km=lim, lons= row[2], lats=row[1], n_samples=36 , ec= "black",  zorder=10, alpha= a, color= col)
   # if row[3] < 58.8:
   #    ax4.tissot(rad_km=lim, lons= row[2], lats=row[1], n_samples=36 , ec= "black",  zorder=10, alpha= a, color= col)

fig.tight_layout()
# fig.subplots_adjust(wspace= -0.73)
# fig.subplots_adjust(hspace= 0.2)
plt.show()







