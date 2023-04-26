from range_analysis import iso_cities
import trips_file_creation as fc
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from cartopy import crs as ccrs
from random import sample
import pandas as pd
import os


#=========================================================================
# Get the required data to plot with
#=========================================================================

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
europe = world[world.continent == 'Europe']
europe = europe.to_crs(epsg=3395) # make the plot of Europe a conformal projection
plt_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "plotting_df.csv"))


#=========================================================================
# Here under are all the plotting commands
# epsg(3395) is a conformal projection (lat and lon lines remain adjacent)
#=========================================================================

fig = plt.figure(figsize=(8,4))
ax1 = plt.subplot(221, projection=ccrs.epsg(3395))
europe.plot(legend=False, cmap=matplotlib.cm.Greys, ec="black", lw=0.4,alpha=0.8,ax=ax1) 
plt.xlim([-2.26e6,3.78e6])
plt.ylim([3.7e6, 1.07e7])
plt.title(" GDP > 159.2 ")
ax2 = plt.subplot(222, projection=ccrs.epsg(3395)) 
europe.plot(legend=False, cmap=matplotlib.cm.Greys, ec="black", lw=0.4,alpha=0.8,ax=ax2) 
plt.xlim([-2.26e6,3.78e6])
plt.ylim([3.7e6, 1.07e7])
plt.title("152.2 > GDP > 84.9")
ax3 = plt.subplot(223, projection=ccrs.epsg(3395)) 
europe.plot(legend=False, cmap=matplotlib.cm.Greys, ec="black", lw=0.4,alpha=0.8,ax=ax3) 
plt.xlim([-2.26e6,3.78e6])
plt.ylim([3.7e6, 1.07e7])
plt.title("84.9 > GDP > 58.8 ")
ax4 = plt.subplot(224, projection=ccrs.epsg(3395))  
europe.plot(legend=False, cmap=matplotlib.cm.Greys, ec="black", lw=0.4,alpha=0.8,ax=ax4) 
plt.xlim([-2.26e6,3.78e6])
plt.ylim([3.7e6, 1.07e7])
plt.title("58.8 > GDP ")

lim = 300
a = 0.38
iso = iso_cities(lim)

#-----------------------------------------------------------------------------------
# This loop plots all the cities and their surrounding circles into their subplots
# It first checks whether the city is isolated to mark it black.
# then there some if statements used to put them into the right category of gdp
# the function .tissot from cartopy is used for the circles
#---------------------------------------------------------------------------------

for row in np.delete(plt_data.to_numpy(), 0 , 1):
   if row[0] in iso:
      print(f"{row[0]} isolated city")
      col = "black"
   else:
      col = sample(["b", "g", "r", "gold","peru", "purple"],1)[0]
   
   if row[3] >= 159.2:
      ax1.tissot(rad_km=lim, lons= row[2], lats=row[1], n_samples=36 , ec= "black",  zorder=10, alpha= a, color= col)
   if row[3] >= 84.9 and row[3] < 159.2:
      ax2.tissot(rad_km=lim, lons= row[2], lats=row[1], n_samples=36 , ec= "black",  zorder=10, alpha= a, color= col)
   if row[3] >= 58.8 and row[3] < 84.9:
      ax3.tissot(rad_km=lim, lons= row[2], lats=row[1], n_samples=36 , ec= "black",  zorder=10, alpha= a, color= col)
   if row[3] < 58.8:
      ax4.tissot(rad_km=lim, lons= row[2], lats=row[1], n_samples=36 , ec= "black",  zorder=10, alpha= a, color= col)

fig.tight_layout()
fig.subplots_adjust(wspace= -0.73)
fig.subplots_adjust(hspace= 0.2)
plt.show()







