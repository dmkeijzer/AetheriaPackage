
from pdffit import distfit as pf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stat
import os
import sys
import pathlib as pl
import geopandas as gpd
import numpy as np
import pandas as pd
from geopy.distance import distance
import os
import pathlib as pl
from math import ceil

#----------------------------
# Get the required data
#--------------------------------

df_trips = pd.read_csv(r"input/RangeAnalysisData/trip_2.0.csv").to_numpy() # array - example row ['The Hague-Bremen' 326.5480004995904 'The-Bre-dir' 99.0] 


class Bin():
    def __init__(self, frequency, edges, flight_df, total_df_gdp):
        self.left_bound =  edges[0]
        self.right_bound = edges[1]
        self.flights = flight_df[(flight_df[:,1].astype("float64") >= self.left_bound) * (flight_df[:,1].astype("float64") < self.right_bound)]
        self.freq = frequency
        self.df_gdp = total_df_gdp

    def bin_expectation(self):
        pass
        labels = self.flights[:,2]
        sum_bin_gdp =  np.sum(self.flights[:,3].astype("float64"))

        eq_par_trip = [] # list - "equal parent trip, meaning flight that came from the same two trip flight"
        
        for idx_i, i in enumerate(labels):
            for idx_j, j in enumerate(labels):
                if idx_i != idx_j:
                    if i == j:
                        eq_par_trip.append([self.flights[idx_i,0], self.flights[idx_j,0], self.flights[idx_i, 3]])
        eq_par_trip = np.array(eq_par_trip)
        if len(eq_par_trip) > 0:
            corr_fac = np.sum(eq_par_trip[: : 2,2].astype("float64")) / self.df_gdp  # corrrection factor for rule of addition, P(A n B) happens when a two way trip had the same parent trip
            corr_bin_gdp = sum_bin_gdp / self.df_gdp - corr_fac
            weighted_freq = self.freq * corr_bin_gdp
        else: 
            weighted_freq = self.freq *  (sum_bin_gdp / self.df_gdp)
        
        return weighted_freq

#===================================================================
# Define all the functions for the analysis, see their docstrings
#===================================================================

def iso_cities(lim):
    """Returns a list of isolated cities, i.e cities that cannot be reached by the eVTOL

    :param lim: Set the range limit on what is not reachable
    :type lim: int
    :return: A list of isolated cities in string format
    :rtype: list
    """
    iso = [] # ['Madrid', 'Lisbon', 'Warsaw', 'Bucharest'] result with 400

    for idx_i, city in enumerate(df_geo[:,0]):
        loc = []
        
        for idx_j, trip in enumerate(df_trips[:,0]):
            if trip.find(city) != -1: #checks whether city from outer loop is in the trip from the inner loop
                loc.append(idx_j)
        
        if len(loc) == 0: # if city is not recognized, move to next city
            continue
        all_trips_check = df_trips[loc, 1] < lim # list of all trips containing the city from outer loop
        if np.sum(all_trips_check) == 0:
            iso.append(city)
    
    return iso

def two_trip_analysis(lim):
    """_summary_

    :param lim: _description_
    :type lim: _type_
    :return: _description_
    :rtype: _type_
    """
    
    data = []

    two_trip = df_trips[(df_trips[:,1] < 2 * lim) * (df_trips[:,1] > lim) ] #selects trips which could be flown in two flight but not one
    one_trip = df_trips[df_trips[:,1] < lim] #array - selects direct flights

    for trip_2 in two_trip:
        depar_connect = np.ones((1,4)) # array - initialize arrays which will contain all direct connections to the current hub in the loop
        dest_connect = np.ones((1,4)) 
        
        depar = trip_2[0].split("-")[0] # string - The departure hub/city
        dest = trip_2[0].split("-")[1]
        
        for i in one_trip: # loop to get all connections from departure hub under the limit
            if i[0].split("-")[0] == depar or i[0].split("-")[1] == depar: 
                depar_connect = np.append(depar_connect, [i], axis=0)
        
        for j in one_trip: # loop to get all connect to destination hub under the limit
            if j[0].split("-")[1] == dest or j[0].split("-")[0] == dest: 
                dest_connect = np.append(dest_connect, [j] , axis=0)

        depar_connect = np.delete(depar_connect, 0 , axis=0) # deleting ones created at the beginning to avoid error
        dest_connect = np.delete(dest_connect, 0 , axis=0)
        
        hubs_depar = [i.split("-")[1] if i.split("-")[0] == depar else i.split("-")[0]  for i in depar_connect[:,0]] #creating list with connected hubs so we math them
        hubs_dest = [i.split("-")[1] if i.split("-")[0] == dest else i.split("-")[0]  for i in dest_connect[:,0]]

        if len(set(hubs_dest).intersection(hubs_depar)) == 0: # bool - See if a match exists between connected hubs, if not continue to next trip
            continue
        else:
            intersections = set(hubs_dest).intersection(hubs_depar) # set - intersection refers to the hubs both departure and destination could fly to
            lst_depar_intersec = [] # list - initalize list that will contain distance fro departure to intersection
            lst_dest_intersec = []
            
            for idx in range(len(intersections)): #this loop gets the distances from all possible routes so  from departure - intersections and intersection - destination
                lst_depar_intersec.append([i[1]  for i in one_trip if depar + "-" + list(intersections)[idx] == i[0] or list(intersections)[idx] + "-"  + depar == i[0]][0]) 
                lst_dest_intersec.append([i[1]  for i in one_trip if dest + "-" + list(intersections)[idx] == i[0] or list(intersections)[idx] + "-"  + dest == i[0]][0])
            
            sum_range = np.array(lst_depar_intersec) + np.array(lst_dest_intersec) # array - sum the departure - intersection - destination ranges 
            idx_intersec = np.where(sum_range == min(sum_range))[0][0] # integer - represents the index
            shorterst_intersec = list(intersections)[idx_intersec] # string - city shortest intersectiom
            label = depar[:3] + "-" + dest[:3] + "-2step"
            sum_gdp = trip_2[3] # Get the sum gdp, which will be necessary for the probability distribution
            data.append([depar + "-" + shorterst_intersec, float(lst_depar_intersec[idx_intersec]) ,label , sum_gdp] )
            data.append([shorterst_intersec + "-" + dest, float(lst_dest_intersec[idx_intersec]) , label , sum_gdp])
    
    data = np.array(data) # array - example of a row ['Lyon-Milan', '342.78286276713493', 'Par-Mil-2step', '955.8'] 
    return data

def plot_hist_two_trip_weightless(lim , n_bins):
    """_summary_

    :param lim: _description_
    :type lim: _type_
    :param n_bins: _description_
    :type n_bins: _type_
    """
    
    
    direct_flights = df_trips[df_trips[:,1] < lim]
    two_step_flights = two_trip_analysis(lim)
    if np.size(two_step_flights) != 0:
        total_flights = np.append(direct_flights, two_step_flights, axis= 0) # array - example ['Paris-Dortmund', 470.1237855871516, 'Par-Dor-dir', 910.4]
    else:
        total_flights = direct_flights
    hist_data = total_flights[:,1].astype("float64") 
    bfd = pf.BestFitDistribution(pd.DataFrame(hist_data))
    bfd.analyze(title="Two trip distribution weightless", x_label="range", y_label='freq', allBins= n_bins, outputFilePrefix= f"../distr_fig/two_trip_fit_weightless_" + "range_" + str(lim) + "_bins_" + str(n_bins) , imageFormat="pdf")
    print(bfd.best_dist)


def plot_hist_two_trip_weights(lim, n_bins=9): 
    """_summary_

    :param lim: _description_
    :type lim: _type_
    :param n_bins: _description_, defaults to 9
    :type n_bins: int, optional
    """
    corr_bins = [] # corrected bins - initialize which will contain classes
    
    direct_flights = df_trips[df_trips[:,1] < lim]
    two_step_flights = two_trip_analysis(lim)
    if np.size(two_step_flights) != 0:
        total_flights = np.append(direct_flights, two_step_flights, axis= 0) # array - example ['Paris-Dortmund', 470.1237855871516, 'Par-Dor-dir', 910.4]
    else:
        total_flights = direct_flights
    hist_data = total_flights[:,1].astype("float64")
    if np.size(two_step_flights) != 0:
        df_sum_gdp = np.sum(direct_flights[:,3].astype("float64")) + np.sum(two_step_flights[:,3].flatten().astype("float64")[::2])
    else:
        df_sum_gdp = np.sum(direct_flights[:,3].astype("float64"))
    
    hist_freq, bin_edges = np.histogram(hist_data, bins= n_bins)
    bin_edges[-1] = lim # alter last edge manually to the limit value
    
    for idx, i in enumerate(bin_edges):
        if bin_edges[idx] == bin_edges[-1]:
            continue
        state_edges = [i, bin_edges[idx + 1]]
        state_freq = hist_freq[idx]
        state_bin = Bin(state_freq, state_edges, total_flights, df_sum_gdp) # Use created class to do computations on the bins 
        corr_bins.append(state_bin)

    
    raw_data_expon_fit = [list(np.linspace(i.left_bound, i.right_bound,  ceil(i.bin_expectation()))) for i in corr_bins ]
    
    
    flat_raw_data_expon = np.array([item for sublist in raw_data_expon_fit for item in sublist]) # flattening using list comprehensino nparray.flatten() did not work
    bfd = pf.BestFitDistribution(pd.DataFrame(flat_raw_data_expon))
    best_fit, params  = bfd.best_fit_distribution()[0][0:2]

    arg = params[:-2]
    loc = params[-2]
    scale =  params[-1]
    x = np.linspace(np.min(flat_raw_data_expon), np.max(flat_raw_data_expon) ,  1000)
    pdf = best_fit.pdf(x, loc= loc, scale= scale, *arg)
    cdf = best_fit.cdf(x, loc= loc, scale= scale, *arg)
    # matplotlib.use('agg')
    plt.hist(flat_raw_data_expon, bins=n_bins, alpha= 0.4)
    plt.twinx()
    plt.plot(x, pdf, "k-.", label= "pdf")
    plt.xlabel("Kilometers")
    plt.show()
    return best_fit, arg, loc, scale



#--------------------------------
# Required paths and reading in files
#--------------------------------
df_cities_shp = gpd.read_file(r'input/RangeAnalysisData/ne_10m_populated_places_simple.shp')
cities_df = pd.read_excel(r'input/RangeAnalysisData/cities_by_gmp.xlsx')
city_coord_gdp = pd.read_csv(r'input/RangeAnalysisData/plotting_df.csv')
city_coord_gdp_arr = pd.read_csv(r'input/RangeAnalysisData/plotting_df.csv').to_numpy()

#----------------------------------------------------------------------------------------------
# The code hereunder checks which cities are available in the shapefile from our required
# list and their index in the shapefile to be able to retrieve them.
#----------------------------------------------------------------------------------------------

shp_cities = df_cities_shp["name"].to_numpy() # Available cities from shapefile
list_cities = cities_df["metropolitan_area"].to_numpy() #list of required cities
loc = []

for city in list_cities:
    idx = np.where(shp_cities == city)[0]
    if len(idx) > 0:
        loc.append(idx[0])

eur_cities = df_cities_shp.iloc[loc] # final result (european cities)

#---------------------------------------------------------------------------------------------------
# Some of the data gets filtered and replaced manually. Also some missing data gets added manually
# Only Rhine - Neckar is left out due to it consisting out of a lot of municipals
#---------------------------------------------------------------------------------------------------

df_geo = eur_cities[["name", "latitude", "longitude"]].to_numpy()
filter = (df_geo[:,0]!="Valencia") * (df_geo[:,0]!="Naples") * (df_geo[:,0]!="Dublin") * (df_geo[:,0]!="Athens") * (df_geo[:,0]!="Barcelona")
df_geo = df_geo[filter, :] # removed cities due to untrustworthy data 37 cities left
correction_data = [["London", 51.509865, -0.118092], # Here they are added back again with the correct coordinates
                   ["Valencia", 39.466667, -0.375000],
                   ["Naples", 40.853294, 14.305573],
                   ["Dublin", 53.350140, -6.266155],
                   ["Athens", 37.983810, 23.727539],
                   ["Barcelona", 41.390205, 2.154007],
                   ["Gothenburg", 57.708870, 11.974560],
                   ["Nuremburg", 49.460983, 11.061859],
                   ["Braunschweig", 52.266666, 10.51667],
                   ["Hanover", 52.373920, 9.735603],
                   ["Antwerp", 51.2194475, 4.4024643],
                   ["Brescia", 45.541553, 10.211802]]

df_geo = np.append(df_geo, correction_data, axis=0)




def create_trip_file():

    n = 1 
    df_distance = [] 

    for idx_i, row_i in enumerate(city_coord_gdp.to_numpy()):  
        i_str = str(row_i[1])
        i_coord = (row_i[2] , row_i[3])
        
        for idx_j, row_j in enumerate(city_coord_gdp.to_numpy()[n:, :], start= n):
            j_str = str(city_coord_gdp_arr[idx_j,1])
            j_coord = (city_coord_gdp_arr[idx_j, 2] , city_coord_gdp_arr[idx_j, 3])
            journey = i_str + "-" + j_str
            dist_journey = distance(i_coord, j_coord).km #computes distance based on coordinates
            sum_gdp = row_i[4] + city_coord_gdp_arr[idx_j, 4]
            label = i_str[:3] + "-" + j_str[:3] + "-dir" #give label important for last computational step
            
            df_distance.append([journey, dist_journey, label, round(sum_gdp,1) ])
        
        n = n + 1

    #write data to csv file

    df_distance = pd.DataFrame(df_distance)
    df_distance.to_csv(r"C:\Users\damie\OneDrive\Desktop\Damien\Wigeon_proj\own_code\trip_2.0.csv", header=["city-city", "dist", "label", "sum_gdp"], index= False)
    print("Journey file has been correctly written and saved")







