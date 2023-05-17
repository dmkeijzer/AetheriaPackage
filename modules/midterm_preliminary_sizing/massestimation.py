# -*- coding: utf-8 -*-
import csv
from scipy import stats
import numpy as np
import os
import sys
import pathlib as pl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

mass_data = os.path.join(list(pl.Path(__file__).parents)[2], "input/Preliminary_sizing/massdatabase.csv")

# function as y = ax + b
def func(x, slope, intercept):
  x = np.array(x)
  return slope*x + intercept

def save_fig(name):
  output_directory = str(list(pl.Path(__file__).parents)[2]) + '\\output\\mass_regression_plots\\'
  plt.savefig(output_directory + str(name) + '.png')

# function that calculates the OEM and MTOM of Aetheria based on payload
def mass_estimation(payloadAetheria, PLOT=False, PRINT=False):
  # define lists for range, payload, OEM, MTOM and prototype
  Range, Payload, OEM, MTOM, PROTO = [], [], [], [], []
  # define lists for range, payload, OEM, MTOM for only prototypes
  Rangereal, Payloadreal, OEMreal, MTOMreal = [], [], [], []
  # read csv file
  with open(mass_data) as file:
    csvreader = csv.reader(file)
    for row in csvreader:
      # include all prototypes under 10000 kg and fill array for each variable
      if int(row[4]) < 10000:
      #if int(row[4]) < 3180 and int(row[4]) > 500 and int(row[5]) == 1:
        Range.append(int(row[1]))
        Payload.append(int(row[2]))
        OEM.append(int(row[3]))
        MTOM.append(int(row[4]))
        PROTO.append(int(row[5]))
      # do the same for only prototypes
      if int(row[5]) == 1:
        Rangereal.append(int(row[1]))
        Payloadreal.append(int(row[2]))
        OEMreal.append(int(row[3]))
        MTOMreal.append(int(row[4]))


  ## Regression Payload vs MTOM
    # for all eVTOLs:
  slope1, intercept1, r1, p1, std_err1 = stats.linregress(Payload, MTOM)
    # for only real flying prototypes
  sloper, interceptr, rr, pr, std_errr = stats.linregress(Payloadreal, MTOMreal)
  # MTOM lines for all and real eVTOLs:
  MTOMregress = func(Payload, slope1, intercept1)
  MTOMregressreal = func(Payload, sloper, interceptr)
  MTOMregressreal2 = func(Payloadreal, sloper, interceptr)
  rmse = np.sqrt(mean_squared_error(MTOM, MTOMregress))
  rmser = np.sqrt(mean_squared_error(MTOMreal, MTOMregressreal2))

  # MTOM and first OEM calculations for Aetheria
  mtomAetheria = payloadAetheria*slope1 + intercept1
  mtomAetheriar = payloadAetheria*sloper + interceptr
  oemAetheria = mtomAetheria - payloadAetheria
  oemAetheriar = mtomAetheriar - payloadAetheria

  # Regression MTOM vs OEM
    # for all eVTOLs:
  slope2, intercept2, r2, p2, std_err2 = stats.linregress(MTOM, OEM)
    # for only real flying prototypes
  slope2r, intercept2r, r2r, p2r, std_err2r = stats.linregress(MTOMreal, OEMreal)

  # OEM lines for all and real eVTOLs:
  OEMregress = func(MTOM, slope2, intercept2)
  OEMregressreal = func(MTOM, slope2r, intercept2r)
  OEMregressrealrmse = func(MTOMreal, slope2r, intercept2r)

  # second OEM calculations for Aetheria
  oemAetheria2 = mtomAetheria*slope2+intercept2
  oemAetheria2r = mtomAetheriar*slope2r+intercept2r

  if PLOT:
    # Payload vs MTOM plot
    for i in range(len(PROTO)):
      if PROTO[i] == 1:
        blue_dots = plt.scatter(Payload[i], MTOM[i], color='blue')
      else:
        red_dots = plt.scatter(Payload[i], MTOM[i], color='red')

    green_dot = plt.scatter(payloadAetheria, mtomAetheriar, color='green', zorder = 2)
    red_line, = plt.plot(Payload, MTOMregress, linestyle='-', color='red', alpha=0.7, label="Regression all")
    dashed_line, = plt.plot(Payload, MTOMregress, linestyle=(5, (5, 5)), color='blue', alpha=0.7, label="Regression all")
    blue_line, = plt.plot(Payload, MTOMregressreal, color='blue', zorder=1, label="Regression prototypes")

    plt.legend([(blue_line),(red_line, dashed_line), (blue_dots), (red_dots), (green_dot)], ['Regression prototypes','Regression all', 'Prototype eVTOLs', 'Concept eVTOLs', 'Aetheria'])

    plt.xlabel('Payload [kg]')
    plt.ylabel('MTOM [kg]')
    save_fig('payload_MTOM')
    plt.show()

    # MTOM vs OEM plot
    for i in range(len(PROTO)):
      if PROTO[i] == 1:
        blue_dots2 = plt.scatter(MTOM[i], OEM[i], color='blue')
      else:
        red_dots2 = plt.scatter(MTOM[i], OEM[i], color='red')
    green_dot2 = plt.scatter(mtomAetheriar, oemAetheriar, color='green', zorder=2)
    blue_line2, = plt.plot(MTOM, OEMregressreal,color='blue', zorder=1)
    plt.legend([(blue_line2),  (blue_dots2), (red_dots2), (green_dot2)],
               ['Regression prototypes', 'Prototype eVTOLs', 'Concept eVTOLs', 'Aetheria'])
    plt.xlabel('MTOM [kg]')
    plt.ylabel('OEM [kg]')
    save_fig('MTOM_OEM')
    plt.show()

  if PRINT:
    print("            all    only real")
    print("MTOM:      ", round(mtomAetheria), "   "  , round(mtomAetheriar),  "   kg")
    print("OEM:       ", round(oemAetheria) , "   "  , round(oemAetheriar) ,  "   kg")
    print("OEM2:      ", round(oemAetheria2), "   "  , round(oemAetheria2r),  "   kg")
    print("R^2:       ", round(r1*r1, 2)    , "   "  ,round(rr*rr, 2))
    print("rmse:      ", round(rmse)        , "    " ,round(rmser)         , "    kg")
    print("Data size: ", len(MTOM)          , "     ", len(MTOMreal))
  return mtomAetheriar, oemAetheriar






