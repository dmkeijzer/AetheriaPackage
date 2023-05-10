import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def airfoil_stats():
    df1 = pd.read_csv("Airfoil_data_midterm2/NACA44017_Re2.300.csv")
    df2 = pd.read_csv("Airfoil_data_midterm2/NACA44017_Re1.700.csv")
    df1["cl/cd"] = df1["CL"]/df1["CD"]

    Clmax = np.max(df2["CL"])
    Cdmin = np.min(df1["CD"])
    Cl_Cdmin = np.average(df1["CL"][df1["CD"] == Cdmin])
    Cm = np.average(df1["Cm"][df1["CL"] == Cl_Cdmin])

    Clalpha = (np.average(df1["CL"][df1["alpha"] == 7]) -np.average(df1["CL"][df1["alpha"] == 1]))/6
    Clalpha1 = (np.average(df1["CL"][df1["alpha"] == 5]) - np.average(df1["CL"][df1["alpha"] == -1])) / 6

    clcdmax = np.max(df1["cl/cd"])
    Cl_maxld = np.average(df1["CL"][df1["cl/cd"] == clcdmax])
    a_clmax = np.average(df2["alpha"][df2["CL"] == Clmax])
    a_0L = -np.average(df1["CL"][df1["alpha"] ==0])/Clalpha1

    return Clmax, Cdmin, Cl_Cdmin, Cm, Clalpha, clcdmax, Cl_maxld, a_clmax, a_0L

def airfoil_datapoint(type, Re, alpha):

    if Re == "Stall":
        df = pd.read_csv("Airfoil_data_midterm2/NACA44017_Re1.700.csv")
    else:
        df = pd.read_csv("Airfoil_data_midterm2/NACA44017_Re2.300.csv")

    return np.average(df[type][df["alpha"] == alpha])


def Cd(CL):
    df = pd.read_csv("Airfoil_data_midterm2/NACA44017_Re2.300.csv")
    Cl_vals = np.array(df["CL"][df["alpha"]<17])
    Cd_vals = np.array(df["CD"][df["alpha"]<17])
    # print(Cl_vals)
    fcd = interp1d(Cl_vals,Cd_vals,kind = 'quadratic')
    return fcd(CL)

# plt.plot(np.arange(0,1.7,0.05), Cd(np.arange(0,1.7,0.05)))
# plt.show()

