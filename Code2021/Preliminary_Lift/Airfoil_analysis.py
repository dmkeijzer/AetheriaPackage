import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress

def airfoil_stats():
    df1 = pd.read_csv(r"C:\Users\damie\OneDrive\Desktop\Damien\Honours\Wigeon_code\DSE2021\Preliminary_Lift\Airfoil_data\NACA44017_Re4.500.csv")
    df2 = pd.read_csv(r"C:\Users\damie\OneDrive\Desktop\Damien\Honours\Wigeon_code\DSE2021\Preliminary_Lift\Airfoil_data\NACA44017_Re2.300.csv")
    df1["cl/cd"] = df1["CL"]/df1["CD"]

    Clmax = np.max(df2["CL"])
    Cdmin = np.min(df1["CD"])
    Cl_Cdmin = np.average(df1["CL"][df1["CD"] == Cdmin])
    Cm = np.average(df1["Cm"][df1["CL"] == Cl_Cdmin])

    Clalpha = (np.average(df1["CL"][df1["alpha"] == 8]) - np.average(df1["CL"][df1["alpha"] == 0]))/8
    Clalpha1 = (np.average(df1["CL"][df1["alpha"] == 5]) - np.average(df1["CL"][df1["alpha"] == -1])) / 6

    clcdmax = np.max(df1["cl/cd"])
    Cl_maxld = np.average(df1["CL"][df1["cl/cd"] == clcdmax])
    a_clmax = np.average(df2["alpha"][df2["CL"] == Clmax])
    a_0L = -np.average(df1["CL"][df1["alpha"] ==0])/Clalpha1

    return Clmax, Cdmin, Cl_Cdmin, Cm, Clalpha, clcdmax, Cl_maxld, a_clmax, a_0L
# print(airfoil_stats()[4])
def airfoil_datapoint(type, Re, alpha):

    if Re == "Stall":
        df = pd.read_csv(r"C:\Users\damie\OneDrive\Desktop\Damien\Honours\Wigeon_code\DSE2021\Preliminary_Lift\Airfoil_data\NACA44017_Re2.300.csv")
    else:
        df = pd.read_csv(r"C:\Users\damie\OneDrive\Desktop\Damien\Honours\Wigeon_code\DSE2021\Preliminary_Lift\Airfoil_data\NACA44017_Re4.500.csv")
    return np.average(df[type][df["alpha"] == float(alpha)])


def Cd(CL):
    df = pd.read_csv(r"C:\Users\damie\OneDrive\Desktop\Damien\Honours\Wigeon_code\DSE2021\Preliminary_Lift\Airfoil_data\NACA44017_Re4.500.csv")
    Cl_vals = np.array(df["CL"][df["alpha"]<18.25])
    Cd_vals = np.array(df["CD"][df["alpha"]<18.25])
    ## # print(Cl_vals)
    fcd = interp1d(Cl_vals, Cd_vals, kind='quadratic', fill_value="extrapolate")

    CL = np.minimum(CL, np.array(df["CL"])[np.array(df["alpha"]) == 18])
    CD = fcd(CL)
    if len(CD)==1:
        CD = float(CD)
    return CD

def Cm_ac(sweep, ARw):
    df1 = pd.read_csv(r"C:\Users\damie\OneDrive\Desktop\Damien\Honours\Wigeon_code\DSE2021\Preliminary_Lift\Airfoil_data\NACA44017_Re4.500.csv")
    alpha = np.array(df1["alpha"][(df1["alpha"] <5) & (df1["alpha"] >-3)])
    Cm_lst = np.array(df1["Cm"][(df1["alpha"] <5) & (df1["alpha"] >-3)])
    CN_lst = np.array(df1["CL"][(df1["alpha"] <5) & (df1["alpha"] >-3)])*np.cos(alpha*np.pi/180)+ np.array(df1["CD"][(df1["alpha"] <5) & (df1["alpha"] >-3)])*np.sin(alpha*np.pi/180)
    Cm_curve = linregress(alpha,Cm_lst)
    CN_curve = linregress(alpha, CN_lst)
    # plt.plot(alpha,Cm_lst)
    # plt.show()
    ac = 0.25 + Cm_curve[0]/CN_curve[0]
    Cm_ac = np.average(Cm_lst+(ac-0.25)*CN_lst)
    Cm_ac_w = Cm_ac*(ARw*np.cos(sweep))/(ARw+2*np.cos(sweep))
    return Cm_ac_w, Cm_ac, ac, Cm_curve[2], CN_curve[2]

#plt.plot(np.arange(0,1.7,0.05), Cd(np.arange(0,1.7,0.05)))
#plt.show()
