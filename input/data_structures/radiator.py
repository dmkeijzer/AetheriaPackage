from dataclasses import dataclass
import pandas as pd
import os
import sys
import pathlib as pl



sys.path.append(str(list(pl.Path(__file__).parents)[3]))
os.chdir(str(list(pl.Path(__file__).parents)[3]))


@dataclass
class Radiator:

    #needed parameters
    W_HX : float
    H_HX: float
    Z_HX: float
    h_tube : float = None
    t_tube : float = None
    t_channel : float = None
    s_fin: float = None
    l_fin: float = None
    h_fin: float = None
    t_fin : float = None

    #calcalated parameters
    
    HX_alpha: float = None
    HX_gamma : float = None
    HX_delta : float= None

    #numbers
    N_channel: int = None
    N_fin : int = None

    #surface area's
    A_cold: float = None
    A_cross_hot: float = None
    A_hot: float = None
    A_fin : float = None
    A_fs_cross : float = None
    W_channel: float = None


    def load(self):
        df = pd.read_csv(r"input/radiator_values/HX.csv")
        self.h_tube = df['h_tube'][0]
        self.t_tube = df['t_tube'][0]
        self.t_channel = df['t_channel'][0]
        self.s_fin = df['s_fin'][0]
        self.t_fin = df['t_fin'][0]
        self.h_fin = df['h_fin'][0]
        self.l_fin = df['l_fin'][0]

    def dump(self):

        column =  [ 'h_tube', 't_tube', 't_channel', 's_fin', 'l_fin', 'h_fin', 't_fin']
        data = [ self.h_tube, self.t_tube, self.t_channel, self.s_fin, self.l_fin, self.h_fin, self.t_fin] 
        data = dict(zip(column,data))
        df = pd.DataFrame.from_dict([data])
        df.to_csv("radiator_values/HX.csv", columns=list(data.keys()), index= False)


