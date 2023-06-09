from dataclasses import dataclass
@dataclass
class Radiator:

    #needed parameters
    W_HX : float
    H_HX: float
    Z_HX: float
    h_tube : float 
    t_tube : float
    t_channel : float
    s_fin: float
    l_fin: float
    h_fin: float
    t_fin : float

    #fin parameters
    
    HX_alpha: float = None
    HX_gamma : float = None
    HX_delta : float= None

    #calculated parameters
   

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

    
