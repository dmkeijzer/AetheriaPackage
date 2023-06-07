from dataclasses import dataclass
@dataclass
class Radiator:
    #tubing parameters
    h_tube : float 
    t_tube : float
    t_channel : float

    #fin parameters
    t_fin : float
    HX_alpha: float
    HX_gamma : float
    HX_delta : float
    s_fin: float = None
    l_fin: float = None
    h_fin: float = None

    #numbers
    N_channel: int = None
    N_fin : int = None

    #surface area's
    A_cold: float = None
    A_hot: float = None
    A_fin : float = None
    A_fs_cross : float = None
