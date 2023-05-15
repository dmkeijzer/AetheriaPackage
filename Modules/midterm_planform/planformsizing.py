import numpy as np
from dataclasses import dataclass

@dataclass
class Wing():
    surface: float
    taper: float
    aspectratio: float
    span: float = None
    chord_root: float = None
    chord_tip: float = None
    chord_mac: float = None
    y_mac: float = None
    tan_sweep_LE: float = None
    quarterchord_sweep: float = None
    X_lemac: float = None
    effective_aspectratio: float = None
    effective_span: float = None

def wing_planform(wing: Wing):

    wing.span  = np.sqrt( wing.aspectratio * wing.surface)
    wing.chord_root = 2 * wing.surface / ((1 + wing.taper) * wing.span)
    wing.chord_tip = wing.taper * wing.chord_root
    wing.chord_mac = (2 / 3) * wing.chord_root  * ((1 + wing.taper + wing.taper ** 2) / (1 + wing.taper))
    wing.y_mac = (wing.span / 6) * ((1 + 2 * wing.taper) / (1 + wing.taper1))
    wing.tan_sweep_LE = 0.25 * (2 * wing.chord_root / wing.span) * (1 - wing.taper) + np.tan(wing.quarterchord_sweep)

    wing.X_lemac = wing.y_mac * wing.tan_sweep_LE
    return wing

def winglet_correction(wing: Wing, winglet_correction: float):
    wing.effective_aspectratio = wing.aspectratio * winglet_correction
    wing.effective_span = wing.span*np.sqrt(wing.effective_aspectratio/wing.aspectratio)
    return wing
    
