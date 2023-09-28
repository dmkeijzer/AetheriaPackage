import numpy as np
from dataclasses import dataclass


def wing_planform(wing, MTOM: float, WS_cruise: float):

    wing.surface = MTOM / WS_cruise * 9.81
    wing.span  = np.sqrt( wing.aspect_ratio * wing.surface)
    wing.chord_root = 2 * wing.surface / ((1 + wing.taper) * wing.span)
    wing.chord_tip = wing.taper * wing.chord_root
    wing.chord_mac = (2 / 3) * wing.chord_root  * ((1 + wing.taper + wing.taper ** 2) / (1 + wing.taper))
    wing.y_mac = (wing.span / 6) * ((1 + 2 * wing.taper) / (1 + wing.taper))
    wing.sweep_LE = 0.25 * (2 * wing.chord_root / wing.span) * (1 - wing.taper) + np.tan(np.radians(wing.quarterchord_sweep))

    wing.x_lemac = wing.y_mac * wing.sweep_LE
    return wing

def winglet_correction(wing, winglet_correction: float):
    wing.effective_aspect_ratio = wing.aspect_ratio * winglet_correction
    wing.effective_span = wing.span*np.sqrt(wing.effective_aspectratio/wing.aspect_ratio)
    return wing
    
def winglet_factor(h_wl, b, k_wl):  #https://www.fzt.haw-hamburg.de/pers/Scholz/Aero/AERO_PUB_Winglets_IntrinsicEfficiency_CEAS2017.pdf

    return (1+(2/k_wl)*(h_wl/b))**2
