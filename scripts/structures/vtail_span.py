import numpy as np
from warnings import warn

def span_vtail(r, w, g):
    """ Computes the span of the vtail based of the propellor radius, width of the fuselage and dihedral of the vtail

    :param r: propellor radius
    :type r: float
    :param w: width of the fuselage
    :type w: float
    :param g: dihedral of vtail
    :type g: float
    :return: span vtail
    :rtype: float
    """    
    l = np.linspace(0, 2, 1000)
    formula = (w/2 - (l + r)*np.cos(g))**2 + ((l+r)*np.sin(g))**2 - r**2
    # Bisection method
    warn("A built-in method could be used which would likely be much faster")

    tolerance = 1e-6
    a = l[0]
    b = l[-1]
    while b - a > tolerance:
        c = (a + b) / 2
        fc = (w/2 - (c + r)*np.cos(g))**2 + ((c+r)*np.sin(g))**2 - r**2
        if fc == 0:
            # Found exact zero-crossing
            return c
        elif np.sign(fc) == np.sign(formula[0]):
            # Zero-crossing lies between a and c
            a = c
        else:
            # Zero-crossing lies between c and b
            b = c
    # Return the approximate zero-crossing
    if np.tan(g)*w/2 < r:
        s = (a+b)/2 + r
    else:
        s = r/np.sin(g)
    return s


if __name__ == "__main__":
    prop_radius = 1                 # propeller radius in m
    fuselage_width = 1.7            # fuselage width in m
    dihedral_vtail = 30*np.pi/180   # dihedral of vtail

    s = span_vtail(prop_radius, fuselage_width, dihedral_vtail)

