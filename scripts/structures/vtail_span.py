import numpy as np
import matplotlib.pyplot as plt

def span_vtail(r, fuselage_width, dihedral):
    l = np.linspace(0, 2, 1000)
    formula = (fuselage_width/2 - (l + r)*np.cos(dihedral))**2 + ((l+r)*np.sin(dihedral))**2 - r**2
    plt.plot(l, formula)
    plt.show()
    # Bisection method
    tolerance = 1e-6
    a = l[0]
    b = l[-1]
    while b - a > tolerance:
        c = (a + b) / 2
        fc = (fuselage_width/2 - (c + r)*np.cos(dihedral))**2 + ((c+r)*np.sin(dihedral))**2 - r**2

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
    if np.tan(dihedral)*fuselage_width/2 < r:
        s = (a+b)/2 + r
    else:
        s = r/np.sin(dihedral)
        print("Radius too small")

    return s

r = 1
w = 1.7
dihedral = 35*np.pi/180

s = span_vtail(r, w, dihedral)

print(s)