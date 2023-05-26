import numpy as np


def datcom_cl_alpha(A, mach, sweep_half):
    beta = np.sqrt(1 - mach**2)
    return (2*np.pi*A)/(2 + np.sqrt(4 + ((A*beta)/0.95)**2*(1 + (np.tan(sweep_half)**2)/(beta**2))))