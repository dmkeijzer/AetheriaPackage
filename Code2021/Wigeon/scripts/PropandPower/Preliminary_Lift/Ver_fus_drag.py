import numpy as np

def fus_drag(Re, M, S_n, S_c, S_t, S):
    CDfp = 0.455 / (((np.log10(Re)) ** 2.58) * (1 + 0.144 * M * M) ** 0.58)

    K_n = 1.75
    K_c = 1.22
    K_t = 0.8

    return (K_n*S_n + K_c*S_c + K_t*S_t)*CDfp*(1/S)


print(fus_drag(50507738, 0.208, 6.43974,24.8657, 9.2226, 17))
