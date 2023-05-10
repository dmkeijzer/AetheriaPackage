import numpy as np
from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
from Preliminary_Lift.LLTtest2 import downwash, downwash_upwash
from stab_and_ctrl.xcg_limits import bf_br, crf_crr, CLa, lambda_c4_to_lambda_c2


def deps_da_emp(lambda_c4f, bf, lh, h_ht, A, CLaf):
    """
    Estimate the downwash gradient of the front wing on the rear wing
    (accounting for additional downwash due to propellers).
    :param lambda_c4f: Quarter-chord sweep of the front wing
    :param bf: Span of the front wing
    :param lh: x-distance between aerodynamic centres of front and rear wing
    :param h_ht: Distance between wings normal to their chord planes
    :param A:  Aspect ratio of the front wing
    :param CLaf: Lift slope of the front wing
    :return: Downwash gradient of the front wing on the rear wing
    """
    r = lh * 2 / bf
    mtv = h_ht * 2 / bf  # Approximation
    Keps = (0.1124 + 0.1265 * lambda_c4f + 0.1766 * lambda_c4f ** 2) / r ** 2 + 0.1024 / r + 2
    Keps0 = 0.1124 / r ** 2 + 0.1024 / r + 2
    v = 1 + (r ** 2 / (r ** 2 + 0.7915 + 5.0734 * mtv ** 2)) ** (0.3113)
    de_da = Keps / Keps0 * CLaf / (np.pi * A) * (
            r / (r ** 2 + mtv ** 2) * 0.4876 / (np.sqrt(r ** 2 + 0.6319 + mtv ** 2)) + v * (
            1 - np.sqrt(mtv ** 2 / (1 + mtv ** 2))))

    return de_da


def de_da_compute(taperf, taperr, lambda_c4f, lambda_c4r, alpha, V,
                 dxrange, dzrange, Srange, Sr_Sfrange, Afrange, Arrange,
                 filename):
    dxgrid, dzgrid, Sgrid, Sr_Sfgrid, Afgrid, Argrid = np.meshgrid(
        dxrange, dzrange, Srange, Sr_Sfrange, Afrange, Arrange, indexing="ij"
    )

    bf, br = bf_br(Sgrid, Sr_Sfgrid, Afgrid, Argrid)
    crf, crr = crf_crr(Sgrid, Sr_Sfgrid, Afgrid, Argrid, taperf, taperr)

    de_da = np.zeros(dxgrid.shape)
    de_da_up = np.zeros(dxgrid.shape)

    for i, dx in enumerate(dxrange):
        for j, dz in enumerate(dzrange):
            for k, S in enumerate(Srange):
                for l, Sr_Sf in enumerate(Sr_Sfrange):
                    for m, Af in enumerate(Afrange):
                        for n, Ar in enumerate(Arrange):
                            print(bf[i, j, k, l, m, n],
                                    Af,
                                    crf[i, j, k, l, m, n],
                                    crf[i, j, k, l, m, n] * taperf,
                                    lambda_c4f,
                                    alpha,
                                    dz,
                                    dx,
                                    br[i, j, k, l, m, n],
                                    Ar,
                                    crr[i, j, k, l, m, n],
                                    crr[i, j, k, l, m, n] * taperr,
                                    lambda_c4r,
                                    V)
                            de_da[i, j, k, l, m, n], \
                            de_da_up[i, j, k, l, m, n] = \
                                downwash_upwash(
                                    bf[i, j, k, l, m, n],
                                    Af,
                                    crf[i, j, k, l, m, n],
                                    crf[i, j, k, l, m, n] * taperf,
                                    lambda_c4f,
                                    alpha,
                                    dz,
                                    dx,
                                    br[i, j, k, l, m, n],
                                    Ar,
                                    crr[i, j, k, l, m, n],
                                    crr[i, j, k, l, m, n] * taperr,
                                    lambda_c4r,
                                    V)

    np.save(filename + "_raw_de_da", de_da)
    np.save(filename + "_raw_de_da_up", de_da_up)

    de_da = de_da.flatten()
    de_da_up = de_da_up.flatten()

    pts = np.array([dxgrid.flatten(), dzgrid.flatten(), Sgrid.flatten(),
                    Sr_Sfgrid.flatten(), Afgrid.flatten(), Argrid.flatten()])
    pts = np.moveaxis(pts, [0], [1])

    np.save(filename + "_pts", pts)
    np.save(filename + "_de_da_vals", de_da)
    np.save(filename + "_de_da_up_vals", de_da_up)


def de_da_load(filename):
    pts = np.load(filename + "_pts.npy")
    de_da = np.load(filename + "_de_da_vals.npy")
    de_da_up = np.load(filename + "_de_da_up_vals.npy")
    return LinearNDInterpolator(pts, de_da), \
           LinearNDInterpolator(pts, de_da_up)


def compare_ll_interp(filename):
    taperf = 0.45
    taperr = 0.45
    lambda_c4f = 0
    lambda_c4r = 0

    de_da_interp, de_da_up_interp = de_da_load(filename)

    dx = 6
    dz = 1.2
    S = 8.417113787320769 * 2
    Sr_Sf = 1
    Af = 8
    Ar = 8

    res = 5
    dxrange = np.linspace(0, 9, res)
    dzrange = np.linspace(0, 3, res)
    Srange = np.linspace(5, 35, res)
    Sr_Sfrange = np.linspace(0.2, 5, res)
    Afrange = np.linspace(2, 14, res)
    Arrange = np.linspace(2, 14, res)

    def create_plot(dx, dz, S, Sr_Sf, Af, Ar, xaxis_range, xaxis_label,
                    alpha=np.deg2rad(5), V=55):
        bf, br = bf_br(S, Sr_Sf, Af, Ar)
        crf, crr = crf_crr(S, Sr_Sf, Af, Ar, taperf, taperr)

        ll_down_list = []
        interp_down_list = []
        ll_up_list = []
        interp_up_list = []
        for i in range(res):
            bf_pt = bf if not isinstance(bf, np.ndarray) else bf[i]
            Af_pt = Af if not isinstance(Af, np.ndarray) else Af[i]
            crf_pt = crf if not isinstance(crf, np.ndarray) else crf[i]
            dz_pt = dz if not isinstance(dz, np.ndarray) else dz[i]
            dx_pt = dx if not isinstance(dx, np.ndarray) else dx[i]
            br_pt = br if not isinstance(br, np.ndarray) else br[i]
            crr_pt = crr if not isinstance(crr, np.ndarray) else crr[i]
            S_pt = S if not isinstance(S, np.ndarray) else S[i]
            Sr_Sf_pt = Sr_Sf if not isinstance(Sr_Sf, np.ndarray) else Sr_Sf[i]
            Ar_pt = Ar if not isinstance(Ar, np.ndarray) else Ar[i]

            ll = downwash_upwash(bf_pt, Af_pt, crf_pt, crf_pt * taperf,
                                 lambda_c4f, alpha, dz_pt, dx_pt, br_pt,
                                 Ar_pt, crr_pt, crr_pt * taperr, lambda_c4r, V)
            # interp_down_val = de_da_interp(dx_pt, dz_pt, S_pt, Sr_Sf_pt,
            #                                Af_pt, Ar_pt)
            # interp_up_val = de_da_up_interp(dx_pt, dz_pt, S_pt, Sr_Sf_pt,
            #                                 Af_pt, Ar_pt)
            ll_down_list.append(ll[0])
            # interp_down_list.append(interp_down_val)
            ll_up_list.append(ll[1])
            # interp_up_list.append(interp_up_val)
        interp_down_list = de_da_interp(dx, dz, S, Sr_Sf, Af, Ar)
        interp_up_list = de_da_up_interp(dx, dz, S, Sr_Sf, Af, Ar)

        plt.subplot(211)
        plt.title("Downwash")
        plt.xlabel(xaxis_label)
        plt.ylabel(r"$d\varepsilon/d\alpha$")
        plt.plot(xaxis_range, interp_down_list, label="interpolated lifting line",
                 color="tab:blue", marker="o")
        plt.plot(xaxis_range, ll_down_list, label="lifting line",
                 color="tab:orange", marker="o")

        plt.subplot(212)
        plt.title("Upwash")
        plt.xlabel(xaxis_label)
        plt.ylabel(r"$d\varepsilon/d\alpha$")
        plt.plot(xaxis_range, interp_up_list,
                 label="interpolated lifting line",
                 color="tab:blue", marker="o")
        plt.plot(xaxis_range, ll_up_list, label="lifting line",
                 color="tab:orange", marker="o")

        plt.legend()

    create_plot(dxrange, dz, S, Sr_Sf, Af, Ar, dxrange, r"$dx$")
    plt.figure()
    create_plot(dx, dzrange, S, Sr_Sf, Af, Ar, dzrange, r"$dz$")
    plt.figure()
    create_plot(dx, dz, Srange, Sr_Sf, Af, Ar, Srange, r"$S$")
    plt.figure()
    create_plot(dx, dz, S, Sr_Sfrange, Af, Ar, Sr_Sfrange, r"$S_r/S_f$")
    plt.figure()
    create_plot(dx, dz, S, Sr_Sf, Afrange, Ar, Afrange, r"$A_f$")
    plt.figure()
    create_plot(dx, dz, S, Sr_Sf, Af, Arrange, Arrange, r"$A_r$")
    plt.show()


def compare_ll_emp_downwash():
    taperf = 0.45
    taperr = 0.45
    lambda_c4f = 0
    lambda_c4r = 0
    Claf = 6.1879
    S = 8.417113787320769 * 2

    xf = 0.5
    xr = 6.5
    zf = 0.3
    zr = 1.5
    Sr_Sf = 1
    Af = 8
    Ar = 8

    res = 10
    xfrange = np.linspace(0.5, 2, res)
    xrrange = np.linspace(6, 7, res)
    zfrange = np.linspace(0.3, 1, res)
    zrrange = np.linspace(1, 1.7, res)
    Sr_Sfrange = np.linspace(0.2, 5, res)
    Afrange = np.linspace(1, 12, res)
    Arrange = np.linspace(1, 12, res)
    alpharange = np.linspace(np.deg2rad(1), np.deg2rad(15), res)
    Vrange = np.linspace(35, 70, res)
    Srange = np.linspace(8, 32, res)

    def create_plot(S, xf, xr, zf, zr, Sr_Sf, Af, Ar, xaxis_range,
                    alpha=np.deg2rad(5), V=55):
        lambda_c2f = lambda_c4_to_lambda_c2(Af, taperf, lambda_c4f)
        CLaf = CLa(Claf, Af, lambda_c2f)
        bf, br = bf_br(S, Sr_Sf, Af, Ar)
        crf, crr = crf_crr(S, Sr_Sf, Af, Ar, taperf, taperr)

        emp = deps_da_emp(lambda_c4f, bf, xr - xf, zr - zf, Af, CLaf)

        ll_list = []
        for i in range(res):
            bf_pt = bf if not isinstance(bf, np.ndarray) else bf[i]
            Af_pt = Af if not isinstance(Af, np.ndarray) else Af[i]
            crf_pt = crf if not isinstance(crf, np.ndarray) else crf[i]
            zf_pt = zf if not isinstance(zf, np.ndarray) else zf[i]
            zr_pt = zr if not isinstance(zr, np.ndarray) else zr[i]
            xf_pt = xf if not isinstance(xf, np.ndarray) else xf[i]
            xr_pt = xr if not isinstance(xr, np.ndarray) else xr[i]
            br_pt = br if not isinstance(br, np.ndarray) else br[i]
            crr_pt = crr if not isinstance(crr, np.ndarray) else crr[i]
            alpha_pt = alpha if not isinstance(alpha, np.ndarray) else alpha[i]
            V_pt = V if not isinstance(V, np.ndarray) else V[i]

            ll = downwash(bf_pt, Af_pt, crf_pt, crf_pt * taperf,
                          lambda_c4f, alpha_pt, zr_pt - zf_pt, xr_pt - xf_pt,
                          br_pt, crr_pt, crr_pt * taperr, lambda_c4r, V_pt)
            ll_list.append(ll)

        if isinstance(emp, np.ndarray):
            plt.plot(xaxis_range, emp, label="empirical", marker="o")
        else:
            plt.axhline(emp, label="empirical")

        plt.plot(xaxis_range, ll_list, label="lifting line",
                 color="tab:orange", marker="o")
        plt.legend()

    plt.xlabel(r"$x_f$")
    plt.ylabel(r"$d\varepsilon/d\alpha$")
    create_plot(S, xfrange, xr, zf, zr, Sr_Sf, Af, Ar, xfrange)
    plt.figure()
    plt.xlabel(r"$x_r$")
    plt.ylabel(r"$d\varepsilon/d\alpha$")
    create_plot(S, xf, xrrange, zf, zr, Sr_Sf, Af, Ar, xrrange)
    plt.figure()
    plt.xlabel(r"$z_f$")
    plt.ylabel(r"$d\varepsilon/d\alpha$")
    create_plot(S, xf, xr, zfrange, zr, Sr_Sf, Af, Ar, zfrange)
    plt.figure()
    plt.xlabel(r"$z_r$")
    plt.ylabel(r"$d\varepsilon/d\alpha$")
    create_plot(S, xf, xr, zf, zrrange, Sr_Sf, Af, Ar, zrrange)
    plt.figure()
    plt.xlabel(r"$S_r/S_f$")
    plt.ylabel(r"$d\varepsilon/d\alpha$")
    create_plot(S, xf, xr, zf, zr, Sr_Sfrange, Af, Ar, Sr_Sfrange)
    plt.figure()
    plt.xlabel(r"$A_f$")
    plt.ylabel(r"$d\varepsilon/d\alpha$")
    create_plot(S, xf, xr, zf, zr, Sr_Sf, Afrange, Ar, Afrange)
    plt.figure()
    plt.xlabel(r"$A_r$")
    plt.ylabel(r"$d\varepsilon/d\alpha$")
    create_plot(S, xf, xr, zf, zr, Sr_Sf, Af, Arrange, Arrange)
    plt.figure()
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$d\varepsilon/d\alpha$")
    create_plot(S, xf, xr, zf, zr, Sr_Sf, Af, Ar, alpharange, alpha=alpharange)
    plt.figure()
    plt.xlabel(r"$V$")
    plt.ylabel(r"$d\varepsilon/d\alpha$")
    create_plot(S, xf, xr, zf, zr, Sr_Sf, Af, Ar, Vrange, V=Vrange)
    plt.figure()
    plt.xlabel(r"$S$")
    plt.ylabel(r"$d\varepsilon/d\alpha$")
    create_plot(Srange, xf, xr, zf, zr, Sr_Sf, Af, Ar, Srange)

    plt.show()


if __name__ == "__main__":
    # taperf = 0.45
    # taperr = 0.45
    # lambda_c4f = 0
    # lambda_c4r = 0
    # alpha = np.deg2rad(5)  # value doesn't matter
    # V = 55  # value doesn't matter
    #
    # dxrange = np.linspace(2, 9, 2)
    # dzrange = np.linspace(0.5, 2.5, 2)
    # Srange = np.linspace(7, 35, 3)
    # Sr_Sfrange = np.linspace(0.3, 3, 4)
    # Afrange = np.linspace(2, 14, 3)
    # Arrange = np.linspace(2, 14, 3)
    #
    # de_da_compute(taperf, taperr, lambda_c4f, lambda_c4r, alpha, V, dxrange,
    #               dzrange, Srange, Sr_Sfrange, Afrange, Arrange,
    #               "downwash_interp_1506_1431")

    compare_ll_interp("downwash_interp_1506_1431")



