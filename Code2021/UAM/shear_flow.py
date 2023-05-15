def N_xy(b, c_r, t, t_sp, t_rib, L, b_st, h_st):
    h1 = height(b, c_r)
    ch = chord(b, c_r)
    tarr = t_arr(b, t, L)
    sta = rib_coordinates(b, L)
    shear = shear_eng(b, c_r, t, t_sp, t_rib, L, b_st, h_st)[1]
    Vz = np.zeros(len(tarr))
    for i in range(len(tarr)):
        Vz[i] = shear[2 * i]

    Nxy = np.zeros(len(tarr))

    for i in range(len(tarr)):
        Ixx1 = I_xx(b, tarr[i], c_r, h_st, b_st)
        Ixx = Ixx1(sta[i])
        h = h1(sta[i])
        l_sk = sqrt(h ** 2 + (0.25 * c_r) ** 2)
        c = ch(sta[i])

        # Base region 1
        qb1 = lambda z: Vz[i] * tarr[i] * (0.5 * h) ** 2 * (np.cos(z) - 1) / Ixx
        I1 = qb1(pi / 2)

        # Base region 2
        qb2 = lambda z: -Vz[i] * t_sp * z ** 2 / (2 * Ixx)
        I2 = qb2(h)
        s2 = np.arange(0, h+ 0.1, 0.1)

        # Base region 3
        qb3 = lambda z: - Vz[i] * tarr[i] * (0.5 * h) * z / Ixx + I1 + I2
        I3 = qb3(0.6 * c)
        s3 = np.arange(0, 0.6*c+ 0.1, 0.1)

        # Base region 4
        qb4 = lambda z: -Vz[i] * t_sp * z ** 2 / (2 * Ixx)
        I4 = qb4(h)
        s4=np.arange(0, h+ 0.1, 0.1)

        # Base region 5
        qb5 = lambda z: -Vz[i] * tarr[i] / Ixx * (0.5 * h * z - 0.5 * 0.5 * h * z ** 2 / l_sk) + I3 + I4
        I5 = qb5(l_sk)

        # Base region 6
        qb6 = lambda z: Vz[i] * tarr[i] / Ixx * 0.5 * 0.5 * h / l_sk * z ** 2 + I5
        I6 = qb6(l_sk)

        # Base region 7
        qb7 = lambda z: -Vz[i] * t_sp * 0.5 * z ** 2 / Ixx
        I7 = qb7(-h)


        # Base region 8
        qb8 = lambda z: -Vz[i] * 0.5 * h * t_sp * z / Ixx + I6 - I7
        I8 = qb8(0.6 * c)

        # Base region 9
        qb9 = lambda z: -Vz[i] * 0.5 * t_sp * z ** 2 / Ixx
        I9 = qb9(-h)

        # Base region 10
        qb10 = lambda z: -Vz[i] * tarr[i] * (0.5 * h) ** 2 * (np.cos(z) - 1) / Ixx + I8 - I9
        
        # Redundant shear flow
        A11 = pi * (0.5 * h) / tarr[i] + h / t_sp
        A12 = -h / t_sp
        A21 = - h / t_sp
        A22 = 1.2 * c / tarr[i]
        A23 = -h / t_sp
        A32 = - h / t_sp
        A33 = 2 * l_sk / tarr[i] + h / t_sp

        B1 = 0.5 * h / tarr[i] * quad(qb1, 0, pi / 2)[0] + quad(qb2, 0, 0.5 * h)[0] / t_sp - quad(qb9, -0.5 * h, 0)[
            0] / t_sp + quad(qb10, -pi / 2, 0)[0] * 0.5 * h / tarr[i]
        B2 = quad(qb2, 0, 0.5 * h)[0] / t_sp + quad(qb3, 0, 0.6 * c)[0] / tarr[i] - quad(qb7, -0.5 * h, 0)[0] / t_sp + \
             quad(qb4, 0, 0.5 * h)[0] / t_sp + quad(qb8, 0, 0.6 * c)[0] / tarr[i] - quad(qb9, -0.5 * h, 0)[0] / t_sp
        B3 = quad(qb5, 0, l_sk)[0] / tarr[i] + quad(qb6, 0, l_sk)[0] / tarr[i] + quad(qb4, 0, 0.5 * h)[0] / t_sp - \
             quad(qb9, -0.5 * h, 0)[0] / t_sp

        A = np.array([[A11, A12, 0], [A21, A22, A23], [0, A32, A33]])
        B = -np.array([[B1], [B2], [B3]])
        X = np.linalg.solve(A, B)

        q01 = float(X[0])
        q02 = float(X[1])
        q03 = float(X[2])

        # Compute final shear flow
        q2 = qb2(s2) - q01 + q02
        q3 = qb3(s3) + q02
        q4=qb4(s4)+q03-q02

        max_region2 = max(q2)
        max_region3 = max(q3)
        max_region4 = max(q4)
        determine = max(max_region2, max_region3, max_region4)
        Nxy[i] = determine
    return Nxy
