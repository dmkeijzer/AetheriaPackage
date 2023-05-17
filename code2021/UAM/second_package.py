'''
Computing bending moment
-first due to skin+rib contribution only
-second function is with engine contribution included
'''

def m(b, c_r, t, t_sp, t_rib, L, b_st, h_st):
    f = skin_interpolation(b, c_r, t, t_sp, L, b_st, h_st)
    sta = rib_coordinates(b, L)
    rbw = rib_weight(b, c_r, t_rib)

    f2 = interp1d(sta, f)

    rib_w = np.zeros(len(sta))
    moment = np.zeros(len(sta))

    for i in range(len(rib_w)):
        rib_w[i] = rbw(sta[i])

    for i in range(1, len(sta)):
        cursor = sta[i] * np.ones(len(sta))
        diff = np.subtract(cursor, sta)
        d = [diff > 0]
        diff = diff[d]
        rib_w = np.flip(rib_w)
        l = len(diff)
        rib_w = rib_w[0:l]
        produ = np.multiply(rib_w, diff)
        s = np.sum(produ)
        f3 = quad(f2, 0, diff[0],points=sta, limit=300)[0]
        moment[i] = 9.81 * f3 + 9.81 * s
    moment = np.flip(moment)

    import matplotlib.pyplot as plt
    plt.plot(sta, moment)
    plt.show()

    return moment



def m_eng(b, c_r, t, t_sp, t_rib, L, b_st, h_st):
    moment=m(b, c_r, t, t_sp, t_rib, L, b_st, h_st)
    x = rib_coordinates(b, L)
    f = interp1d(x, moment, kind='quadratic')

    x_engine = np.array([0.5 * b / 4, 0.5 * b / 2, 0.5 * 3 * b / 4])
    x_combi = np.concatenate((x, x_engine))
    x_sort = np.sort(x_combi)

    index1 = np.where(x_sort == 0.5 * 3 * b / 4)
    if len(index1[0]) == 1:
        index1 = int(index1[0])
    else:
        index1 = int(index1[0][0])
    y_new1 = f(x_sort[index1])

    index2 = np.where(x_sort == 0.5 * b / 2)
    if len(index2[0]) == 1:
        index2 = int(index2[0])
    else:
        index2 = int(index2[0][0])
    y_new2 = f(x_sort[index2])

    index3 = np.where(x_sort == 0.5 * b / 4)
    if len(index3[0]) == 1:
        index3 = int(index3[0])
    else:
        index3 = int(index3[0][0])
    y_new3 = f(x_sort[index3])

    y_engine = np.ndarray.flatten(np.array([y_new1, y_new2, y_new3]))
    y_combi = np.concatenate((moment, y_engine))
    y_sort = np.sort(y_combi)
    y_sort = np.flip(y_sort)

    for i in range(int(index1)):
        y_sort[i]=y_sort[i]+9.81*W_eng*(0.5*3*b/4-x_sort[i])
    for i in range(int(index2)):
        y_sort[i]=y_sort[i]+9.81*W_eng*(0.5*2*b/4-x_sort[i])
    for i in range(int(index3)):
        y_sort[i]=y_sort[i]+9.81*W_eng*(0.5*b/4-x_sort[i])

    import matplotlib.pyplot as plt
    plt.plot(x_sort, y_sort)
    plt.show()

    return x_sort,y_sort

'''
Nx and Nxy [N/m]
'''


def N_x(b, c_r, t, t_sp, t_rib, L, b_st, h_st):
    sta = rib_coordinates(b, L)
    moment = m_eng(b, c_r, t, t_sp, t_rib, L, b_st, h_st)[1]
    x_sort= m_eng(b, c_r, t, t_sp, t_rib, L, b_st, h_st)[0]
    h = height(b, c_r)
    tarr = t_arr(b, t, L)
    Nx = np.zeros(len(tarr))

    index1 = np.where(x_sort == 0.5 * 3 * b / 4)
    if len(index1[0]) == 1:
        index1 = int(index1[0])
    else:
        index1 = int(index1[0][0])

    index2 = np.where(x_sort == 0.5 * b / 2)
    if len(index2[0]) == 1:
        index2 = int(index2[0])
    else:
        index2 = int(index2[0][0])

    index3 = np.where(x_sort == 0.5 * b / 4)
    if len(index3[0]) == 1:
        index3 = int(index3[0])
    else:
        index3 = int(index3[0][0])

    moment=np.delete(moment, np.array([index1,index2,index3]))


    for i in range(len(tarr)):
        Ixx = I_xx(b, tarr[i], c_r, h_st, b_st)(sta[i])
        bend_stress = moment[i] * 0.5*h(sta[i]) / Ixx
        Nx[i] = bend_stress* tarr[i]
    return Nx
