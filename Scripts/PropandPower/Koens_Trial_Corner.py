import numpy as np
import matplotlib.pyplot as plt

# N = 10
# C_array = np.linspace(1,6000000,40)
# print(C_array)
#
# def C_sum(C_in, C_add):
#     return 1 / (1/C_in + 1/C_add)
#
# C_Sum = C_array[0]
# sum_list = list()
# i_list = list()
# counter = 0
# for i in C_array:
#     counter += 1
#     C_Sum = C_sum(C_Sum,i)
#     sum_list.append(C_Sum)
#     i_list.append(counter)
#
# print(sum_list)
#
# plt.plot(i_list,sum_list)
# plt.show()

# Nc = 100
#
# DoD_list = list()
# count_list = list()
# count = 0
# for i in range(1, Nc):
#     count += 1
#     DoD = 175 - 30 * np.log10(i)
#     count_list.append(count)
#     DoD_list.append(DoD)
#
# plt.plot(count_list,DoD_list)
# plt.show()
#
# print(DoD_list)

# def closestNumber(n, m):
#     # Find the quotient
#     q = int(n / m)
#
#     # 1st possible closest number
#     n1 = m * q
#
#     # 2nd possible closest number
#     if ((n * m) > 0):
#         n2 = (m * (q + 1))
#     else:
#         n2 = (m * (q - 1))
#
#     # if true, then n1 is the required closest number
#     if n1 > n:
#         return n1
#     else:
#         return n2
#
# print(closestNumber(13,4))


def closestNumber(n, m):
    # Find the quotient
    q = int(n / m)

    # 1st possible closest number
    n1 = m * q

    # 2nd possible closest number
    if ((n * m) > 0):
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))

    # if true, then n1 is the required closest number
    if (abs(n - n1) < abs(n - n2)):
        return n1

    # else n2 is the required closest number
    return n2

print(closestNumber(535,24))