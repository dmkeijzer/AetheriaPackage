import Aero_tools as at

RN_spacing = 100000

# Round Reynolds number to 100,000 to retrieve appropriate file from airfoil data folder
RN = RN_spacing * round(1278980 / RN_spacing)
filename1 = "4412_Re%d_up" % RN

print(RN)
print(filename1)
