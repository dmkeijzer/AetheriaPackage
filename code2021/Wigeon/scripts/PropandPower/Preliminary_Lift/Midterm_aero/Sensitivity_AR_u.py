import matplotlib.pyplot as plt

from Preliminary_Lift.Drag import *
import numpy as np
import os
import json
root_path = os.path.join(os.getcwd(), os.pardir)

AR12 = np.linspace(5,9,41)
AR3 = np.linspace(7,13,61)
C_L = np.linspace(0, 1.5, 15001)
LD1 = []
LD2 = []
LD3 = []
datafile1 = open(os.path.join(root_path, "data/inputs_config_1.json"), "r")
data1 = json.load(datafile1)
datafile1.close()
AE1 = data1["Aerodynamics"]
CLmin1 = AE1["CLforCDmin"]
CDmin1 = AE1["CDmin"]

datafile2 = open(os.path.join(root_path, "data/inputs_config_2.json"), "r")
data2 = json.load(datafile2)
datafile2.close()
AE2 = data2["Aerodynamics"]
CLmin2 = AE2["CLforCDmin"]
CDmin2 = AE2["CDmin"]

datafile3 = open(os.path.join(root_path, "data/inputs_config_3.json"), "r")
data3 = json.load(datafile3)
datafile3.close()
AE3 = data3["Aerodynamics"]
CLmin3 = AE3["CLforCDmin"]
CDmin3 = AE3["CDmin"]

ulst = np.linspace(0,45,46)

for AR in AR12:
    e_ref = e_OS(AR)
    e1 = e_factor("tandem",0.2,1,e_ref)
    e2 = e_factor("box", 0.2, 1, e_ref)
    print(AR,e2)
    CD1 = CDmin1 + (((C_L - CLmin1) ** 2)/(np.pi * AR * e1))
    CD2 = CDmin2 + (((C_L - CLmin1) ** 2) / (np.pi * AR * e2))
    LD1.append(np.max(C_L/CD1))
    LD2.append(np.max(C_L/CD2))

for AR in AR3:
    e_ref = e_OS(AR)
    CD3 = CDmin3 + (((C_L - CLmin3) ** 2) / (np.pi * AR * e_ref))
    LD3.append(np.max(C_L / CD3))


print(AR12)
#print(e1)
fig , ax = plt.subplots(1,2)
ax[0].plot(AR12, LD1, color = 'blue', alpha = 0.7,label = 'Tandem configuration')
ax[0].plot(AR12, LD2, color = 'red', alpha = 0.7, label = 'Box wing configuration')
ax[0].set_xlabel("AR[-]", fontsize = 16)
ax[0].set_ylabel("Lift over Drag ratio[-]", fontsize = 16)
ax[0].grid()
ax[1].plot(AR3, LD3, color = 'green', alpha = 0.7, label = 'Single wing configuration')
ax[1].set_xlabel("AR[-]", fontsize = 16)
ax[1].set_ylabel("Lift over Drag ratio[-]", fontsize = 16)
plt.grid()
plt.show()



