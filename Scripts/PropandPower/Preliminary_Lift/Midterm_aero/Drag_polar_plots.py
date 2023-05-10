import numpy as np
import matplotlib.pyplot as plt
from Preliminary_Lift.Drag import  C_D, e_factor

C_L = np.linspace(-0.5, 1.5,151)
C_Dmin1 = 0.0327
C_Dmin2 = 0.0322
CD_min3 =0.0285

e_ref = 0.7566
e_tandem = 1.1302
e_box = 1.1738
C_D_n8 = []
C_D_n9 = []
C_D_n10 = []
C_D_t6 = []
C_D_t7 = []
C_D_t8 = []
C_D_b6 = []
C_D_b7 = []
C_D_b8 = []
LoD = []
for i in C_L :
    C_D_n8.append(C_D(i,0.2927, C_Dmin1, 7, e_tandem ))
    C_D_n9.append(C_D(i,0.2927, C_Dmin2, 7, e_box))
    C_D_n10.append(C_D(i,0.2401, CD_min3, 10, e_ref))
    #C_D_t6.append(C_D(i, C_D0, 6, e_tandem))
   # C_D_t7.append(C_D(i, C_D0, 7, e_tandem))
   # C_D_t8.append(C_D(i, C_D0, 8, e_tandem))
    #C_D_b6.append(C_D(i, C_D0, 6, e_box))
    #C_D_b7.append(C_D(i, C_D0, 7, e_box))
    #C_D_b8.append(C_D(i, C_D0, 8, e_box))

#fig , ax = plt.subplots(1,1)
plt.plot(C_D_n8, C_L, color = 'blue', alpha = 0.7,label = 'Tandem configuration')
plt.plot(C_D_n9, C_L, color = 'red', alpha = 0.7, label = 'Box wing configuration')
plt.plot(C_D_n10, C_L, color = 'green', alpha = 0.7, label = 'Flying wing configuration')
plt.grid()
plt.xlabel("$C_D$[-]", fontsize = 20)
plt.xticks(fontsize= 18)
plt.yticks(fontsize= 18)
plt.ylabel("$C_L[-]$", fontsize = 20)
"""
ax[0,1].plot(C_D_t6, C_L, color = 'blue', label = 'Tandem A = 6')
ax[0,1].plot(C_D_t7, C_L, color = 'blue', alpha = 0.7,  label = 'Tandem A = 7')
ax[0,1].plot(C_D_t8, C_L, color = 'blue', alpha = 0.4, label = 'Tandem A = 8')
ax[1,0].plot(C_D_b6, C_L, color = 'green',  label = 'Box Wing A = 6')
ax[1,0].plot(C_D_b7, C_L, color = 'green', alpha = 0.7, label = 'Box Wing A = 7')
ax[1,0].plot(C_D_b8, C_L, color = 'green', alpha = 0.4, label = 'Box Wing A = 8')
ax[1,1].plot(C_D_n8, C_L, color = 'red', label = 'Wing A = 8')
ax[1,1].plot(C_D_t8, C_L, color = 'blue', alpha = 0.4, label = 'Tandem A = 8')
ax[1,1].plot(C_D_b8, C_L, color = 'green', alpha = 0.4, label = 'Box Wing A = 8')
"""
plt.legend(fontsize = 18)

plt.show()

