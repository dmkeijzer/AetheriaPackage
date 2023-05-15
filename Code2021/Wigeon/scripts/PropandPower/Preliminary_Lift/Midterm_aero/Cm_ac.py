import numpy as np

EPPLER_Cm= np.array([ 0.0494, 0.0457, 0.0421])
EPPLER_CN = np.array([0.2410*np.cos(5*np.pi/180),0.5340*np.cos(5*np.pi/180), 0.8182*np.cos(7.5*np.pi/180)])


CNda1 = (EPPLER_CN[1]-EPPLER_CN[0])/2.5
CNda2 = (EPPLER_CN[2]-EPPLER_CN[1])/2.5
CN2da2 = (CNda2-CNda1)/2.5
Cmda1 = (EPPLER_Cm[1]-EPPLER_Cm[0])/2.5
Cmda2 = (EPPLER_Cm[2]-EPPLER_Cm[1])/2.5
Cm2da2 = (Cmda2-Cmda1)/2.5
xn1 = (0.25*CNda1-Cmda1)/CNda1
xn2 = (0.25*CNda2-Cmda2)/CNda2
x2 = (0.25*CN2da2-Cm2da2)/CN2da2
print(xn1,xn2, x2)
x_ac = (xn1+xn2)/2

cm_ac = EPPLER_Cm[0] + EPPLER_CN[0]*(0.25-x_ac)
print(cm_ac)