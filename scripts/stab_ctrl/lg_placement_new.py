import numpy as np
x_lg_fwd_arr=np.linspace(0,3,300)
x_lg_main_arr=np.linspace(6,8,200)  ##Make sure this maximum is less than x_corner and more than x_cg_aft
h_lg_strut_arr=np.linspace(0.2,1,80) 


tire_radius=0.37/2
Dp=2.1
x_corner=8
x_cg_aft=5.7105
x_cg_fwd=5.3084
z_cg=0.6*2
fuselage_width=1.8
h_engine_inner=1.8/2
h_engine_outer=1.8/2
y_engine_inner=2.6202
y_engine_outer=4.7977
y_additional=0.385  ##0.38 does not work.
tipback=15*np.pi/180
x_min_for_front_lg=0.144


for x_lg_fwd in x_lg_fwd_arr:
    for x_lg_main in x_lg_main_arr:
        for h_lg_strut in h_lg_strut_arr:

            err=0
            
            weight_on_fwd_lg_max= (x_lg_main-x_cg_aft)/(x_lg_main-x_lg_fwd)
            weight_on_fwd_lg_min=  (x_lg_main-x_cg_fwd)/(x_lg_main-x_lg_fwd)

            #print(weight_on_fwd_lg_max, weight_on_fwd_lg_min)
            if weight_on_fwd_lg_max>0.15:
                err=1
            if weight_on_fwd_lg_min<0.08:
                err=1
            if x_lg_main>x_corner:
                err=1
            
            if x_cg_aft+np.tan(tipback)*(z_cg+h_lg_strut)>x_lg_main:    ###tipback during landing
                err=1
            
            if ((y_engine_outer-fuselage_width/2-y_additional)*np.tan(5*np.pi/180))>(h_lg_strut+h_engine_outer-Dp/2):   ##engine touch
                err=1
            
            if ((y_engine_inner-fuselage_width/2-y_additional)*np.tan(8*np.pi/180))>(h_lg_strut+h_engine_inner-Dp/2): ##engine touch
                err=1
            
            alpha=np.arctan((fuselage_width/2+y_additional)/(x_lg_main-x_lg_fwd))
            c=np.sin(alpha)*(x_cg_aft-x_lg_fwd)
            h_lg_strut_max=np.tan(55*np.pi/180)*c-z_cg

            if h_lg_strut_max<h_lg_strut:                              ####turnover 
                err=1

            if err==0:
                print(h_lg_strut, x_lg_fwd, x_lg_main)
                
    
