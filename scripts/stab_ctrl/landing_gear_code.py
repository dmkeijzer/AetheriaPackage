import numpy as np
def place_gear(tire_radius,Dp,x_corner,x_cg_aft,x_cg_fwd,z_cg,fuselage_width,h_engine_inner,h_engine_outer,y_engine_inner,y_engine_outer,y_additional=0,tipback=15*np.pi/180,landing_gear_strut_max_limit=0.6,x_min_for_front_lg=0.144):
    x_lg_main=x_cg_aft+(landing_gear_strut_max_limit+z_cg)*np.tan(tipback)    
    if x_lg_main>x_corner:
        print("The main landing gear is way too aft. Fuselage needs to be extended. Any result is invalid.")
    
    ###the minimum weight on the the front lg happens when the cg is most aft and the front lg is as far forward 
    x_lg_fwd_min=-(0.92*(x_lg_main-x_cg_aft)/0.08-x_cg_aft)
    x_lg_fwd_max=-(0.85*(x_lg_main-x_cg_fwd)/0.15-x_cg_fwd)
    print(x_lg_fwd_min, x_lg_fwd_max)
    if x_lg_fwd_max<x_lg_fwd_min:
        print("Problem positioning front landing gear due to loading requirement on front landing gear")
        
    elif x_lg_fwd_min<x_min_for_front_lg:
        x_lg_fwd=x_min_for_front_lg
    else:
        x_lg_fwd=x_lg_fwd_min   ##Place front landing gear at most forward to make sure tipback is not sure

    ####Now determine landing gear height assuming y_lg_main=fuselage width/2:

    h_lg_strut_min1= np.tan(tipback)*(x_corner-x_lg_main) ##h_lg_min is based on tipback
    
    ###H_engine_min=(y_engine-y_lg)*tan(PSI)
    h_lg_strut_min2= (fuselage_width/2+y_additional)*np.tan(5*np.pi/180)-h_engine_outer+Dp/2   #based on ground constraint of outer propeller
    h_lg_strut_min3= (fuselage_width/2+y_additional)*np.tan(8*np.pi/180)-h_engine_inner+Dp/2    #based on ground constraint of inner propeller
    
    h_lg_strut_min=max(h_lg_strut_min1,h_lg_strut_min2,h_lg_strut_min3)

    ####determine h_lg_strutmax1 height from turnover constraint:
    
    alpha=np.arctan((fuselage_width/2+y_additional)/(x_lg_main-x_lg_fwd))
    c=np.sin(alpha)*(x_cg_aft-x_lg_fwd)
    h_lg_strut_max=np.tan(55*np.pi/180)*c-z_cg
    print(h_lg_strut_min)
    print(h_lg_strut_max)
    if h_lg_strut_min>h_lg_strut_max:
        print("Landing gear cannot be placed")

    elif h_lg_strut_min>landing_gear_strut_max_limit:
        print("Required landing gear height is too high")

    elif h_lg_strut_min<tire_radius:
         h_lg_strut=tire_radius+0.1
    else:
         h_lg_strut=h_lg_strut_min  ##to avoid buckling.
    
    return h_lg_strut, x_lg_fwd, x_lg_main  

h_lg_strut, x_lg_fwd, x_lg_main= place_gear(tire_radius=0.37/2,Dp=2.1,x_corner=7,x_cg_aft=5.7105,x_cg_fwd=5.3084,z_cg=0.6*2,fuselage_width=1.8/2,h_engine_inner=1.8,h_engine_outer=1.8,y_engine_inner=2.6202,y_engine_outer=4.7977,y_additional=0,tipback=15*np.pi/180,landing_gear_strut_max_limit=0.7)

        
    
    
