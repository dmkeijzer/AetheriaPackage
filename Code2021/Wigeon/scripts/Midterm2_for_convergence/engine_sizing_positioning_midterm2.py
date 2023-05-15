import numpy as np


class PropSizing:
    def __init__(self, wing_span, fuselage_width, N_eng, clearance_fus_prop, clearance_prop_prop, MTOM, xi_0):
        """
        :param wing_span: FUll span of one wing (TIP TO TIP) [m]
        :param fuselage_width: Width of the WHOLE fuselage (not half) [m]
        :param N_eng: Number of propellers in total [-]
        :param clearance_fus_prop: horizontal distance between propeller and fuselage [m]
        :param clearance_prop_prop: Horizontal distance between propeller and propeller
        :param MTOM: Maximum take-off mass [kg]
        :param xi_0: Hub radius ratio (r_hub/R)
        """
        self.b = wing_span
        self.wf = fuselage_width
        self.N_tot = N_eng
        self.N_per_half_wing = N_eng/4
        self.c_fp = clearance_fus_prop
        self.c_pp = clearance_prop_prop
        self.MTOM = MTOM
        self.xi_0 = xi_0

    def radius(self):
        """
        Size the radius based on the span, assuming one tip mounted engine
        """
        return (self.b/2 - self.wf/2 - self.c_fp - (self.N_per_half_wing - 1)*self.c_pp) / \
               (2*(self.N_per_half_wing - 1) + 1)

    def diameter(self):
        return 2*self.radius()

    def area_prop(self):
        return np.pi*(self.radius()**2 - (self.xi_0*self.radius())**2)

    def disk_loading(self):
        """
        Calculate disk loading in hover based on propeller area and MTOM
        Calculate total area and divide mass by that

        :return: Disk loading in kg/m^2
        """
        area_tot = self.N_tot * self.area_prop()
        return self.MTOM/area_tot
