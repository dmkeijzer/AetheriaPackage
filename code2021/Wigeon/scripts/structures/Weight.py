""" New weight estimation file """
import sys
import numpy as np

sys.path.append('../Final_optimization/')
import constants_final as const

# from Final_optimization import constants_final as const
class Vtail:
    def __init__(self, mtom, Sv, Av, rchord, toc, sweep_deg):
        self.mtom_lbs = 2.20462 * mtom
        self.Sv_ft = Sv * 3.28084 ** 2
        self.Av = Av
        self.trv = rchord * toc * 3.28084
        self.sweep = sweep_deg * np.pi/180
        self.mass = ((1.68 * self.mtom_lbs ** 0.567 * self.Sv_ft ** 1.249 * self.Av ** 0.482)/(639.95 * self.trv ** 0.747 * np.cos(self.sweep)**0.882)) * 0.453592


class Wing:
    # Roskam method (not accurate because does not take into account density of material but good enough for comparison
    def __init__(self, mtom, S1, S2, n_ult, A_1, A_2, pos=[], wmac = 0.8, toc = 0.17):
        self.S1_ft, self.S2_ft, self.S1, self.S2 = S1 * 3.28084 ** 2, S2 * 3.28084 ** 2, S1, S2
        self.n_ult = n_ult
        self.A_1, self.A_2 = A_1, A_2
        self.mtow_lbs = 2.20462 * mtom
        self.pos1, self.pos2 = pos
        self.wweight1 = 0.04674*((self.mtow_lbs/2)**0.397)*(self.S1_ft**0.36)*(self.n_ult**0.397)*(self.A_1**1.712)*0.453592
        self.wweight2 = 0.04674*((self.mtow_lbs/2)**0.397)*(self.S2_ft**0.36)*(self.n_ult**0.397)*(self.A_2**1.712)*0.453592
        self.mass = np.array([self.wweight1, self.wweight2])
        self.moment = np.array(self.mass)*np.array(pos)
        self.wmac, self.toc = wmac, toc

class Fuselage:
    # Roskam method (not accurate because does not take into account density of material but good enough for comparison
    def __init__(self, mtom, Pmax, lf, npax, pos, wf=1.55):
        self.mtow_lbs = 2.20462 * mtom
        self.lf_ft, self.lf = lf*3.28084, lf
        self.Pmax_ft = Pmax*3.28084
        self.pos = pos
        self.npax = npax
        self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.Pmax_ft)**0.778)*(self.lf_ft**0.383)*(self.npax**0.455)
        self.fweight_low = 0.04682*(self.mtow_lbs**0.692)*(self.Pmax_ft**0.379)*(self.lf_ft**0.590)
        self.fweight = (self.fweight_high + self.fweight_low)/2
        self.mass = self.fweight*0.453592
        self.moment = self.mass * self.pos
        self.wf = wf

class LandingGear:
    def __init__(self, mtom, pos):
        self.mass = 0.04*mtom
        self.pos = pos
        self.moment = self.mass * self.pos

class Propulsion:

    def __init__(self, n_prop, m_prop = [], pos_prop = []):
        self.nprop = n_prop
        self.wprop = np.array(m_prop)
        self.pos_prop = np.array(pos_prop)
        self.moment_prop = self.wprop*self.pos_prop
        self.mass = np.sum(self.wprop)
        self.moment = np.sum(self.wprop*self.pos_prop)

class Weight:

    def __init__(self, m_pax, wing, fuselage, landing_gear, propulsion, cargo_m, cargo_pos, battery_m, battery_pos, p_pax = [],
                 contingency = False, Vtail_mass = 1, vtail_pos = 1):
        self.m_pax, self.p_pax = m_pax, p_pax
        self.wing, self.fuselage, self.landing_gear, self.prop = wing, fuselage, landing_gear, propulsion
        # weights of components
        self.tot_m_pax = self.m_pax * 5
        self.wmass, self.fmass, self.lmass, self.pmass = np.sum(self.wing.mass), self.fuselage.mass, self.landing_gear.mass, self.prop.mass
        self.cmass, self.cpos = cargo_m, cargo_pos
        self.bmass, self.battery_pos = battery_m, battery_pos
        self.vmass, self.vpos = Vtail_mass, vtail_pos
        # moments of components
        self.moment_pax = np.sum(self.m_pax * np.array(self.p_pax))
        self.moment_w = np.sum(np.array(self.wing.moment))
        self.moment_f = self.fuselage.moment
        self.moment_l = self.landing_gear.moment
        self.moment_p = self.prop.moment
        self.moment_c = self.cmass * self.cpos
        self.moment_b = self.bmass * self.battery_pos
        self.moment_v = self.vmass * self.vpos
        # operational empty mass centre of gravity
        self.oem_cg = (self.moment_w + self.moment_f + self.moment_l + self.moment_p + self.moment_b + self.moment_v) \
        /(self.wmass + self.pmass + self.lmass + self.fmass + self.bmass + self.vmass)

        # masses
        self.oem = (self.wmass + self.pmass + self.lmass + self.fmass + self.bmass + self.vmass)

#         print("Check inside function:", self.wmass, self.pmass, self.lmass, self.fmass, self.cmass, self.bmass, self.tot_m_pax)
#         print("")

        if contingency:
            # self.mtom = (self.wmass*const.mass_cont + self.pmass*const.mass_cont + self.lmass*const.mass_cont +
            #              self.fmass*const.mass_cont + self.cmass + self.bmass*const.mass_cont + self.tot_m_pax + self.vmass * const.mass_cont)
            # self.mtom_cg = (self.moment_w*const.mass_cont + self.moment_f*const.mass_cont + self.moment_l*const.mass_cont +
            #                 self.moment_p*const.mass_cont + self.moment_pax + self.moment_c + self.moment_b*const.mass_cont + self.moment_v * const.mass_cont) \
            #                / (self.wmass*const.mass_cont + self.pmass*const.mass_cont + self.lmass*const.mass_cont +
            #                   self.fmass*const.mass_cont + self.cmass + self.bmass*const.mass_cont + self.tot_m_pax + self.vmass * const.mass_cont)


            self.mtom = (self.wmass*const.mass_cont + self.pmass*const.mass_cont + self.lmass*const.mass_cont +
                         self.fmass*const.mass_cont + self.cmass + self.bmass*const.mass_cont + self.tot_m_pax + self.vmass*const.mass_cont)
            self.mtom_cg = (self.moment_w*const.mass_cont + self.moment_p*const.mass_cont + self.moment_l*const.mass_cont + self.moment_f*const.mass_cont + self.moment_c + self.moment_b*const.mass_cont +self.moment_pax  +  self.moment_v*const.mass_cont) \
                           /self.mtom #(self.wmass + self.pmass + self.lmass + self.fmass + self.cmass + self.bmass + self.tot_m_pax + self.vmass)
            print('testing', self.moment_l)#(self.moment_w*const.mass_cont + self.moment_p*const.mass_cont + self.moment_l*const.mass_cont + self.moment_f*const.mass_cont + self.moment_c + self.moment_b*const.mass_cont +self.moment_pax  +  self.moment_v*const.mass_cont))
        else:
            self.mtom = (self.wmass + self.pmass + self.lmass +
                         self.fmass + self.cmass + self.bmass + self.tot_m_pax + self.vmass)
            self.mtom_cg = (self.moment_w + self.moment_p + self.moment_l + self.moment_f + self.moment_c + self.moment_b + self.moment_pax + self.moment_v) \
                           / self.mtom # (self.wmass + self.pmass + self.lmass + self.fmass + self.cmass + self.bmass + self.tot_m_pax + self.vmass)
            print('testing NC', self.moment_l)#(self.moment_w + self.moment_p + self.moment_l + self.moment_f + self.moment_c + self.moment_b + self.moment_pax + self.moment_v))

            # self.moment_p: 1480 vs 17767
    def print_weight_fractions(self):
        d = {}
        d["Front wing"] = [self.wing.mass[0], self.wing.mass[0]/self.oem, self.wing.mass[0]/self.mtom]
        d["Back wing"] = [self.wing.mass[1], self.wing.mass[1]/self.oem, self.wing.mass[1]/self.mtom]
        d['Fuselage'] = [self.fmass, self.fmass/self.oem, self.fmass/self.mtom]
        d['Landing gear'] = [self.lmass, self.lmass/self.oem, self.lmass/self.mtom]
        d['Vertical tail'] = [self.vmass, self.vmass/self.oem, self.vmass/self.mtom]
        d['Propulsion'] = [self.pmass, self.pmass/self.oem, self.pmass/self.mtom]
        d['Cargo'] = [self.cmass, 0.0, self.cmass/self.mtom]
        d['Battery'] = [self.bmass, self.bmass/self.oem, self.bmass/self.mtom]
        d['Payload'] = [self.tot_m_pax, 0.0, self.tot_m_pax/self.mtom]

        print("{:<15} {:<20} {:<25} {:<15}".format('Component', 'Mass[kg]', 'fraction of OEM', 'fraction of MTOM'))
        print('--------------------------------------------------------------------------------')
        for k, v in d.items():
            mass, oem_frac, mtom_frac = v
            print("{:<15} {:<20} {:<25} {:<15}".format(k, mass, oem_frac, mtom_frac))
        print('')
        print(f'Where OEM is {self.oem}kg with CG of {self.oem_cg}m, and MTOM is {self.mtom}kg with CG of {self.mtom_cg}m')
        # for key in d:
        #     d[key] = {k: list(i) if isinstance(i, np.ndarray) else i for k, i in zip(["mass", "fracOEM", "fracEM"], d[key])}
        # return d
    def MMI(self, wmac = [], toc = [], vpos_wing = []):
        # COORDINATE SYSTEM: x points towards nose, y points towards right wing, z points down
        # fuselage  - modeled as a hollow cylinder with wall thickness of 5 cm
        lf = self.fuselage.lf
        irad = (self.fuselage.wf/2 - 0.05)
        fus_mmi_y = self.fmass * (self.fuselage.lf**2 + 3*(self.fuselage.wf/2)**2 + 3*irad**2)/12
        fus_mmi_z = fus_mmi_y
        fus_mmi_x = self.fmass * ((self.fuselage.wf/2)**2 + irad**2)/2

        # front wing - modeled as a prism span, average thickness and width at mac
        vpos = vpos_wing
        t1 = wmac[0] * toc[0]
        span1 = np.sqrt(self.wing.A_1 * self.wing.S1)

        wing1_mmi_x, wing1_mmi_y, wing1_mmi_z = self.wing.mass[0] * (span1 ** 2 + t1 ** 2) / 12, self.wing.mass[0] \
                                             * (span1 ** 2 + wmac[0] ** 2) / 12, \
                                             self.wing.mass[0] * (wmac[0] ** 2 + t1 ** 2) / 12

        # back wing - modeled as a prism with span, average thickness and width at mac
        t2 = wmac[1] * toc[1]
        span2 = np.sqrt(self.wing.A_2 * self.wing.S2)

        wing2_mmi_x, wing2_mmi_y, wing2_mmi_z = self.wing.mass[1]*(span2**2 + t2**2)/12, self.wing.mass[1]*(span2**2 + wmac[1]**2)/12, \
                                             self.wing.mass[1] * (wmac[1] ** 2 + t2 ** 2) / 12

        # passengers - modeled as a prism 2.2 x 1.1 x 1 m (length x width x height)
        pld_mmi_x, pld_mmi_y, pld_mmi_z = self.tot_m_pax * (1.1**2 + 1**2)/12, self.tot_m_pax*(1.1**2 + 2.2**2)/12, self.tot_m_pax*(2.2**2 + 1**2)


        # propulsion - modeled as a solid cylinder
        m_prop = self.pmass/self.prop.nprop
        lprop, rprop = 0.4, 0.12
        prop_mmi_x, prop_mmi_y, prop_mmi_z = m_prop*(rprop**2)/2, m_prop*(lprop**2 + 3 * rprop**2)/12, m_prop*(lprop**2 + 3 * rprop**2)/12

        # battery - modeled as a prism
        lbat, tbat, wbat = 0.4*self.fuselage.lf, 0.2, 1
        bat_mmi_x, bat_mmi_y, bat_mmi_z = self.bmass*(wbat**2 + tbat**2)/12, self.bmass*(wbat**2 + lbat**2)/12, self.bmass*(tbat**2 + lbat**2)/12
        span = span2
        oem_mmi_x = fus_mmi_x + (
                    wing1_mmi_x + self.wing.mass[0] * (0.7 - vpos[0])**2) + (wing2_mmi_x + self.wing.mass[1] * (vpos[1] - 0.7)**2) + \
                    4 * np.sum(m_prop * (np.sqrt((0.7)**2 +  (np.linspace(0.9, span1/2, int(self.prop.nprop/4))) ** 2 ))** 2) + \
                                         self.prop.nprop * prop_mmi_x + bat_mmi_x + pld_mmi_x

        oem_mmi_y = fus_mmi_y + (wing1_mmi_y + self.wing.mass[0]*(self.mtom_cg - self.wing.pos1)**2) + (wing2_mmi_y + self.wing.mass[1]*(self.wing.pos2 - self.mtom_cg)**2)+\
                    4 * np.sum(m_prop * (np.sqrt((self.wing.pos2 - 0.7)**2 + (np.linspace(0.9, span/2, int(self.prop.nprop/4))) ** 2))**2) +\
                                                         self.prop.nprop * prop_mmi_y + bat_mmi_y + self.bmass * (self.mtom_cg - self.battery_pos)**2 + pld_mmi_y + self.tot_m_pax*((lf/2 - 0.5) - self.mtom_cg)**2

        oem_mmi_z = fus_mmi_z + (wing1_mmi_z + self.wing.mass[0] * (
            np.sqrt((0.7 - vpos[0]) ** 2 + (self.mtom_cg - self.wing.pos1) ** 2)) ** 2) + (wing2_mmi_z + self.wing.mass[1] * (
            np.sqrt((vpos[1] - 0.7) ** 2 + (self.wing.pos2 - self.mtom_cg) ** 2)) ** 2) + bat_mmi_z + self.bmass * (self.mtom_cg - self.battery_pos)**2 + (
                                m_prop * ((np.sqrt((0.7 - vpos[0]) ** 2 + (self.mtom_cg - self.wing.pos1) ** 2)) ** 2) + prop_mmi_z) * self.prop.nprop/2 + \
                    pld_mmi_z + self.tot_m_pax*((lf/2 - 0.5) - self.mtom_cg)**2 + \
                    (m_prop * ((np.sqrt((vpos[1] - 0.7) ** 2 + (self.wing.pos2 - self.mtom_cg) ** 2)) ** 2) + prop_mmi_z) * self.prop.nprop/2


        oem_mmi_xy = (self.wing.mass[0] * (0.7 - vpos[0]) * (self.mtom_cg - self.wing.pos1)) + self.prop.nprop * (m_prop * (0.7-vpos[0]) * (self.mtom_cg-self.wing.pos1))/2 + \
                     (self.wing.mass[1] * (vpos[1] - 0.7) * (self.wing.pos2 - self.mtom_cg)) +self.prop.nprop * (m_prop * (vpos[0]-0.7) * (self.wing.pos2-self.mtom_cg))/2
        return oem_mmi_x, oem_mmi_z, oem_mmi_y, oem_mmi_xy

if __name__ == '__main__':
    mtom = 3000 # maximum take-off mass from statistical data - Class I estimation
    S1, S2 = 9.910670535618632, 9.910670535618632 # surface areas of wing one and two
    A1 = 6.8 # aspect ratio of a wing, not aircraft
    A2 = 6.8  # aspect ratio of a wing, not aircraft
    n_ult = 3.4*1.5 # 3.2 is the max we found, 1.5 is the safety factor
    Pmax = 17 # this is defined as maximum perimeter in Roskam, so i took top down view of the fuselage perimeter
    lf = 7.379403359777299 # length of fuselage
    m_pax = 88 # average mass of a passenger according to Google
    n_prop = 12 # number of engines
    n_pax = 5 # number of passengers (pilot included)
    pos_fus = 3.6 # fuselage centre of mass away from the nose
    pos_lgear = 3.6 # landing gear position away from the nose
    pos_frontwing, pos_backwing = 0.2, 7 # positions of the wings away from the nose
    m_prop = [576.2165134804536/12]*12 # list of mass of engines (so 30 kg per engine with nacelle and propeller)
    pos_prop = [-0.02401748, -0.02401748, -0.02401748, -0.02401748, -0.02401748, -0.02401748, 5.57598252,  5.57598252, 5.57598252,  5.57598252,  5.57598252,  5.57598252] # 8 on front wing and 8 on back wing
    wing = Wing(mtom, S1, S2, n_ult, A1, A2, [pos_frontwing, pos_backwing])
    fuselage = Fuselage(mtom, Pmax, lf, n_pax, pos_fus)
    lgear = LandingGear(mtom, pos_lgear)
    props = Propulsion(n_prop, m_prop, pos_prop)
    weight = Weight(m_pax, wing, fuselage, lgear, props, cargo_m = 35, cargo_pos = 6, battery_m = 880, battery_pos = 3.0, p_pax = [1.5, 3, 3, 4.2, 4.2])
    print(weight.print_weight_fractions())
    Ixx, Iyy, Izz, Ixz = weight.MMI(wmac = [0.7, 0.8], toc = [0.17, 0.17], vpos_wing = [0.3, 1.6])
    print('Ixx, Iyy, Izz, Ixz:', + Ixx, Iyy, Izz, Ixz)