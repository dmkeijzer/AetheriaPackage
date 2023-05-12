import numpy as np



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