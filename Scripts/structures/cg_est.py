import numpy as np

class Wing:
    # Roskam method (not accurate because does not take into account density of material but good enough for comparison
    def __init__(self, mtom, S1, S2, n_ult, A, pos, config = None):
        self.config = config
        self.S1_ft, self.S2_ft = S1 * 3.28084 ** 2, S2 * 3.28084 ** 2
        self.mtow_lbs = 2.20462 * mtom
        self.pos = pos
        if self.config is not None and self.config in [1,2,3]:
            if self.config == 1:
                self.wweight1 = 0.04674*(self.mtow_lbs**0.397)*(self.S1_ft**0.36)*(n_ult**0.397)*(A**1.712)
                self.wweight2 = 0.04674*(self.mtow_lbs**0.397)*(self.S2_ft**0.36)*(n_ult**0.397)*(A**1.712)
            if self.config == 2:
                self.wweight1 = 0.04674 * (self.mtow_lbs ** 0.397) * (self.S1_ft ** 0.36) * (n_ult ** 0.397) * (A ** 1.712)
                self.wweight2 = 0.04674 * (self.mtow_lbs ** 0.397) * (self.S2_ft ** 0.36) * (n_ult ** 0.397) * (A ** 1.712)
            if self.config == 3:
                self.wweight = 0.04674 * (self.mtow_lbs ** 0.397) * (self.S1_ft ** 0.36) * (n_ult ** 0.397) * (A ** 1.712)
        else:
            print("Please select configuration from 1 to 3")

    def get_weight(self):
        if self.config is None:
            return
        if self.config == 1 or self.config == 2:
            return self.wweight1*0.453592, self.wweight2*0.453592
        if self.config == 3:
            return self.wweight*0.453592, 0

    def get_moment(self):
        pos1, pos2 = self.pos
        if self.config is None:
            return
        if self.config == 1 or self.config == 2:
            return self.wweight1*0.453592*pos1, self.wweight2*0.453592*pos2
        if self.config == 3:
            return self.wweight * pos1

class Fuselage:
    # Roskam method (not accurate because does not take into account density of material but good enough for comparison
    def __init__(self, mtom, Pmax, lf, npax, pos, config = None):
        self.config = config
        self.mtow_lbs = 2.20462 * mtom
        self.lf_ft = lf*3.28084
        self.Pmax_ft = Pmax*3.28084
        self.pos = pos
        if self.config is not None and self.config in [1, 2, 3]:
            if self.config == 1 or config == 2:
                self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.Pmax_ft)**0.778)*(self.lf_ft**0.383)*(npax**0.455)
                self.fweight_low = 0.04682*(self.mtow_lbs**0.692)*(self.Pmax_ft**0.379)*(self.lf_ft**0.590)
                self.fweight = (self.fweight_high + self.fweight_low)/2
            if self.config == 3:
                self.fweight = 0.04682 * (self.mtow_lbs ** 0.692) * (self.Pmax_ft ** 0.379) * (self.lf_ft ** 0.590)
                # for high wing uncomment the next line
                # self.fweight_high = 14.86 * (self.mtow_lbs ** 0.144) * ((self.lf_ft / self.Pmax_ft) ** 0.778)\
                #                     * (self.lf_ft ** 0.383) * (npax ** 0.455)
        else:
            print("Please select configuration from 1 to 3")

    def get_weight(self):
        if self.config is None:
            return
        if self.config in [1,2,3]:
            return self.fweight*0.453592

    def get_moment(self):
        if self.config is None:
            return
        if self.config in [1,2,3]:
            return self.fweight*0.453592 * self.pos


class LandingGear:

    def __init__(self, mtom, pos):
        self.m = 0.04*mtom
        self.pos = np.array(pos)

    def get_weight(self):
        return np.sum(self.m)

    def get_moment(self):
        return np.sum(self.m * self.pos)

class Propulsion:

    def __init__(self, n_prop, m_prop = [], pos_prop = []):
        self.nprop = n_prop
        self.wprop = np.array(m_prop)
        self.pos_prop = np.array(pos_prop)
        self.moment_prop = self.wprop*self.pos_prop

    def get_weight(self):
        return np.sum(self.wprop)

    def get_moment(self):
        return np.sum(self.wprop*self.pos_prop)

class Weight:

    def __init__(self, m_pax, wing, fuselage, landing_gear, propulsion, cargo_m, cargo_p, battery_m, battery_p, p_pax = []):
        self.w_pax = m_pax
        self.wing = wing
        # weights of components
        self.tot_pax_w = self.w_pax * 5
        self.wweight = np.sum(wing.get_weight()) if wing.config != 3 else wing.get_weight()[0]
        self.fweight = fuselage.get_weight()
        self.lweight = landing_gear.get_weight()
        self.pweight = propulsion.get_weight()
        self.cweight = cargo_m
        self.bweight = battery_m
        #moments of components
        self.moment_pax = np.sum(self.w_pax * np.array(p_pax))
        self.moment_w = np.sum(np.array(wing.get_moment())) if type(wing.get_moment()) is tuple else wing.get_moment()
        self.moment_f = fuselage.get_moment()
        self.moment_l = landing_gear.get_moment()
        self.moment_p = propulsion.get_moment()
        self.moment_c = self.cweight * cargo_p
        self.moment_b = self.bweight * battery_p
        #operational empty mass centre of gravity
        self.oem_cg = (self.moment_w + self.moment_f + self.moment_l + self.moment_p + self.moment_b) \
        /(self.wweight + self.pweight + self.lweight + self.fweight + self.bweight)
        self.mtom_cg = (self.moment_w + self.moment_f + self.moment_l + self.moment_p + self.moment_pax + self.moment_c + self.moment_b) \
        /(self.wweight + self.pweight + self.lweight + self.fweight + self.cweight + self.bweight + self.tot_pax_w)

        #masses
        self.oem = (self.wweight + self.pweight + self.lweight + self.fweight + self.bweight)
        self.mtom = (self.wweight + self.pweight + self.lweight + self.fweight + self.cweight + self.bweight + self.tot_pax_w)

    def print_weight_fractions(self):
        d = {}
        if type(self.wing.get_weight()) is tuple:
            d["Front wing"] = [self.wing.get_weight()[0], self.wing.get_weight()[0]/self.oem, self.wing.get_weight()[0]/self.mtom]
            d["Back wing"] = [self.wing.get_weight()[1], self.wing.get_weight()[1]/self.oem, self.wing.get_weight()[1]/self.mtom]
        else:
            d["Wing"] = [self.wing.get_weight(), self.wing.get_weight()/self.oem, self.wing.get_weight()/self.mtom]
        d['Fuselage'] = [self.fweight, self.fweight/self.oem, self.fweight/self.mtom]
        d['Landing gear'] = [self.lweight, self.lweight/self.oem, self.lweight/self.mtom]
        d['Propulsion'] = [self.pweight, self.pweight/self.oem, self.pweight/self.mtom]
        d['Cargo'] = [self.cweight, 0.0, self.cweight/self.mtom]
        d['Battery'] = [self.bweight, self.bweight/self.oem, self.bweight/self.mtom]
        d['Payload'] = [self.tot_pax_w, 0.0, self.tot_pax_w/self.oem]

        # print("{:<15} {:<20} {:<25} {:<15}".format('Component', 'Mass[kg]', 'fraction of OEM', 'fraction of MTOM'))
        print('--------------------------------------------------------------------------------')
        for k, v in d.items():
            mass, oem_frac, mtom_frac = v
            # print("{:<15} {:<20} {:<25} {:<15}".format(k, mass, oem_frac, mtom_frac))
        print('')
        print(f'Where OEM is {self.oem}kg with CG of {self.oem_cg}m, and MTOM is {self.mtom}kg with CG of {self.mtom_cg}m')
        for key in d:
            d[key] = {k: list(i) if isinstance(i, np.ndarray) else i for k, i in zip(["mass", "fracOEM", "fracEM"], d[key])}
        return d

if __name__ == '__main__':
    mtom = 1930
    S1, S2 = 11, 11
    A = 11.4
    n_ult = 3.38*1.5
    Pmax = 8.77182
    lf = 4
    m_pax = 88+9
    n_prop = 16
    m_prop = [40]*16
    pos_prop = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 3.8, 3.8, 3.8, 3.8, 3.8, 3.8, 3.8, 3.8]
    wing = Wing(mtom, S1, S2, n_ult, A, (0.2,3.8), config = 3)
    # mtom, Pmax, lf, npax, pos, config = None
    fuselage = Fuselage(mtom, Pmax, lf, 5, 2, config = 3)
    lgear = LandingGear(mtom, 1.5)
    props = Propulsion(n_prop, m_prop, pos_prop)
    weight = Weight(m_pax, wing, fuselage, lgear, props, 85, 3, 400, 2, p_pax = [0.8, 1.3, 1.3, 2.5, 2.5])
    # print(weight.oem, weight.mtom, weight.mtom_cg, weight.oem_cg)
    # print(wing.get_moment())
    # # (mtom, S1, S2, n_ult, A, pos, config = None)
    # print(wing.get_weight())



