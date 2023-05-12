import numpy as np



class VtolWeightEstimation:
    def __init__(self) -> None:
        self.components = []

    def add_component(self, CompObject):
        self.components.append(CompObject)  

    def compute_mass(self):
        mass_lst = [i.mass for i in self.components]
        return np.sum(mass_lst)


class Wing:
    # Roskam method (not accurate because does not take into account density of material but good enough for comparison
    def __init__(self, mtom, S1, S2, n_ult, A_1, A_2, pos=[], wmac = 0.8, toc = 0.17):
        self.id = "wing"
        self.S1_ft, self.S2_ft, self.S1, self.S2 = S1 * 3.28084 ** 2, S2 * 3.28084 ** 2, S1, S2
        self.n_ult = n_ult
        self.A_1, self.A_2 = A_1, A_2
        self.mtow_lbs = 2.20462 * mtom
        self.pos1, self.pos2 = pos
        self.wweight1 = 0.04674*((self.mtow_lbs/2)**0.397)*(self.S1_ft**0.36)*(self.n_ult**0.397)*(self.A_1**1.712)*0.453592
        self.mass = np.array([self.wweight1])
        self.moment = np.array(self.mass)*np.array(pos)
        self.wmac, self.toc = wmac, toc

class Fuselage:
    # Roskam method (not accurate because does not take into account density of material but good enough for comparison
    def __init__(self, mtom, Pmax, lf, npax, pos, wf=1.55):
        self.id = "fueselage"
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
        self.id = "landing gear"
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


        
