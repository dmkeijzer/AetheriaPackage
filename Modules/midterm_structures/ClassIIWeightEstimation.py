import numpy as np



class VtolWeightEstimation:
    def __init__(self) -> None:
        self.components = []

    def add_component(self, CompObject):
        self.components.append(CompObject)  

    def compute_mass(self):
        mass_lst = [i.return_mass() for i in self.components]
        return np.sum(mass_lst)

class Component():
    """ This is the parent class for all weight components, it initalized the mass
    attribute and a way of easily returning it. This is used in VtolWeightEstimation.
    """    
    def __init__(self) -> None:
        self.mass = None

    def return_mass(self): return self.mass


class Wing(Component):
    # Roskam method (not accurate because does not take into account density of material but good enough for comparison
    def __init__(self, mtom, S, n_ult, A):
        super().__init__()
        self.id = "wing"
        self.S_ft = S*3.28084
        self.n_ult = n_ult
        self.A = A
        self.mtow_lbs = 2.20462 * mtom
        self.mass = 0.04674*((self.mtow_lbs/2)**0.397)*(self.S_ft**0.36)*(self.n_ult**0.397)*(self.A**1.712)*0.453592

class Fuselage(Component):
    # Roskam method (not accurate because does not take into account density of material but good enough for comparison
    def __init__(self, mtom, Pmax, lf, npax):
        """ Component Class

        :param mtom: _description_
        :type mtom: _type_
        :param Pmax: _description_
        :type Pmax: _type_
        :param lf: _description_
        :type lf: _type_
        :param npax: _description_
        :type npax: _type_
        """        
        super().__init__()
        self.id = "fuselage"
        self.mtow_lbs = 2.20462 * mtom
        self.lf_ft, self.lf = lf*3.28084, lf
        self.Pmax_ft = Pmax*3.28084
        self.npax = npax
        self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.Pmax_ft)**0.778)*(self.lf_ft**0.383)*(self.npax**0.455)
        self.fweight_low = 0.04682*(self.mtow_lbs**0.692)*(self.Pmax_ft**0.379)*(self.lf_ft**0.590)
        self.fweight = (self.fweight_high + self.fweight_low)/2
        self.mass = self.fweight*0.453592

class LandingGear(Component):
    def __init__(self, mtom):
        super().__init__()
        self.id = "landing gear"
        self.mass = 0.04*mtom
        self.moment = self.mass * self.pos

class Propulsion(Component):
    def __init__(self, n_prop, m_prop = [] ):
        super().__init__()
        self.nprop = n_prop
        self.wprop = np.array(m_prop)
        self.moment_prop = self.wprop*self.pos_prop
        self.mass = np.sum(self.wprop)


        
