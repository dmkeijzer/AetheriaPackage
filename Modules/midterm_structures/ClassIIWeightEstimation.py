import numpy as np



class VtolWeightEstimation:
    def __init__(self) -> None:
        self.components = []

    def add_component(self, CompObject):
        """ Method for adding a component to the VTOL

        :param CompObject: The component to be added to the VTOL
        :type CompObject: Component parent class
        """        
        self.components.append(CompObject)  

    def compute_mass(self):
        """ Computes the mass of entire vtol

        :return: Entire mass of VTOL
        :rtype: float
        """        
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
    def __init__(self,identifier, mtom, Pmax, lf, npax):
        """ Returns fuselage weight

        :param mtom: Maximum take off weight
        :type mtom: float
        :param Pmax:  Maximium perimeter of the fuselage
        :type Pmax: float
        :param lf: Fuselage length
        :type lf: float
        :param npax: Amount of passengers
        :type npax: int
        """        
        super().__init__()
        self.id = "fuselage"
        self.mtow_lbs = 2.20462 * mtom
        self.lf_ft, self.lf = lf*3.28084, lf
        self.Pmax_ft = Pmax*3.28084
        self.npax = npax
        if identifier == "J1":
            self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.Pmax_ft)**0.778)*(self.lf_ft**0.383)*(self.npax**0.455)
            self.mass = self.fweight_high*0.453592
        else:
            self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.Pmax_ft)**0.778)*(self.lf_ft**0.383)*(self.npax**0.455)
            self.fweight_low = 0.04682*(self.mtow_lbs**0.692)*(self.Pmax_ft**0.379)*(self.lf_ft**0.590)
            self.fweight = (self.fweight_high + self.fweight_low)/2
            self.mass = self.fweight*0.453592

class LandingGear(Component):
    def __init__(self, mtom):
        super().__init__()
        self.id = "landing gear"
        self.mass = 0.04*mtom

class Engines(Component):
    def __init__(self,p_max, p_dense ):
        """Returns the mas of the engines based on power

        :param p_max: Maximum power [w]
        :type p_max: float
        :param p_dense: Power density [w]
        :type p_dense: float
        """        
        super().__init__()
        self.mass = p_max/p_dense


#TODO add hydrogen system
#TODO add battery/turbine engine system
#TODO add tail 
#TODO Think of penalty for weight of fuselage for crashworthiness, firewall et cetera  

        
