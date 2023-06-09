import numpy as np
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
sys.path.append(os.path.join(list(pl.Path(__file__).parents)[2], "modules","powersizing"))

from modules.powersizing.battery import BatterySizing
from modules.powersizing.fuellCell import FuellCellSizing
from modules.powersizing.hydrogenTank import HydrogenTankSizing
from modules.powersizing.energypowerrequirement import MissionRequirements
# from modules.powersizing.powersystem import PropulsionSystem, onlyFuelCellSizing
import input.data_structures.GeneralConstants as  const
from input.data_structures import *




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
        return np.sum(mass_lst)*const.oem_cont

class Component():
    """ This is the parent class for all weight components, it initalized the mass
    attribute and a way of easily returning it. This is used in VtolWeightEstimation.
    """    
    def __init__(self) -> None:
        self.mass = None

    def return_mass(self): return self.mass


class WingWeight(Component):
    def __init__(self, mtom, S, n_ult, A):
        """Retunrs the weight of the wing, Cessna method cantilever wings pg. 67 pt 5. Component weight estimation Roskam

        :param mtom: maximum take off mass
        :type mtom: float
        :param S: Wing area
        :type S: float
        :param n_ult: Ultimate load factor
        :type n_ult: float
        :param A: Aspect ratio
        :type A: float
        """        
        super().__init__()
        self.id = "wing"
        self.S_ft = S*10.7639104
        self.n_ult = n_ult
        self.A = A
        self.mtow_lbs = 2.20462 * mtom
        self.mass = 0.04674*(self.mtow_lbs**0.397)*(self.S_ft**0.36)*(self.n_ult**0.397)*(self.A**1.712)*0.453592

class FuselageWeight(Component):
    def __init__(self,identifier, mtom, lf, nult, wf, hf, Vc):
        """ Returns fuselage weight, cessna method page 75 Pt 5. component weight estimaation Roskam.

        :param mtom: Maximum take off weight
        :type mtom: float
        :param max_per:  Maximium perimeter of the fuselage
        :type max_per: float
        :param lf: Fuselage length
        :type lf: float
        :param npax: Amount of passengers including pilot
        :type npax: int
        """        
        super().__init__()
        self.id = "fuselage"
        self.mtow_lbs = 2.20462 * mtom
        self.lf_ft, self.lf = lf*3.28084, lf

        self.nult = nult # ultimate load factor
        self.wf_ft = wf*3.28084 # width fuselage [ft]
        self.hf_ft = hf*3.28084 # height fuselage [ft]
        self.Vc_kts = Vc*1.94384449 # design cruise speed [kts]

        self.fweigh_USAF = 200*((self.mtow_lbs*self.nult/10**5)**0.286*(self.lf_ft/10)**0.857*((self.wf_ft+self.hf_ft)/10)*(self.Vc_kts/100)**0.338)**1.1
        self.mass = self.fweigh_USAF*0.453592

        #if identifier == "J1":
        #    self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.max_per_ft)**0.778)*(self.lf_ft**0.383)*(self.npax**0.455)
        #    self.mass = self.fweight_high*0.453592
        #else:
        #    self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.max_per_ft)**0.778)*(self.lf_ft**0.383)*(self.npax**0.455)
        #    self.fweight_low = 0.04682*(self.mtow_lbs**0.692)*(self.max_per_ft**0.374)*(self.lf_ft**0.590)
        #    self.fweight = (self.fweight_high + self.fweight_low)/2
        #    self.mass = self.fweight*0.453592

class LandingGear(Component):
    def __init__(self, mtom):
        """Computes the mass of the landing gear, simplified Cessna method for retractable landing gears pg. 81 Pt V component weight estimation

        :param mtom: maximum take off weight
        :type mtom: float
        """        
        super().__init__()
        self.id = "landing gear"
        self.mtow_lbs = 2.20462 * mtom
        self.mass = (0.04*self.mtow_lbs + 6.2)*0.453592


class Powertrain(Component):
    def __init__(self,p_max, p_dense ):
        """Returns the mas of the engines based on power

        :param p_max: Maximum power [w]
        :type p_max: float
        :param p_dense: Power density [w/kg]
        :type p_dense: float
        """        
        super().__init__()
        self.id = "Powertrain"
        self.mass = 12 * (13 + 10) #p_max/p_dense 12 engines (13 kg) and inverters (10 kg) These are scimo engines and converters https://sci-mo.de/motors/


class Propeller(Component):
    def __init__(self ):
        """Returns the mas of the engines based on power

        :param p_max: Maximum power [w]
        :
        """
        
        super().__init__()
        self.id = "Propeller"
        self.mass = 6 * 20 # 6 propellers and 30 kg per proppeller (I just googled a bit)

class HorizontalTailWeight(Component):
    def __init__(self, w_to, S_h, A_h, t_r_h ):
        """Computes the mass of the horizontal tail, only used for Joby. Cessna method pg. 71 pt V component weight estimation

        :param W_to: take off weight in  kg
        :type W_to: float
        :param S_h: Horizontal tail area in  m^2
        :type S_h: float
        :param A_h: Aspect ratio horizontal tail
        :type A_h: float
        :param t_r_h: Horizontal tail maximum root thickness in m 
        :type t_r_h: float
        """        

        self.id = "Horizontal tail"
        w_to_lbs = 2.20462262*w_to
        S_h_ft = 10.7639104*S_h
        t_r_h_ft = 3.2808399*t_r_h

        super().__init__()
        self.mass =  (3.184*w_to_lbs**0.887*S_h_ft**0.101*A_h**0.138)/(174.04*t_r_h_ft**0.223)*0.45359237

class NacelleWeight(Component):
    def __init__(self, p_to):
        """ Returns nacelle weight

        :param w_to: Total take off weight aka MTOM
        :type w_to: float
        """        
        super().__init__()
        self.id = "Nacelles"
        self.p_to_hp = 0.001341*p_to
        self.mass = 0.24*self.p_to_hp*0.45359237 # Original was 0.24 but decreased it since the electric aircraft would require less structural weight0

class H2System(Component):
    def __init__(self, energy, cruisePower, hoverPower):
        """Returns the lightest solutions of the hydrogen 

        :param energy: Amount of energy consumed
        :type energy: float
        :param cruisePower: Power during cruise
        :type cruisePower: float
        :param hoverPower: Power during hover
        :type hoverPower: float
        """        
        raise Exception("This function is deprecated and no longer suppored, only here for the sake of documentation")
        super().__init__()
        self.id = "Hydrogen system"
        echo = np.arange(0,1.5,0.05)

        #batteries
        Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, charging_efficiency= const.ChargingEfficiency, depth_of_discharge= const.DOD, discharge_effiency=0.95)
        Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= const.ChargingEfficiency, depth_of_discharge= const.DOD, discharge_effiency=0.95)
        Solidstatebat = BatterySizing(sp_en_den= 0.5, vol_en_den=1, sp_pow_den=10,cost =82.2, charging_efficiency= const.ChargingEfficiency, depth_of_discharge= const.DOD, discharge_effiency=0.95)
        #HydrogenBat = BatterySizing(sp_en_den=1.85,vol_en_den=3.25,sp_pow_den=2.9,cost=0,discharge_effiency=0.6,charging_efficiency=1,depth_of_discharge=1)


        #-----------------------Model-----------------
        BatteryUsed = Liionbat
        FirstFC = FuellCellSizing(const.PowerDensityFuellCell,const.VolumeDensityFuellCell,const.effiencyFuellCell, 0)
        FuelTank = HydrogenTankSizing(const.EnergyDensityTank,const.VolumeDensityTank,0)
        InitialMission = MissionRequirements(EnergyRequired= energy, CruisePower= cruisePower, HoverPower= hoverPower )


        #calculating mass
        Mass = PropulsionSystem.mass(np.copy(echo),
                                                                    Mission= InitialMission, 
                                                                    Battery = BatteryUsed, 
                                                                    FuellCell = FirstFC, 
                                                                    FuellTank= FuelTank)

        TotalMass, TankMass, FuelCellMass, BatteryMas, coolingmass= Mass

        # OnlyH2Tank, OnlyH2FC, OnlyH2TankVol, OnlyH2FCVol = onlyFuelCellSizing(InitialMission, FuelTank, FirstFC)

        self.mass = np.min(TotalMass)


class Miscallenous(Component):
    def __init__(self, mtom, oew, npax) -> None:
        """ Returns the miscallenous weight which consists out of flight control, electrical system
        , avionics, aircondition and furnishing. All in line comments refer to pages in
        Pt. 5 Component weight estimation by Roskam

        :param mtom: Maximum take-off weight
        :type mtom: float
        """        
        super().__init__()
        self.id = "misc"
        w_to_lbs = 2.20462262*mtom
        w_oew_lbs = 2.20462262*oew

        mass_fc = 0.0168*w_to_lbs # flight control system weight Cessna method pg. 98
        mass_elec = 0.0268*w_to_lbs # Electrical system mass  cessna method pg. 101
        mass_avionics = 40 + 0.008*w_to_lbs # Avionics system mass Torenbeek pg. 103
        mass_airco = 0.018*w_oew_lbs   # Airconditioning mass Torenbeek method pg. 104
        mass_fur = 0.412*npax**1.145*w_to_lbs**0.489 # Furnishing mass Cessna method pg.107

        self.mass = np.sum([mass_fur, mass_airco, mass_avionics, mass_elec, mass_fc])*0.45359237


        
def get_weight_vtol(perf_par: PerformanceParameters, fuselage: Fuselage, wing: Wing,  engine: Engine, vtail: VeeTail, test= False):
    """ This function is used for the final design, it reuses some of the codes created during
    the midterm. It computes the final weight of the vtol using the data structures created in the
    final design phase

    It uses the following weight components
    --------------------------------------
    Powersystem mass -> Sized in power sizing, retrieved from perf class
    Engine mass -> Scimo engines and inverters used
    wing mass -> class II/wingbox code
    vtail mass -> Class II/wingbox code
    fuselage mass -> Class II
    landing gear mass -> Class II
    nacelle mass -> class II
    misc mass -> class II
    --------------------------------------
    """    


    # Wing mass 
    wing.wing_weight = WingWeight(perf_par.MTOM, wing.surface, perf_par.n_ult, wing.aspectratio).mass #This is automatically updated in the wing box calculations if they work

    # Vtail mass
    # Wing equation is used instead of horizontal tail because of the heay load of the engine which is attached
    vtail.vtail_weight = WingWeight(perf_par.MTOM, vtail.surface, perf_par.n_ult, vtail.aspectratio).mass

    #fuselage mass
    fuselage.fuselage_weight = FuselageWeight("J1", perf_par.MTOM, fuselage.length_fuselage, perf_par.n_ult, fuselage.width_fuselage_outer, fuselage.height_fuselage_outer, const.v_cr).mass

    #landing gear mass
    lg_weight = LandingGear(perf_par.MTOM).mass

    # Nacelle and engine mass

    total_engine_mass = Powertrain(perf_par.hoverPower, const.p_density).mass + Propeller().mass + 90 #90 kg is for the pylon length
    nacelle_mass = NacelleWeight(perf_par.hoverPower).mass

    engine.totalmass = nacelle_mass + total_engine_mass
    engine.mass_perpowertrain = (engine.totalmass)/engine.no_engines
    engine.mass_pernacelle = nacelle_mass/engine.no_engines
    engine.mass_pertotalengine = total_engine_mass/engine.no_engines

    # Misc mass
    misc_mass = Miscallenous(perf_par.MTOM, perf_par.OEM, const.npax + 1).mass

    perf_par.OEM = np.sum([ perf_par.powersystem_mass ,   wing.wing_weight, vtail.vtail_weight, fuselage.fuselage_weight, nacelle_mass, total_engine_mass,  lg_weight, misc_mass])
    perf_par.MTOM =  perf_par.OEM + const.m_pl

    # Update weight not part of a data structure

    with open(const.json_path, "r") as f:
        data = json.load(f)
    

    data["misc_weight"] = misc_mass
    data["lg_weight"] = lg_weight
    data["nacelle_weight"] = nacelle_mass
    data["powertrain_weight"] =  total_engine_mass

    if not test:
        with open(const.json_path, "w") as f:
            json.dump(data, f, indent= 6)
    else:
        return perf_par, wing, vtail, fuselage, engine, data

    return perf_par, wing, vtail, fuselage, engine

