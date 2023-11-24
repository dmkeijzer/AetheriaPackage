
import os
import sys
import pathlib as pl
from math import ceil
import numpy as np
from AetheriaPackage.data_structs import *

class CoolingsystemPerformance:
    """This is wrapper class for all the functions depended on the coolingperformance """

    def mass_flow(heat, delta_temperature: float, heat_capacity: float ) -> float:
        """calculates mass flow on the basis of the maximum allowable dT
            :param: heat: [W]
            :param: delta_temperature: maximum allowable coolant temperature rise [K]
            :param: heat_capacity: heat capacity of liquid [J/(kg k)]
        """
        return heat / (delta_temperature * heat_capacity)

    def exchange_effectiveness(Cr: float, NTU: float) -> float: 
        """
        :param: Cr[-]: c_min / c_max 
        :param: NTU [-]: Number of transfer units
        :return: epsilon[-]: exchange effectiveness
        """
        return 1 - np.exp( (1/Cr )*  NTU ** (0.22) * ( np.exp(-1 * Cr* NTU ** (0.78)) -1 ) )

    def power_fan_massflow(massflow: float, density: float, fan_area: float) -> float:
        """
        Calculate the fan power required to obtain specific massflow

        :param: massflow[kg/s] 
        :param: density[kg/(m^3)]
        :param: fan_area[m^2]
        :return: fanpower[W]
        """
        return 0.25 * (massflow * massflow * massflow) / (density * density * fan_area * fan_area) # 1/4 * massflow^3 / (rho^2 * area^2)

    def power_fan_airspeed(airspeed: float, fan_area: float, density:float):
        """
        Calculate the fan power required to obtain specific airspeed
        
        :param: airspeed[m/s] 
        :param: density[kg/(m^3)]
        :param: fan_area[m^2]
        :return: fanpower[W]
        """

        return 0.5* density * fan_area * airspeed * airspeed * airspeed

    def max_heat_transfer(c_min: float, T_hot_in:float, T_cold_in: float) -> float:
        """
        Calculate the maximum heat transfer rate.

        :param c_min: Minimum heat capacity rate [W/k]
        :param T_hot_in: Inlet temperature of the hot fluid [K]
        :param T_cold_in: Inlet temperature of the cold fluid [K]
        :return: Maximum heat transfer rate [W]
        """
        return c_min * (T_hot_in - T_cold_in)

    def number_transfer_units(overall_heat_transfer_capacity:float , cmin: float ):
        """
        Calculate the NTU

        :param overall_heat_transfer_capacity: [W/K]
        :param cmin:  Minimum heat capacity rate [W/K]
        :return: NTU [-]
        """
        return overall_heat_transfer_capacity  / cmin


    def calculate_heat_expelled(c_hot: float, c_cold: float, T_hot_in: float, T_cold_in: float,overall_heat_transfer_capacity : float):
        """
        Calculate the heat expelled by a heat exchanger.

        :param c_hot: heat capacity rate of the hot fluid [W/k]
        :param c_cold: heat capacity rate of the cold fluid [W/k]
        :param T_hot_in: Inlet temperature of the hot fluid [K]
        :param T_cold_in: Inlet temperature of the cold fluid [K]
        :param overall_heat_transfer_capacity: [W / k]
        :return: Heat expelled by the heat exchanger [W]
        """
        
        #determine which side is min and which side is the max
        c_min = min(c_hot,c_cold)
        c_max = max(c_cold, c_hot)
        
        #determine the max heat transfer out
        Qmax = CoolingsystemPerformance.max_heat_transfer(c_min= c_min , T_hot_in= T_hot_in, T_cold_in= T_cold_in)

        #determine heat exchanger effectivenes
        cr = c_min/c_max
        NTU = CoolingsystemPerformance.number_transfer_units(overall_heat_transfer_capacity,c_min)
        epsilon = CoolingsystemPerformance.exchange_effectiveness(cr, NTU)

        #determine heat expelled by the radiator
        Q_expelled = epsilon * Qmax
        return Q_expelled, epsilon



class RadiatorPerformance:
    def hx_geometry(radiator : Radiator) -> Radiator:
        """calculate surface areas of the radiator 
            Function based on the geometry specified in A. Scoccimarro thesis on preliminary thermal management sizing
        """
        radiator.W_channel = radiator.h_tube - 2* radiator.t_tube
        H_channel = radiator.h_tube - 2* radiator.t_tube

        radiator.HX_gamma = radiator.t_fin /radiator.s_fin
        radiator.HX_delta = radiator.t_fin / radiator.l_fin
        radiator.HX_alpha = radiator.s_fin / radiator.h_fin

        radiator.N_channel = np.floor(radiator.Z_HX / (radiator.W_channel + radiator.t_channel)) * np.floor(radiator.H_HX / (radiator.h_fin + radiator.h_tube)) 
       
        
        radiator.A_hot = 2 * (radiator.W_channel +  H_channel) * radiator.W_HX * radiator.N_channel

        radiator.N_fin = np.floor(radiator.W_HX / (radiator.s_fin + radiator.t_fin)) * np.floor(radiator.H_HX / (radiator.h_fin + radiator.h_tube)) * np.floor(radiator.Z_HX / radiator.l_fin)
        radiator.A_fin = 2 * radiator.h_fin * radiator.l_fin + 2 * radiator.h_fin * radiator.t_fin + radiator.s_fin * radiator.t_fin

        A_primary = 2 * radiator.s_fin * radiator.l_fin

        radiator.A_cold = radiator.N_fin * (radiator.A_fin + A_primary)
        radiator.A_fs_cross = radiator.s_fin * radiator.h_fin * radiator.N_fin / np.floor(radiator.Z_HX / radiator.l_fin)
        radiator.A_cross_hot = radiator.W_channel * H_channel * radiator.N_channel

        return radiator


    def fin_efficiency(h_c_cold: float, thermal_conductivity_fin: float, radiator) -> float:
        """
        calculate fin efficiency to be used in the surface efficiency

        :param: h_c_cold[W/(m^2 K)]: Heat transfer coefficient cold side (air)
        :param: thermal_conductivity_fin[W/(m K)]: 
        :param: radiator
        :return: eta_fin[-]: fin efficiency
        """
        ml = np.sqrt(2* h_c_cold / (thermal_conductivity_fin * radiator.t_fin)) * (radiator.h_fin + radiator.t_fin)/2
        eta_fin = np.tanh(ml)/ ml
        return eta_fin

    def surface_efficiency(radiator: Radiator, eta_fin: float ) -> float:
        """ function that calculates surface efficiency for thermal resistance"""
        return 1- (radiator.A_fin/ radiator.A_cold * (1 - eta_fin))  

    def hx_thermal_resistance(radiator: Radiator, h_c_cold: float, h_c_hot: float, eta_surface: float) -> Radiator:
        return 1/(h_c_hot * radiator.A_hot) + 1 / (radiator.A_cold * h_c_cold * eta_surface)

    def colburn_factor(reynolds: float, alpha: float, delta: float, gamma: float) -> float:
        """calculates colburn factor"""
        j = 0.6522 * reynolds**(-0.5403) * alpha**(-0.1541) * delta**0.1499 * gamma**(-0.0678)  *  (1 + 5.269e-5 * reynolds**1.34 * alpha**0.504 * delta**0.456 * gamma**(-1.055))**0.1
        return j

    def hydralic_diameter_HX( width: float, height: float ) -> float:
        """ calcualte hydraulic diamter of rectangular tube
        
        :param: width[m]: channel width
        :param: height[m]: channel height
        :return: hydraulic diameter[m]:
        """
        return 2 * (width * height ) / (width + height)
    
    def reynolds_HX(mass_flux: float, hydraulic_diameter: float, viscosity) -> float:
        return mass_flux * hydraulic_diameter / viscosity

    def prandtl_heat(heat_capacity: float, viscosity: float, thermal_conductivity: float) -> float:
        return heat_capacity * viscosity / thermal_conductivity

    def heat_capacity_cold(colburn : float, mass_flux: float, c_p: float, prandtl: float) -> float:
        return colburn * mass_flux * c_p / (prandtl ** (2/3))

    def calculate_flam(Re, AR_channel):
        flam = 24 * (1 - 1.3553 * AR_channel + 1.9467 * AR_channel**2 - 1.7012 * AR_channel**3 + 0.9564 * AR_channel**4 - 0.2537 * AR_channel**5) * Re**(-1)
        return flam

    def heat_capacity_hot(Re: float, Pr: float, f: float, Dh: float, k: float, pipe_length: float = 0) -> float:
        if Re > 3e3 and Re < 1e4:
            hc = ((Re - 1000) * Pr * (f / 2) * (k / Dh)) / (1 + 12.7 * (Pr**(2/3) - 1) * (f / 2)**0.5)
            return hc
        elif Re > 1e4 and Re < 5e6:
            hc = (Re * Pr * (f / 2) * (k / Dh)) / (1 + 12.7 * (Pr**(2/3) - 1) * (f / 2)**0.5)
            return hc
        elif Re < 3e3:
            Nu = 7.54 +  (0.03 * (Dh /pipe_length) * Re * Pr) / (1 + 0.016 *((Dh/ pipe_length) *Re * Pr ) ** (2/3))
            #Nu = 0.664 * Re **0.5 * Pr ** (1/3)
            hc = k * Nu / Dh
            return hc
        else: 
            raise ValueError("Re is not in range to calculate heat transfer coefficient hot")

    def pressure_drop(friction: float, mass_flux: float, length:float, hydraulic_diameter: float, density: float)-> float:
        """ 
            calculate pressure drop of the coolant

            :param: friction: friction factor [-]
            :param: mass_flux: [kg / (m^2 s)]
            :param: length: length of the channel [m]
            :param: hydraulic diamter: [m]
            :param: density: density of the coolant [kg/m^3]
            :return: pressure drop


        """
        return (2 * friction * mass_flux * mass_flux * length)/ (hydraulic_diameter * density) 


    def mass_flux(mass_flow: float, A_crossectional: float) -> float:
        """ calculate mass flux 

        :param: mass_flow[kg/s]:
        :param: A_crossectional[m^2]: cross sectional area of all tunnels combined
        :returns: mass_flux[kg/(m^2 s)]
        
        """
        return mass_flow / A_crossectional

    def mass_radiator(HX: Radiator, density_material: float) -> float:
        """Calculate radiator mass
        :param: HX: Heat exhanger
        :param: density_material: material density[kg/m^3]
        :return: radiator mass: radiator mass with 20% contigency[kg]    """
        fin_volume = (HX.s_fin + 2 * HX.h_fin) * HX.t_fin * HX.l_fin * HX.N_fin
        primary_cold_volume = (HX.A_cold - HX.A_fin * HX.N_fin) * HX.t_tube
        hot_volume = HX.A_hot * HX.t_channel
        total_volume = fin_volume + primary_cold_volume + hot_volume
        return total_volume * density_material * 1.2

    def cooling_radiator(HX:Radiator, mass_flow_cold: float, mass_flow_hot: float, air: Fluid, coolant: Fluid) -> float:
        """calculates thermal resistance of the radiator

        :param: HX: Radiator data
        :param: mass_flow_cold[kg/s]: air mass flow
        :param: mass_flow_hot[kg/s]: coolant mass flow
        :param: air
        :param: coolant: cooling liquid

        :return: thermal resistance: 1/UA
        """
        
        #cold side
        massflux_cold = RadiatorPerformance.mass_flux(mass_flow_cold,HX.A_fs_cross)
        ##print(f"mass flux cold: { mass_flow_cold}")
        Dh_cold =  RadiatorPerformance.hydralic_diameter_HX(HX.s_fin, HX.h_fin)
        ##print(f"Dh cold: {Dh_cold }")
        Re_cold =  RadiatorPerformance.reynolds_HX(mass_flux= massflux_cold, hydraulic_diameter=Dh_cold,viscosity= air.viscosity) 
        ##print(f"Re cold: {Re_cold}")
        colburn =  RadiatorPerformance.colburn_factor(alpha= HX.HX_alpha, delta= HX.HX_delta, gamma= HX.HX_gamma, reynolds= Re_cold)
        Pr_cold =  RadiatorPerformance.prandtl_heat(air.heat_capacity, viscosity= air.viscosity, thermal_conductivity= air.thermal_conductivity)
        h_c_cold =  RadiatorPerformance.heat_capacity_cold(colburn, massflux_cold, air.heat_capacity, Pr_cold) 
        #print(f"hc cold:  {h_c_cold}\n")

        #hot side
        #print(f"mdot hot: {mass_flow_hot }")
        mass_flux_hot =  RadiatorPerformance.mass_flux(mass_flow_hot,HX.A_cross_hot)
        #print(f"mass flux: {mass_flux_hot}")
        dh_hot =  RadiatorPerformance.hydralic_diameter_HX(HX.W_channel,HX.W_channel)
        #print(f"Dh hot : {dh_hot }")
        Re_hot =  RadiatorPerformance.reynolds_HX(mass_flux_hot, dh_hot,coolant.viscosity) 
        Pr_hot =  RadiatorPerformance.prandtl_heat(coolant.heat_capacity, coolant.viscosity, coolant.thermal_conductivity)
        AR_channel = HX.W_channel  /HX. W_HX
        #print(f"AR: {AR_channel}")
        friction_factor_hot =  RadiatorPerformance.calculate_flam(Re_hot,AR_channel)
        #print(f"Friction factor: {friction_factor_hot }")
        #print(f"Reynolds number hot: {Re_hot }")
        h_c_hot =  RadiatorPerformance.heat_capacity_hot(Re_hot, Pr_hot, friction_factor_hot , dh_hot, coolant.thermal_conductivity, HX.W_HX) 
        #print(f"hc hot: {h_c_hot:,}")

        eta_fin =  RadiatorPerformance.fin_efficiency(h_c_cold, air.thermal_conductivity , HX )
        eta_surface =  RadiatorPerformance.surface_efficiency(HX,eta_fin)
        ##print(eta_surface)
        R_tot =  RadiatorPerformance.hx_thermal_resistance(HX,h_c_cold, h_c_hot,eta_surface)
        delta_pressure = HX.N_channel * RadiatorPerformance.pressure_drop(friction=friction_factor_hot, mass_flux= mass_flux_hot, 
                                                           length= HX.W_HX, hydraulic_diameter=dh_hot,
                                                           density= coolant.density) 

        return R_tot, delta_pressure 

class BatterySizing:
    """This class is to estimate the parameters of a battery"""
    def __init__(self, sp_en_den, vol_en_den, sp_pow_den, cost,discharge_effiency,charging_efficiency,depth_of_discharge):
        """ 
        :param: sp_en_den: Energy Density of the battery [kWh/kg]
        :param: vol_en_den: Volumetric Density [kWh/l]
        :param: sp_power_den: Power Density of the battery [kW/kg]
        :param: CostDensity: Cost per Wh of the battery [US$/kWh]"""
        self.EnergyDensity = sp_en_den
        #self.Energy = tot_energy
        self.VolumeDensity = vol_en_den
        self.PowerDensity = sp_pow_den
        self.CostDensity = cost
        self.Efficiency = discharge_effiency
        self.DOD = depth_of_discharge
        self.ChargingEfficiency = charging_efficiency


    def energymass(self,Energy):
        """
        :param Energy: Required total energy for the battery [kWh]
        :param sp_en_den: Specific energy density of the battery [kWh/kg]
        :return: Mass of the battery [kg]
        """
        return Energy/ self.EnergyDensity


    def volume(self,Energy):
        """
        :param energy: Required total energy for the battery [kWh]
        :param vol_en_den: Volumetric energy density of the battery [kWh/l]
        :return: Volume of the battery [m^3]
        """
        return Energy /self.VolumeDensity * 0.001

    def price(self,Energy):
        """
        :param energy: Required total energy for the battery [kWh]
        :param cost: Cost per Wh of the battery [US$/kWh]
        :return: Approx cost of the battery [US$]
        """
        return Energy *self.Cost


class FuellCellSizing:
    """This class is to estimate the parameters of a Fuell Cell"""
    def __init__(self, sp_power_den:float ,vol_power_den:float, efficiency: float,  cost:float):
        """
            :param: sp_power_den: Energy Density of the battery [kWh/kg]
            :param: vol_en_den: Volumetric Density [kWh/l]
            :param: sp_power_den: Power Density of the battery [kW/kg]
            :param: CostDensity: Cost per Wh of the battery [US$/kWh]
            """
        self.PowerDensity= sp_power_den
        #self.power = power_required
        self.VolumeDensity = vol_power_den
        self.Cost = cost
        self.Efficiency = efficiency

    def mass(self,power):
        """
        :param power: Power requirement for the fuell cell[kW]
        :param sp_P_den: Power density of the fuell cell[kW/kg]
        :return: Mass of the battery
        """
        return power / self.PowerDensity

    def volume(self,power):
        """
        :param Powery: Power required from the fuell cell [kW]
        :param vol_en_den: Volumetric energy density of the fuell cell [kW/l]
        :return: Volume of the fuell cell [m^3]
        """
        return power /self.VolumeDensity  * 0.001

    def price(self,power):
        """
        :param power: Required power for the fuell cell [Wh]
        :param cost: Cost per Wh of the battery [US$/kW]
        :return: Approx cost of the battery [US$]
        """
        return power * self.Cost


class HydrogenTankSizing:
    """This class is to estimate the parameters of a Hydrogen tank"""
    def __init__(self, sp_en_den, vol_en_den, cost):
        """"
        :param: sp_en_den[kWh/kg]: Specific energy density 
        :param: vol_en_den[kWh/l]: specific volumetric density
        :param: cost[US$/kWh]: cost per kWh   """
        self.EnergyDensity = sp_en_den 
        self.VolumeDensity= vol_en_den #[kWh/l]
        self.cost = cost

    def mass(self,energy):
        """
        :param energy: Required total energy for the tank [kWh]
        :param sp_en_den: Specific energy density of the tank + hydrogen in it[kWh/kg]
        :return: Mass of the battery
        """
        return energy / self.EnergyDensity

    def volume(self,energy):
        """
        :param energy: Required total energy from the hydrogen tank [kWh]
        :param vol_en_den: Volumetric energy density of the hydrogen tank [kWh/l]
        :return: Volume of the battery [m^3]
        """
        return energy/self.VolumeDensity * 0.001

    def price(self,energy) :
        """
        :param energy: Required total energy for the battery [kWh]
        :param cost: Cost per kWh of the battery [US$/kWh]
        :return: Approx cost of the battery [US$]
        """
        return energy * self.cost




def energy_cruise_mass(EnergyRequired: float , echo: float , Tank: HydrogenTank, Battery: Battery, FuellCell: FuelCell) -> list[float]:
    """Calculate the mass of the hydrogen tank + the hydrogen itself
        input:
            -EnergyRequired [Wh] : The total Energy required for the mission
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            -EnergyDensityHydrogen [Wh/kg]: The energy density of the tank + hydrogen in it
            -EnergyDensityBattery [Wh/kg]: The enegery density of the battery
            
        output:
            -Tankmass [kg]: The total mass of fuel tanks + the hydrogen in it
            -Batterymass [kg]: the battery mass 
    """
    
    
    #calculating energy required for the fuell cell and the battery
    Tankmass = Tank.mass(EnergyRequired * echo) / FuellCell.efficiency
    Batterymass = Battery.energymass((1-echo)*EnergyRequired) 
    
    return  Tankmass , Batterymass


def power_cruise_mass(PowerRequired: float, echo: float,  FuellCell:FuelCell, Battery:Battery ) -> list[float] :
    """Fuell Cell sizing and battery sizing for cruise conditions
        input:
            -PowerRequired [kW] : The power required during cruise
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            -PowerDensityFC [kW/kg]: The power density of the fuell cell
            -PowerDensityBattery [kW/kg]: The power battery of the battery
        return:
            -FCmass [kg]: Fuell cell mass
            -Batterymass[kg]: Battery mass
    """
    
    FCmass = ceil(PowerRequired/FuelCell.maxpower)*FuellCell.mass
    Batterymass = PowerRequired * (1-echo) / Battery.PowerDensity / Battery.End_of_life
    
    for i in range(len(Batterymass)):
        Batterymass[i] = max(0,Batterymass[i])


    return FCmass, Batterymass

def hover_mass(PowerRequired: float ,MaxPowerFC: float, Battery: Battery) -> float :

    """Battery sizing for hover conditions
        input:
            -PowerRequired [kW] : The power required during hover
            -MaxPowerFC [kW]: The maximum power the Fuell Cell can deliver
            -PowerDensityBattery [kW/kg]: The po
        output:
            -Batterymass
    """
    BatteryMass = (PowerRequired - 0.9*MaxPowerFC) / Battery.PowerDensity /Battery.End_of_life 
    return  BatteryMass

def hover_energy_mass(PowerRequired: float ,MaxPowerFC: float, Battery: Battery, HoverTime:float) -> float:
    BatteryMass = (PowerRequired - MaxPowerFC) * HoverTime /3600 / Battery.EnergyDensity / Battery.Efficiency 
    return BatteryMass

class PropulsionSystem:

    def mass(echo: float , Mission: AircraftParameters , Battery: Battery, FuellCell: FuelCell, FuellTank: HydrogenTank, hovertime: float = 60, extra_power_during_hover_kW: float = 20 ) -> list[float]: #,MaxPowerFC:float,PowerDensityFC: float , PowerDensityBattery: float, EnergyDensityTank: float  ) -> list[float]:
        """Calculate total mass of the propulsion system
        input: 
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            
        returns:
            -Totalmass [kg]
            -FCmass[kg]: Fuell Cell mass
            -Batterymass[kg]"""
            
        
        #Initial sizing for cruise phase
        Tankmass,  EnergyBatterymass = energy_cruise_mass(Mission.mission_energy/ 3.6e6, echo, FuellTank, Battery, FuellCell) #convert to get to Wh
        FCmass, CruiseBatterymass = power_cruise_mass(Mission.cruisePower / 1e3, echo,FuellCell, Battery)
        #initial sizing for hovering phase
        HoverBatterymass = hover_mass(PowerRequired=Mission.hoverPower / 1e3 + extra_power_during_hover_kW ,MaxPowerFC= FuellCell.maxpower,Battery= Battery)
        HoverEnergyBatterymass = hover_energy_mass(PowerRequired= Mission.hoverPower /1e3 + extra_power_during_hover_kW, MaxPowerFC= FuellCell.maxpower ,Battery= Battery,HoverTime= hovertime)

        #heaviest battery is needed for the total mass
        Batterymass = np.zeros(len(echo))
        #need to check which batterymass is limiting at each echo and hardcoded it because i did not trust np.maximum as it gave some weird results
        for i in range(len(echo)):
            Batterymass[i] = max([HoverBatterymass, 2* HoverEnergyBatterymass, CruiseBatterymass[i], EnergyBatterymass[i]])

        #returning total mass and all component masss 
        Totalmass = Tankmass + FCmass + Batterymass #estimation from pim de boer that power density of fuel cell halves for system power density
        return  Totalmass, Tankmass, FCmass, Batterymass

    def volume(echo:float, Battery: Battery, FuellCell: FuelCell, FuellTank: HydrogenTank,Tankmass: float, Batterymass:float) -> tuple[float]:

        #calculating component mass
        TankVolume = Tankmass  * FuellTank.energyDensity / FuellTank.volumeDensity * 0.001
        FuellCellVolume = FuellCell.volume
        BatteryVolume = Battery.EnergyDensity / Battery.VolumeDensity * Batterymass *0.001

        TotalVolume = TankVolume + FuellCellVolume + BatteryVolume

        return TotalVolume , TankVolume, FuellCellVolume, BatteryVolume



def power_system_convergences(powersystem: Power, Mission: AircraftParameters):
    """
        does the computations for the final optimization/ convergences loop
    """
    IonBlock = Battery(Efficiency= 0.9)
    Pstack = FuelCell()
    Tank = HydrogenTank()
    #estimate power system mass
    nu = np.linspace(0,1,1000)
    Totalmass, Tankmass, FCmass, Batterymass= PropulsionSystem.mass(echo= np.copy(nu),
                                Mission= Mission,
                                Battery=IonBlock,
                                FuellCell= Pstack,
                                FuellTank= Tank )
    
    # font_size = 14
    # plt.title("Aetheria Power System distribution", fontsize = font_size)
    # plt.plot(nu, Totalmass + 80, linewidth = 3)
    # plt.yticks(np.linspace(0,1200,7), fontsize = font_size -2)
    # plt.xticks(np.linspace(0,1,6), fontsize = font_size -2, )
    # plt.xlabel(r"Fuel cell cruise fraction $ \nu $ [-]", fontsize = font_size)
    # plt.ylabel("Power system mass [kg]", fontsize = font_size)
    # plt.grid()
    # # plt.show()

    index_min_mass = np.argmin(Totalmass)
    NU = nu[index_min_mass]
    powersystemmass = Totalmass[index_min_mass] + 15 + 35 #kg mass radiator (15 kg) and mass air (35) subsystem. calculated outside this outside loop and will not change signficant 
    Batterymass = Batterymass[index_min_mass]
    powersystem.fuelcell_mass = FCmass


    powersystem.battery_energy = Batterymass * IonBlock.EnergyDensity * 3.6e6
    powersystem.battery_volume = Batterymass * IonBlock.EnergyDensity / IonBlock.VolumeDensity
    powersystem.battery_power = Batterymass * IonBlock.PowerDensity * 1e3
    powersystem.battery_mass = Batterymass
    # powersystem.fuelcell_mass = Pstack.mass
    powersystem.fuelcell_volume = Pstack.volume
    powersystem.h2_tank_mass = Tankmass[index_min_mass]
    powersystem.nu_FC_cruise_fraction = NU
    Mission.powersystem_mass = powersystemmass
    powersystem.powersystem_mass = powersystemmass


    return powersystem, Mission
