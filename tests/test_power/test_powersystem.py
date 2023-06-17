import random
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import Battery, PerformanceParameters, FuelCell, HydrogenTank
from modules.powersizing.powersystem import energy_cruise_mass, power_cruise_mass,hover_mass,hover_energy_mass, PropulsionSystem

def set_up(): 
    
    IonBlock = Battery(Efficiency= 0.9)
    Pstack = FuelCell()
    Tank = HydrogenTank()
    Mission = PerformanceParameters()
    Tank.load()
    Mission.load()
    return Mission, IonBlock, Tank, Pstack


def test_energy_cruise_mass_conservation_energy():
    """Checking whether the total energy is constant"""
    echo = random.random()
    Mission, battery, h2tank, fuelcell = set_up()
    Tankmass, batterymass = energy_cruise_mass(EnergyRequired = Mission.energyRequired /3.6e6,
                                               echo = echo,
                                               Tank = h2tank,
                                               Battery = battery,
                                               FuellCell = fuelcell)
    TankEnergy = Tankmass * h2tank.energyDensity * fuelcell.efficiency 
    batteryEnergy = batterymass * battery.Efficiency *battery.EnergyDensity * battery.End_of_life
    TotalEnergy = TankEnergy + batteryEnergy
    assert np.isclose(TotalEnergy, Mission.energyRequired/3.6e6)

def test_power_cruise_mass_conservation_power():
    """Checking whether the total energy is constant"""
    echo = np.array([random.random()])
    Mission, battery, h2tank, fuelcell = set_up()
    FCmass, batterymass = power_cruise_mass(PowerRequired=Mission.cruisePower/1e3,
                                              echo= echo,
                                              FuellCell=fuelcell,
                                              Battery=battery)
    FCpower = Mission.cruisePower/1e3 * echo
    batterypower= batterymass * battery.PowerDensity * battery.End_of_life * battery.Depth_of_discharge
    Totalpower= FCpower + batterypower
    #print(Totalpower[0], Mission.CruisePower )
    assert np.isclose(Totalpower[0], Mission.cruisePower/1e3)

def test_hover_mass_conservation_power():
    Mission, battery, h2tank, fuelcell = set_up()
    maxpowerfc = random.uniform(100,170)
    batterymass = hover_mass(Mission.hoverPower/1e3,maxpowerfc,battery)
    check = maxpowerfc*0.9 + batterymass * battery.PowerDensity * battery.End_of_life
    assert np.isclose(check, Mission.hoverPower/1e3)

def test_hover_energy_conservation_energy():
    Mission, battery, h2tank, fuelcell = set_up()
    maxpowerfc = random.uniform(100,170)
    hovertime = random.uniform(10,60)
    batterymass = hover_energy_mass(PowerRequired = Mission.hoverPower /1e3,
                                   MaxPowerFC= maxpowerfc,
                                   Battery=battery,
                                   HoverTime=hovertime)
    batteryEnergy = batterymass * battery.EnergyDensity * battery.Efficiency  
    fuelcellEnergy = maxpowerfc * hovertime /3600
    check = batteryEnergy + fuelcellEnergy
    hoverEnergy = Mission.hoverPower/1e3 * hovertime /3600
    assert (hoverEnergy- check < 1e-6)

def test_PropulsionSystem_mass_conservation_energy():
    Mission, battery, h2tank, fuelcell = set_up()
    echo = np.arange(0,1.1,0.1)
    Totalmass, Tankmass, FCmass, Batterymass = PropulsionSystem.mass(echo= echo,
                                                                                  Mission= Mission,
                                                                                  Battery=battery,
                                                                                  FuellCell=fuelcell,
                                                                                  FuellTank=h2tank)
    #calculating usefull energy in each subsystem
    EnergyTank = Tankmass * h2tank.energyDensity * fuelcell.efficiency 
    BatteryEnergy = Batterymass * battery.EnergyDensity * battery.Efficiency * battery.End_of_life
    TotalEnergy = EnergyTank + BatteryEnergy
    #print(TotalEnergy, Mission.EnergyRequired)
    assert(all(TotalEnergy >= Mission.energyRequired/3.6e6 - 1e-10))

def test_PropulsionSystem_mass_meet_power_requirements():
    Mission, battery, h2tank, fuelcell = set_up()
    echo = np.arange(0,1.1,0.1)
    Totalmass, Tankmass, FCmass, Batterymass= PropulsionSystem.mass(echo= echo,
                                                                                  Mission= Mission,
                                                                                  Battery=battery,
                                                                                  FuellCell=fuelcell,
                                                                                  FuellTank=h2tank)
    #calculating usefull energy in each subsystem
    FCPower = Mission.cruisePower /1e3 * echo
    BatteryPower = Batterymass * battery.PowerDensity * battery.End_of_life * battery.Efficiency
    Totalpower = FCPower + BatteryPower

    #substracting very small value to prevent error due to the floating point system
    #print(Totalpower, Mission.CruisePower)
    assert(all(Totalpower >= Mission.climbPower/1e3 - 1e-10))






""" 
test_energy_cruise_mass_conservation_energy()
test_power_cruise_mass_conservation_power()
test_hover_mass_conservation_power()
test_hover_energy_conservation_energy()

test_PropulsionSystem_mass_conservation_energy()

test_PropulsionSystem_mass_meet_power_requirements()"""