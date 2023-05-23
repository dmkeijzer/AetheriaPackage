import random
import sys
import pathlib as pl
import numpy as np
import pytest

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

from modules.powersizing import *
from modules.powersizing.powersystem import *

@pytest.fixture
def set_up(): 
    
    Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, 
                             charging_efficiency= random.uniform(0.4,0.8) ,depth_of_discharge= random.uniform(0.5,0.8), discharge_effiency=random.uniform(0.9,1))
    #fuelcell input
    VolumeDensityFuellCell = 3.25 #kW /l
    PowerDensityFuellCell = 3 #kW/kg
    effiencyFuellCell = 0.55

    #Tank input
    VolumeDensityTank = 0.8 #kWh/l
    EnergyDensityTank = 1.85 # kWh/kg
    J1Mission = MissionRequirements(EnergyRequired= 270,CruisePower=170, HoverPower= 360)
    FirstFC = FuellCellSizing(PowerDensityFuellCell,VolumeDensityFuellCell,effiencyFuellCell, 0)
    FuelTank = HydrogenTankSizing(EnergyDensityTank,VolumeDensityTank,0)
    return J1Mission, Liionbat, FuelTank, FirstFC


def test_energy_cruise_mass_conservation_energy():
    """Checking whether the total energy is constant"""
    echo = random.random()
    Mission, battery, h2tank, fuelcell = set_up()
    Tankmass, batterymass = energy_cruise_mass(EnergyRequired = Mission.EnergyRequired,
                                               echo = echo,
                                               Tank = h2tank,
                                               Battery = battery,
                                               FuellCell = fuelcell)
    TankEnergy = Tankmass * h2tank.EnergyDensity * fuelcell.Efficiency 
    batteryEnergy = batterymass * battery.Efficiency *battery.EnergyDensity * battery.DOD
    TotalEnergy = TankEnergy + batteryEnergy
    #print(TotalEnergy, Mission.EnergyRequired )
    assert np.isclose(TotalEnergy, Mission.EnergyRequired)

def test_power_cruise_mass_conservation_power():
    """Checking whether the total energy is constant"""
    echo = np.array([random.random()])
    Mission, battery, h2tank, fuelcell = set_up()
    FCmass, batterymass = power_cruise_mass(PowerRequired=Mission.CruisePower,
                                              echo= echo,
                                              FuellCell=fuelcell,
                                              Battery=battery)
    FCpower = FCmass * fuelcell.PowerDensity 
    batterypower= batterymass * battery.PowerDensity
    Totalpower= FCpower + batterypower
    #print(Totalpower[0], Mission.CruisePower )
    assert np.isclose(Totalpower[0], Mission.CruisePower)

def test_hover_mass_conservation_power():
    Mission, battery, h2tank, fuelcell = set_up()
    maxpowerfc = random.uniform(100,170)
    batterymass = hover_mass(Mission.HoverPower,maxpowerfc,battery)
    check = maxpowerfc + batterymass * battery.PowerDensity
    #print(check, Mission.HoverPower)
    assert np.isclose(check, Mission.HoverPower)

def test_hover_energy_conservation_energy():
    Mission, battery, h2tank, fuelcell = set_up()
    maxpowerfc = random.uniform(100,170)
    hovertime = random.uniform(10,60)
    batterymass = hover_energy_mass(PowerRequired = Mission.HoverPower,
                                   MaxPowerFC= maxpowerfc,
                                   Battery=battery,
                                   HoverTime=hovertime)
    batteryEnergy = batterymass * battery.EnergyDensity * battery.Efficiency 
    fuelcellEnergy = maxpowerfc * hovertime /3600
    check = batteryEnergy + fuelcellEnergy
    hoverEnergy = Mission.HoverPower * hovertime /3600
    #print(hoverEnergy, check)
    assert np.isclose(hoverEnergy,check)

def test_PropulsionSystem_mass_conservation_energy():
    Mission, battery, h2tank, fuelcell = set_up()
    echo = np.arange(0,1.1,0.1)
    Totalmass, Tankmass, FCmass, Batterymass, masscooling = PropulsionSystem.mass(echo= echo,
                                                                                  Mission= Mission,
                                                                                  Battery=battery,
                                                                                  FuellCell=fuelcell,
                                                                                  FuellTank=h2tank)
    #calculating usefull energy in each subsystem
    EnergyTank = Tankmass * h2tank.EnergyDensity * fuelcell.Efficiency 
    BatteryEnergy = Batterymass * battery.EnergyDensity * battery.Efficiency * battery.DOD
    TotalEnergy = EnergyTank + BatteryEnergy

    #print(TotalEnergy, Mission.EnergyRequired)
    assert(all(TotalEnergy >= Mission.EnergyRequired - 1e-10))

def test_PropulsionSystem_mass_meet_power_requirements():
    Mission, battery, h2tank, fuelcell = set_up()
    echo = np.arange(0,1.1,0.1)
    Totalmass, Tankmass, FCmass, Batterymass, masscooling = PropulsionSystem.mass(echo= echo,
                                                                                  Mission= Mission,
                                                                                  Battery=battery,
                                                                                  FuellCell=fuelcell,
                                                                                  FuellTank=h2tank)
    #calculating usefull energy in each subsystem
    FCPower = FCmass * fuelcell.PowerDensity
    BatteryPower = Batterymass * battery.PowerDensity
    Totalpower = FCPower + BatteryPower

    #substracting very small value to prevent error due to the floating point system
    #print(Totalpower, Mission.CruisePower)
    assert(all(Totalpower >= Mission.CruisePower - 1e-10))
    #print(Totalpower, Mission.HoverPower)
    assert(all(Totalpower >= Mission.CruisePower - 1e-10))





'''
test_energy_cruise_mass_conservation_energy()
test_power_cruise_mass_conservation_power()
test_hover_mass_conservation_power()
test_hover_energy_conservation_energy()
test_PropulsionSystem_mass_conservation_energy()
test_PropulsionSystem_mass_meet_power_requirements()'''