
import numpy as np
import os
import json
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import *
from modules.powersizing import PropulsionSystem
from modules.stab_ctrl.vtail_sizing_optimal import size_vtail_opt
from modules.stab_ctrl.wing_loc_horzstab_sizing import wing_location_horizontalstab_size
from modules.planform.planformsizing import wing_planform

IonBlock = Battery(Efficiency= 0.9)
Pstack = FuelCell()
Tank = HydrogenTank(energyDensity=1.8, volumeDensity=0.6, cost= 16)
Mission = PerformanceParameters()


wing  =  Wing()
horizontal_tail = HorTail()
fuselage = Fuselage()
vtail = VeeTail()
stability = Stab()
aero = Aero()







#planform sizing
wing.load()
wing = wing_planform(wing)
wing.dump()

#power system sizing
Mission.load()
nu = np.arange(0,1.001,0.005)
Totalmass, Tankmass, FCmass, Batterymass= PropulsionSystem.mass(echo= np.copy(nu),
                             Mission= Mission,
                             Battery=IonBlock,
                             FuellCell= Pstack,
                             FuellTank= Tank )

index_min_mass = np.where(Totalmass == min(Totalmass))
NU = nu[index_min_mass][0]
powersystemmass = Totalmass[index_min_mass][0]
Batterymass = Batterymass[index_min_mass][0]
fuelcellmass = Pstack.mass
print(f"power system mass: {powersystemmass}")


#stability and control
wing.load()
horizontal_tail.load()
fuselage.load()
vtail.load()
stability.load()
aero.load()

wing,horizontal_tail,fuselage,vtail, stability = size_vtail_opt(WingClass=  wing,
                                                                HorTailClass= horizontal_tail,
                                                                FuseClass= fuselage,
                                                                VTailClass= vtail, 
                                                                StabClass=stability,
                                                                Aeroclass= aero,
                                                                b_ref= 2, #!!!!!!!!! please update value when we get it
                                                                stepsize= 1e-1 ) 

print(f"V-tail surface:  {vtail.surface} ")
# WingboxClass =  Wing()
