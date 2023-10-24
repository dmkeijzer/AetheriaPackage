import sys
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[0]))
sys.path.append(os.path.dirname(__file__))

from input.data_structures.wing import Wing
from input.data_structures.battery import Battery
from input.data_structures.fuellCell import FuelCell
from input.data_structures.hydrogenTank import HydrogenTank
from input.data_structures.aircraft_parameters import AircraftParameters
from input.data_structures.fuselage import Fuselage
from input.data_structures.aero import Aero
from input.data_structures.fluid import Fluid
from input.data_structures.engine import Engine
from input.data_structures.radiator import Radiator
from input.data_structures.stability import Stab
from input.data_structures.vee_tail import VeeTail
from input.data_structures.power import Power
