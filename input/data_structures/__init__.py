import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[0]))

from wing import Wing
from battery import Battery
from fuellCell import FuelCell
from hydrogenTank import HydrogenTank
from performanceparameters import PerformanceParameters
from fuselage import Fuselage
from aero import Aero
from hor_tail import HorTail