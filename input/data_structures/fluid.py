from dataclasses import dataclass

@dataclass
class Fluid:
    """Stores fluid paramers
    """
    viscosity: float = None
    thermal_conductivity: float = None
    heat_capacity: float = None
    density: float = None
