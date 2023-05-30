@dataclass
class Wing():
    surface: float
    taper: float
    aspectratio: float
    span: float = None
    chord_root: float = None
    chord_tip: float = None
    chord_mac: float = None
    y_mac: float = None
    tan_sweep_LE: float = None
    quarterchord_sweep: float = None
    X_lemac: float = None
    effective_aspectratio: float = None
    effective_span: float = None