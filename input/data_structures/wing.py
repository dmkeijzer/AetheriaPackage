from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

@dataclass
class Wing():
    surface: float = None
    taper: float = None
    aspectratio: float = None
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

    def load(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        self.surface =  data["S"]
        self.taper =  data["taper"]
        self.aspectratio =  data["A"]
        self.span =  data["b"]
        self.chord_root =  data["c_root"]
        self.chord_tip =  data["c_tip"]
        self.chord_mac =  data["mac"]
        self.y_mac =  data["y_mac"]
        self.tan_sweep_LE =  data["sweep_le"]
        self.quarterchord_sweep =  data["sweep_le"]
        self.X_lemac =  data["x_lemac"]
        # self.effective_aspectratio =  data[""] # Left out for now since it is not implemented yet
        # self.effective_span =  data[""] # Left out for now since it is not implemented yet
    
    def dump(self):
        data = {
            "S": self.surface,
            "taper": self.taper,
            "A": self.aspectratio,
            "b": self.span,
            "c_root": self.chord_root,
            "c_tip": self.chord_tip,
            "mac": self.chord_mac,
            "y_mac": self.y_mac,
            "sweep_le": self.tan_sweep_LE,
            "quarterchord_sweep": self.quarterchord_sweep,
            "x_lemac": self.X_lemac
        }

        with open(r"output/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)


if __name__ == "__main__":
    WingClass = Wing()
    WingClass.load()

    print(WingClass.surface)
    print(WingClass.chord_tip)


