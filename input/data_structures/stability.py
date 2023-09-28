from pydantic import BaseModel, FilePath
import json
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

class Stab(BaseModel):
    label : str = "Stability"
    Cm_de: float | None = None
    Cn_dr: float | None = None
    Cxa: float | None = None
    Cxq: float | None = None
    Cza: float | None = None
    Czq: float | None = None
    Cma: float | None = None
    Cmq: float | None = None
    Cz_adot: float | None = None
    Cm_adot: float | None = None
    muc: float | None = None
    Cxu: float | None = None
    Czu: float | None = None
    Cx0: float | None = None
    Cz0: float | None = None
    Cmu: float | None = None
    Cyb: float | None = None
    Cyp: float | None = None
    Cyr: float | None = None
    Clb: float | None = None
    Clp: float | None = None
    Clr: float | None = None
    Cnb: float | None = None
    Cnp: float | None = None
    Cnr: float | None = None
    Cy_dr: float | None = None
    Cy_beta_dot: float | None = None
    Cn_beta_dot: float | None = None
    mub: float | None = None
    sym_eigvals: float | None = None
    asym_eigvals: float | None = None
    cg_front: float | None = None
    cg_rear: float | None = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        return cls(**data["Wing"])
        
    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Stability"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)
