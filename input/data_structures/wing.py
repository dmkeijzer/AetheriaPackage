from pydantic import BaseModel, FilePath
import json
import sys
import os 
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.GeneralConstants import *

class Wing(BaseModel):
    aspect_ratio: float 
    quarterchord_sweep: float 
    taper: float 
    washout: float 
    """"IN RADIANS PLEASE"""
    label : str = "Wing"
    surface: float | None = None
    span: float | None = None
    chord_root: float | None = None
    chord_tip: float | None = None
    chord_mac: float | None = None
    y_mac: float| None = None
    sweep_LE: float| None = None
    x_lemac: float | None= None
    x_lewing: float| None = None
    thickness_to_chord: float = 0.12
    wing_weight: float| None = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Wing"])
        except:
            raise Exception(f"There was an error when loading in {cls}")
        
    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Wing"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)
    


if __name__ == "__main__":
    wing = Wing.load(r"C:\Users\damie\OneDrive\Desktop\Damien\DSE\AetheriaPackage\input\initial_estimate.json")



