import pytest
import sys
import pathlib as pl
from configparser import ConfigParser
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import *

@pytest.fixture
def config():
    config = ConfigParser()
    config.read(r"tests/setup/config_test.cfg")
    return config


@pytest.fixture
def wing(config):
    return Wing.load(config["test_settings"]["dataset"])

@pytest.fixture
def aircraft(config):
    return AircraftParameters.load(config["test_settings"]["dataset"])


@pytest.fixture
def aero(config):
    return Aero.load(config["test_settings"]["dataset"])

@pytest.fixture
def engine(config):
    return Engine.load(config["test_settings"]["dataset"])

def test_fixtures( wing, aircraft, aero, engine):
    print(wing)
    print(aircraft)
    print(aero)
    print(engine)

