import pytest
import sys
import pathlib as pl
from configparser import ConfigParser
import os

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

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
def veetail(config):
    return VeeTail.load(config["test_settings"]["dataset"])

@pytest.fixture
def aircraft(config):
    return AircraftParameters.load(config["test_settings"]["dataset"])

@pytest.fixture
def fuselage(config):
    return Fuselage.load(config["test_settings"]["dataset"])

@pytest.fixture
def aero(config):
    return Aero.load(config["test_settings"]["dataset"])

@pytest.fixture
def engine(config):
    return Engine.load(config["test_settings"]["dataset"])
