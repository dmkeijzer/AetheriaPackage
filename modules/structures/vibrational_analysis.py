import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d import BeamC, BeamCData, BeamCProbe, DOF, INT, DOUBLE

from math import *
import sys
import pathlib as pl
import os
import matplotlib.pyplot as plt
import time

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures.GeneralConstants import *
from input.data_structures.aero import Aero
from input.data_structures.engine import Engine
from input.data_structures.material import Material
from input.data_structures.wing import Wing
from modules.aero.avl_access import get_lift_distr

WingClass = Wing()
EngineClass = Engine()
MaterialClass = Material()
AeroClass = Aero()

WingClass.load()
EngineClass.load()
MaterialClass.load()
AeroClass.load()

class VibrationalAnalysis(wingclass = Wing, engineclass = Engine, materialclass = Material, aeroclass = Aero):
    E = materialclass.E # Pa
    rho = materialclass # kg/m3
    b = 0.05 # m
    h = 0.07 # m
    A = h*b
    Izz = b*h**3/12
    Iyy = b**3*h/12

    prop = BeamProp()
    prop.A = A
    prop.E = E
    scf = 5/6.
    prop.G = scf*E/2/(1+0.3)
    prop.Izz = Izz
    prop.Iyy = Iyy
    prop.intrho = rho*A
    prop.intrhoy2 = rho*Izz
    prop.intrhoz2 = rho*Iyy
    prop.J = Izz + Iyy

    nodes = { #nid: [x, y, z]
        1000: [0., 0.0, 0.],
        1001: [0., 0.1, 0.],
        1002: [0., 0.2, 0.],
        1003: [0., 0.3, 0.],
        1004: [0., 0.4, 0.],
    }
    elements = { #eid: [prop, node 1, node 2]
        1: [prop, 1000, 1001],
        2: [prop, 1001, 1002],
        3: [prop, 1002, 1003],
        4: [prop, 1003, 1004],
    }

    pos = 0
    nid_pos = {}
    ncoords = []
    nids = []
    for nid, xyz in nodes.items():
    nid_pos[nid] = pos
    pos += 1
    ncoords.append(xyz)
    nids.append(nid)
    ncoords = np.asarray(ncoords, dtype=float)
    x, y, z = ncoords.T
    ncoords_flatten = ncoords.flatten()

    num_elements = len(elements)
    print('num_elements', num_elements)

    p = BeamCProbe()
    data = BeamCData()

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
    Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
    Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    N = DOF*len(nids)
    print('num_DOF', N)


    beams = []
    init_k_KC0 = 0
    init_k_M = 0

    for eid, (prop_i, n1, n2) in elements.items():
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        beam = BeamC(p)
        beam.init_k_KC0 = init_k_KC0
        beam.init_k_M = init_k_M
        beam.n1 = n1
        beam.n2 = n2
        beam.c1 = DOF*pos1
        beam.c2 = DOF*pos2
        beam.update_rotation_matrix(0., 0., 1, ncoords_flatten) #Beam local y axis orientation
        beam.update_probe_xe(ncoords_flatten)
        beam.update_KC0(KC0r, KC0c, KC0v, prop_i)
        beam.update_M(Mr, Mc, Mv, prop_i)
        beams.append(beam)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_M += data.M_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

    print('sparse KC0 and M created')

    bk = np.zeros(N, dtype=bool)

    at_root = np.isclose(y, 0.)
    bk[0::DOF][at_root] = True
    bk[1::DOF][at_root] = True
    bk[2::DOF][at_root] = True
    bk[3::DOF][at_root] = True
    bk[4::DOF][at_root] = True
    bk[5::DOF][at_root] = True
    bu = ~bk

    Kuu = KC0[bu, :][:, bu]
    Muu = M[bu, :][:, bu]

    num_eigenvalues = 6
    eigvals, eigvecsu = eigsh(A=Kuu, M=Muu, sigma=-1., which='LM',
            k=num_eigenvalues, tol=1e-3)
    eigvecs = np.zeros((N, num_eigenvalues))
    eigvecs[bu] = eigvecsu
    omegan = eigvals**0.5
    print(omegan, 'rad/s')







    mode = 0

    scale = 1.
    dx = eigvecs[0::DOF, mode]*scale
    dy = eigvecs[1::DOF, mode]*scale
    dz = eigvecs[2::DOF, mode]*scale

    fig = go.Figure(
        data=[
        go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='black', width=4)),
        go.Scatter3d(x=x+dx, y=y+dy, z=z+dz, mode='lines', line=dict(color='red', width=4)),
        ])
    fig.show()