import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import matplotlib.pyplot  as plt
import plotly.graph_objs as go
from IPython.display import display
import plotly.offline as pyo

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
from input.data_structures.vee_tail import VeeTail
from input.data_structures.wing import Wing
from input.data_structures.performanceparameters import PerformanceParameters
from modules.aero.avl_access import get_lift_distr
from modules.structures.wingbox_optimizer_extra import Wingbox
from modules.structures.pylon_design import PylonSizing


WingClass = Wing()
EngineClass = Engine()
MaterialClass = Material()
AeroClass = Aero()
TailClass = VeeTail()
PerfClass = PerformanceParameters()

WingClass.load()
EngineClass.load()
MaterialClass.load()
AeroClass.load()
TailClass.load()


WingboxClass = Wingbox(WingClass, EngineClass, MaterialClass, AeroClass, PerfClass, False)
PylonClass =  PylonSizing(EngineClass, 2.6)


tsp= WingClass.spar_thickness
hst= WingClass.stringer_height
tst= WingClass.stringer_thickness
tsk= WingClass.stringer_thickness
ttb= WingClass.torsion_bar_thickness

X = [tsp,hst,tst,tsk,ttb]

dict_properties = {}
wing_section_loc = [1/16, 3/16, 5/16, 7/16]

for label , loc in  enumerate(wing_section_loc):
    y = WingClass.span*loc

    # Get corrected indices from y array 
    idx = np.where(WingboxClass.y < y)[0][-1]
    prop = BeamProp()
    A = WingboxClass.chord(y)**2*0.6**0.12
    E = E_composite
    prop.A = A 
    prop.E = E_composite
    scf = 5/6.
    prop.G = scf*E_composite/2/(1+0.3)
    Izz =  (WingboxClass.I_xx(X)[idx] + WingboxClass.I_xx(X)[idx + 1])/2
    Iyy = (WingboxClass.I_zz(X)[idx] + WingboxClass.I_zz(X)[idx + 1])/2
    prop.Izz = Izz
    prop.Iyy = Iyy
    prop.intrho = rho_composite*A
    prop.intrhoy2 = rho_composite*Izz
    prop.intrhoz2 = rho_composite*Iyy
    prop.J = Izz + Iyy
    dict_properties[str(label)] = prop

# Properites of the nacelle column


pylon_x = [0.1218, 0.014]

nacelle_prop = BeamProp()
nacelle_prop.A = PylonClass.get_area(pylon_x)
nacelle_prop.E = E_composite
I = PylonClass.I_xx(pylon_x)
nacelle_prop.G = scf*E_composite/2/(1+0.3)
nacelle_prop.Izz = I
nacelle_prop.Iyy = I
nacelle_prop.intrho = rho_composite*PylonClass.get_area(pylon_x)
nacelle_prop.intrhoy2 = rho_composite*I
nacelle_prop.intrhoz2 = rho_composite*I
nacelle_prop.J = 2*I



nodes = { #nid: [x, y, z]
    1000 : [WingClass.x_lewing + TailClass.length_wing2vtail , 0.0, 0.],
    1001: [WingClass.x_lewing + TailClass.length_wing2vtail, TailClass.span/2*np.cos(TailClass.dihedral),TailClass.span/2*np.sin(TailClass.dihedral)],
    1002 : [WingClass.x_lewing, 0. , 0.],
    1003 : [WingClass.x_lewing, WingClass.span/8, 0.],
    1004 : [WingClass.x_lewing, WingClass.span/8*2, 0.],
    1005 : [WingClass.x_lewing, WingClass.span/8*3, 0.],
    1006 : [WingClass.x_lewing, WingClass.span/2, 0.],
    1007 : [WingClass.x_lewing - EngineClass.pylon_length, WingClass.span/8*2, 0.],
    1008: [WingClass.x_lewing + TailClass.length_wing2vtail, -1*TailClass.span/2*np.cos(TailClass.dihedral),TailClass.span/2*np.sin(TailClass.dihedral)], # lhs vtail node
    1009 : [WingClass.x_lewing, -1*WingClass.span/8, 0.],
    1010 : [WingClass.x_lewing, -1*WingClass.span/8*2, 0.],
    1011 : [WingClass.x_lewing, -1*WingClass.span/8*3, 0.],
    1012 : [WingClass.x_lewing, -1*WingClass.span/2, 0.],
    1013 : [WingClass.x_lewing - EngineClass.pylon_length, -1*WingClass.span/8*2, 0.],
    1014 : [0., 0.,  0.],
}
elements = { #eid: [prop, node 1, node 2]
    1: [prop, 1000, 1001],
    2: [prop, 1000, 1002],
    3: [dict_properties["0"], 1002, 1003],
    4: [dict_properties["1"], 1003, 1004],
    5: [dict_properties["2"], 1004, 1005],
    6: [dict_properties["3"], 1005, 1006],
    7: [nacelle_prop, 1004, 1007],
    8: [prop, 1000, 1008], # Element lhs vtail
    9: [dict_properties["0"], 1002, 1009],
    10: [dict_properties["1"], 1009, 1010],
    11: [dict_properties["2"], 1010, 1011],
    12: [dict_properties["3"], 1011, 1012],
    13: [nacelle_prop, 1010, 1013],
    14: [prop, 1002, 1014]
}


# Extract the coordinates for plotting
x = []
y = []
z = []
str_lst = []

for node_id, coords in nodes.items():
    x.append(coords[0])
    y.append(coords[1])
    z.append(coords[2])
    str_lst.append(str(node_id))

# Plot the nodes in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', label='Nodes')

for x1, y1, z1, str_ele in zip(x,y,z, str_lst):
    ax.text(x1, y1, z1 + 0.2, str_ele, (1,0,0))

# Plot the elements in 3D
for elem_id, nodes1 in elements.items():
    node1 = int(str(nodes1[1])[-2:])
    node2 = int(str(nodes1[2])[-2:])

    ax.plot([x[node1], x[node2]], [y[node1], y[node2]], [z[node1], z[node2]], c='red' )

# Set labels and grid
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(True)

# Show the plot
# Make axes equal
ax.set_xlim([2, 10])
ax.set_ylim([-6, 6])
ax.set_zlim([-2, 2])
plt.show()

#---------------------------- Compute actual eigenfrequencie -------------------
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

    if eid ==  1:
        beam.update_rotation_matrix(0., -1*np.sin(TailClass.dihedral), 1*np.cos(TailClass.dihedral), ncoords_flatten) #Beam local y axis orientation 
    if eid ==  8:
        beam.update_rotation_matrix(0., 1*np.sin(TailClass.dihedral), 1*np.cos(TailClass.dihedral), ncoords_flatten) #Beam local y axis orientation 
    else:
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


print('concentrated masses')

name_nid = {
'engine_vtail_rhs': 1001,
'engine_vtail_lhs': 1008,
'engine_inboard_rhs': 1007,
'engine_inboard_lhs': 1013,
'engine_outboard_rhs': 1006,
'engine_outboard_lhs': 1012,
}

masses = {
'engine_vtail_rhs': EngineClass.mass_pertotalengine,
'engine_vtail_lhs':  EngineClass.mass_pertotalengine,
'engine_inboard_rhs': EngineClass.mass_pertotalengine,
'engine_inboard_lhs': EngineClass.mass_pertotalengine,
'engine_outboard_rhs': EngineClass.mass_pertotalengine,
}

for name, mass in masses.items():
    for i in range(3):
        pos = DOF*nid_pos[name_nid[name]] + i
        M[pos, pos] += mass

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
print(omegan/(2*np.pi), 'hz')

for i in range(6):
    mode = i

    scale = 20
    dx = eigvecs[0::DOF, mode]*scale
    dy = eigvecs[1::DOF, mode]*scale
    dz = eigvecs[2::DOF, mode]*scale

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for beam in beams:
        x_coords = [x[nid_pos[beam.n1]], x[nid_pos[beam.n2]]]
        y_coords = [y[nid_pos[beam.n1]], y[nid_pos[beam.n2]]]
        z_coords = [z[nid_pos[beam.n1]], z[nid_pos[beam.n2]]]
        ax.plot(x_coords, y_coords, z_coords, color='blue', linewidth=1)

        x_coords = [(x + dx)[nid_pos[beam.n1]], (x + dx)[nid_pos[beam.n2]]]
        y_coords = [(y + dy)[nid_pos[beam.n1]], (y + dy)[nid_pos[beam.n2]]]
        z_coords = [(z + dz)[nid_pos[beam.n1]], (z + dz)[nid_pos[beam.n2]]]
        ax.plot(x_coords, y_coords, z_coords, color='red', linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(-2, 2)
    ax.set_title(f'Mode {mode}')

    plt.show()



#     # Create a new figure

#     fig = go.Figure()

#     # Iterate over the sequences and add line segments to the figure

#     for beam in beams:

#         x_coords = [x[nid_pos[beam.n1]], x[nid_pos[beam.n2]]]
#         y_coords = [y[nid_pos[beam.n1]], y[nid_pos[beam.n2]]]
#         z_coords = [z[nid_pos[beam.n1]], z[nid_pos[beam.n2]]]

#         fig.add_trace(go.Scatter3d(
#             x=x_coords,
#             y=y_coords,
#             z=z_coords,
#             mode='lines',
#             line=dict(color='blue', width=5)
#         ))

#         x_coords = [(x+dx)[nid_pos[beam.n1]], (x+dx)[nid_pos[beam.n2]]] 
#         y_coords = [(y+dy)[nid_pos[beam.n1]], (y+dy)[nid_pos[beam.n2]]]
#         z_coords = [(z+dz)[nid_pos[beam.n1]], (z+dz)[nid_pos[beam.n2]]]

#         fig.add_trace(go.Scatter3d(
#             x=x_coords,
#             y=y_coords,
#             z=z_coords,
#             mode='lines',
#             line=dict(color='red', width=5)
#         ))


#     # Set the layout of the figure

#     fig.update_layout(

#     scene=dict(
#         xaxis=dict(title='X'),
#         yaxis=dict(title='Y'),
#         zaxis=dict(title='Z', range=[-2, 2])
#     ),

#     title=f'Mode {mode}'

#     )
#     # Display the figure

#     fig.show()
