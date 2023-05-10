from Structure import Structure

Cl = Structure.process_dist("""
 0. 0.07012725 0.1396002  0.20779006 0.27412298 0.33811438
 0.39940413 0.45778292 0.51319906 0.56574244 0.61561101 0.66307066
 0.7084175  0.75194759 0.79393547 0.83462059 0.87420005 0.91282594
 0.9506056  0.98760375 1.02384567 1.05932065 1.09398561 1.12776843
 1.16057106 1.19227226 1.22273008 1.25178397 1.27925667 1.30495596
 1.3286762  1.35019983""")

Cd = Structure.process_dist("""
 0 0.01634525 0.02443397 0.03061296 0.03503482 0.03792653
 0.03956237 0.04023399 0.0402226  0.03977798 0.03910679 0.03836955
 0.03768385 0.03713102 0.03676384 0.03661383 0.03669755 0.03702143
 0.03758554 0.03838613 0.03941739 0.04067269 0.04214514 0.04382809
 0.04571515 0.0478003  0.0500777  0.05254152 0.05518571 0.0580037
 0.06098806 0.06413016""")

inputs = dict(MAC = 1.265147796494402, # Mean Aerodynamic Chord [m]
cruise = True, # boolean depending on whether cruise or take-off is being considered
w_back = True, # will analyse the back wing if True
taper = 0.45, # [-]
rootchord = 1.6651718350228892, # [m]
thicknessChordRatio = 0.17, # [-]
xAC = 0.25, # [-] position of ac with respect to the chord
mtom = 3024.8012022968796, # maximum take-off mass from statistical data - Class I estimation
S1 = 9.910670535618632, 
S2 = 9.910670535618632, # surface areas of wing one and two
span1 = 8.209297146662843,
span2 = 8.209297146662843,
nmax = 3.43, # maximum load factor
Pmax = 17, # this is defined as maximum perimeter in Roskam, so i took top down view of the fuselage perimeter
lf = 7.348878876267166, # length of fuselage
wf = 1.38, # width of fuselage
m_pax = 88, # average mass of a passenger according to Google
n_prop = 12, # number of engines
n_pax = 5, # number of passengers (pilot included)
pos_fus = 2.9395515505068666, # fuselage centre of mass away from the nose
pos_lgear = 3.875, # landing gear position away from the nose
pos_frontwing = 0.5,
pos_backwing = 6.1, # positions of the wings away from the nose
m_prop = [502.6006543358783/12]*12, # list of mass of engines (so 30 kg per engine with nacelle and propeller)
pos_prop =  [-0.01628695, -0.01628695, -0.01628695, -0.01628695, -0.01628695, -0.01628695, 
             5.58371305,  5.58371305,  5.58371305,  5.58371305,  5.58371305,  5.58371305], # 8 on front wing and 8 on back wing
Mac = 0.002866846692576361, # aerodynamic moment around AC
flighttime = 1.5504351809662442, # [hr]
turnovertime = 2, # we dont actually need this xd
takeofftime = 262.839999999906/3600,
engineMass = 502.6006543358783 * 9.81 / 8,
Thover = 34311.7687171136/12,
Tcruise = 153.63377687614096,
p_pax = [1.75, 3.75, 3.75, 6, 6],
battery_pos = 0.5,
cargo_m = 35, cargo_pos = 6.5, battery_m = 886.1868116321529,
materialdata = 'data/materials.csv',
CL = Cl, CD = Cd, # Aerodynamics
Vcruise = 72.18676185339652, # Cruise speed [m/s]
)

state = dict(nStrT=2, nStrB=1,
            thicknessOfSkin=1e-3, thicknessOfSpar=18*1e-3,
            thicknessOfStringer=1e-3, ntofit=20, stringerMat = dict(material='Al 7075', Condition='T6'),
               skinMat = dict(material='Al 7075', Condition='T6'))

struct = Structure(**(inputs | state ))

struct.compute_stresses(**state)
print('Ixx:', struct.loads.wing(0).Ixx())
print('Lug:', struct.design_lug())
topStr, botStr, tsk, tsp, wingmass = struct.optimize()

print(f"Final {topStr, botStr, tsk, tsp, wingmass = }")

topStr, botStr, tsk, tsp, wingmass = struct.compute_tip()

print(f"Final Tip {topStr, botStr, tsk, tsp, wingmass = }")

print(f"{struct.design_lug()}")