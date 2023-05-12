import json
from hover_controllabilty import HoverControlCalcTandem, HoverControlCalcBase


f = open('W1_constants.json')
data = json.load(f)

istandem = True
if data["name"] == "J1":
    istandem = False

lfus = 10.75

## From JSON
StotS = 1.5 #data[StotS]
Sproj = StotS * data["S"]
m = data["mtom"]
n_rot_f = data["nrotf"]    ##must be even
n_rot_r = data["nrotr"]     ##must be even

#spacing = data["d_prop"]    # propeller diameter


if istandem:
    rot_y_range_f = [0, data["b1"]/2]
    rot_y_range_r = [0, data["b2"]/2]
else:
    rot_y_range_f = [0, data["b"]/2]
    rot_y_range_r = [data["bh"] / 2, data["bh"] / 2]
K = 9.80665*m*1.2*(1+1/(n_rot_r+n_rot_f-1))*(1+1.225*4*Sproj/(m*9.80665))/(n_rot_r+n_rot_f)

## NOT from JSON
ku = 0.1
x_wf = 0.5/8.75 * lfus
x_wr = 7.38/8.75 * lfus

hovercalc = HoverControlCalcTandem(m, n_rot_f, n_rot_r, x_wf, x_wr, rot_y_range_f, rot_y_range_r, K, ku)

#hovercalc_same_instance= HoverControlCalcBase(m, rotors)


x_min = 0
x_max = lfus
dx = 0.01
failure_eval_cg = [0.55*lfus,0]
n_failures = [0,1,2]
#cg_range = hovercalc_same_instance.calc_crit_x_cg_range(x_min, x_max, dx,failure_eval_cg, n_failures)
print(hovercalc.acai(failure_eval_cg))
print(hovercalc.calc_x_cg_range(x_min, x_max, dx, 0))
print(hovercalc.calc_crit_x_cg_range(x_min, x_max, dx,failure_eval_cg, n_failures))