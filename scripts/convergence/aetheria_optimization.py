import openmdao.api as om
import time
import json
import sys
import os
import pathlib as pl
import numpy as np
import json
# Path handling

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.convergence.integration import run_integration
import input.GeneralConstants as const
from input.data_structures import AircraftParameters


class VTOLOptimization(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #initialization
        self.label = ("_".join(time.asctime().split(" ")[1:-1])).replace(":",".")[:-3]
        self.dir_path = os.path.join("output", "run_optimizaton_" + self.label)
        os.mkdir(self.dir_path)
        self.json_path = os.path.join(self.dir_path, "design_state_" + self.label + ".json")
        self.outerloop_counter = 1
        self.init_estimate_path = r"input/initial_estimate.json"

        # Read initial estimate
        with open(self.init_estimate_path, 'r') as f:
            init = json.load(f)

        with open(self.json_path, 'w') as f:
            json.dump(init, f, indent=6)

    def setup(self):
    
        # Design variables 
        self.add_input('AR')
        self.add_input('l_fuselage')
        # self.add_input('span')
        # self.add_input('MTOM')

        # Output required
        self.add_output('energy')
        self.add_output('span')
        self.add_output("MTOM")
        self.add_output("crashworthiness_lim")

    def setup_partials(self):

        # Partial derivatives are done using finite difference
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        with open(self.json_path, 'r') as f:
            data = json.load(f)

        data["Wing"]["A"] = inputs["AR"][0]
        data["Fuselage"]["l_fuse"] = inputs["l_fuselage"][0]
        # data["b"] = inputs["span"][0]

        MTOM_one = data["AircraftParameters"]["MTOM"]

        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=6)

        print(f"===============================\nOuter loop iteration = {self.outerloop_counter}\n===============================")
        print(f"MTOM: {MTOM_one}")
        for i in range(1,11):
            print(f'\nInner loop Iteration = {i}') 
            run_integration(self.init_estimate_path  ,(self.outerloop_counter, i), self) # run integration files which can be looped

            # load data so that convergences can be checked
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            MTOM_two = data["AircraftParameters"]["MTOM"]

            #log data so that convergences can be monitored live
            print(f"MTOM: {MTOM_two} kg")

            #break out of the convergences loop if the mtom convergences below 0.5%
            epsilon = abs(MTOM_two - MTOM_one) / MTOM_one
            if epsilon < 0.005: #NOTE 
                print(f" Inner loop has converged -> epsilon is: {epsilon * 100}%")
                break
            MTOM_one = MTOM_two

            

        with open(self.json_path, 'r') as f:
            data = json.load(f)

        outputs['energy'] = data["AircraftParameters"]["mission_energy"]
        outputs['span'] = data["Wing"]["span"]
        outputs['MTOM'] = data["AircraftParameters"]["MTOM"]
        outputs['crashworthiness_lim'] = data["Fuselage"]["length_fuselage"] - data["Fuselage"]["limit_fuselage"]
        self.outerloop_counter += 1


        # Give updates on the design

        print(f"\nUpdates on Design Variables\n-----------------------------------")
        print(f"Aspect ratio = {inputs['AR'][0]}")
        print(f"lengte fuselage = {inputs['l_fuselage'][0]}")
        print(f"Crashworhiness limit = {outputs['crashworthiness_lim'][0]}")
        print(f"Mission Energy= {outputs['energy'][0]/3.6e6} [KwH]")

prob = om.Problem()
prob.model.add_subsystem('Integrated_design',VTOLOptimization())
# Initial values for the optimization TODO: Improve initial values
prob.model.set_input_defaults('Integrated_design.AR', 8.4)
prob.model.set_input_defaults('Integrated_design.l_fuselage', 9)
# prob.model.set_input_defaults('Integrated_design.span', (8.4*data["S"])**0.5 )
# prob.model.set_input_defaults('Integrated_design.span', data["mtom"] )

# Define constraints TODO: Probably better to define them in a central file, like constants
prob.model.add_constraint('Integrated_design.MTOM', upper=3175.)
prob.model.add_constraint('Integrated_design.span', lower= 6, upper= 14.)
prob.model.add_constraint('Integrated_design.crashworthiness_lim', lower= 0, upper = 14)
#prob.model.add_constraint("Integrated_design.AR", upper= 8.5)


prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'COBYLA'
#prob.driver.opt_settings['maxiter'] = 15


prob.model.add_design_var('Integrated_design.AR', lower = 5, upper = 15)
prob.model.add_design_var('Integrated_design.l_fuselage', lower = 8, upper = 16)

prob.model.add_objective('Integrated_design.energy')

prob.setup()
prob.run_driver()
