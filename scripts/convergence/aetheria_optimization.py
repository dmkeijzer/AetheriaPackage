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
import input.data_structures.GeneralConstants as const
from input.data_structures import PerformanceParameters

class VTOLOptimization(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label = ("_".join(time.asctime().split(" ")[1:-1])).replace(":",".")[:-3]
        self.counter = 0

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

        with open(const.json_path, 'r') as f:
            data = json.load(f)

        data["A"] = inputs["AR"][0]
        data["l_fuse"] = inputs["l_fuselage"][0]
        # data["b"] = inputs["span"][0]

        MTOM_one = data["mtom"]

        with open(const.json_path, 'w') as f:
            json.dump(data, f, indent=6)

        print(f"===============================\nOuter loop iteration = {self.counter}\n===============================")
        print(f"MTOM: {MTOM_one}")
        for i in range(10):
            print(f'\nInner loop Iteration = {i}') 
            run_integration(self.label) # run integration files which can be looped

            # load data so that convergences can be checked
            with open(const.json_path, 'r') as f:
                data = json.load(f)
            MTOM_two = data["mtom"]

            #log data so that convergences can be monitored live
            print(f"MTOM: {MTOM_two} kg")

            #break out of the convergences loop if the mtom convergences below 0.5%
            epsilon = abs(MTOM_two - MTOM_one) / MTOM_one
            if epsilon < 0.005: #NOTE 
                print(f" Inner loop has converged -> epsilon is: {epsilon * 100}%")
                break
            MTOM_one = MTOM_two

            

        with open(const.json_path, 'r') as f:
            data = json.load(f)

        outputs['energy'] = data["mission_energy"]
        outputs['span'] = data["b"]
        outputs['MTOM'] = data["mtom"]
        outputs['crashworthiness_lim'] = data["l_fuse"] - data["limit_fuse"]
        self.counter += 1


        # Give updates on the design

        print(f"\nUpdates on Design Variables\n-----------------------------------")
        print(f"Aspect ratio = {inputs['AR'][0]}")
        print(f"lengte fuselage = {inputs['l_fuselage'][0]}")
        print(f"Crashworhiness limit = {outputs['crashworthiness_lim'][0]}")
        print(f"Mission Energy= {outputs['energy'][0]/3.6e6} [KwH]")


# Set up the problem
with open(const.json_path, 'r') as f:
        data = json.load(f)

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