import openmdao.api as om
import time
import json
import json
from AetheriaPackage.integration import run_integration, multi_run
import os



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

        # Output required
        self.add_output('energy')
        self.add_output('span')
        self.add_output("MTOM")

    def setup_partials(self):

        # Partial derivatives are done using finite difference
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        #------------- Loading changed values from optimizers ------------------
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        data["Wing"]["aspect_ratio"] = inputs["AR"][0]

        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=6)
        #---------------- Computing new values -----------------------------

        multi_run(self.init_estimate_path, self.outerloop_counter, self.json_path, self.dir_path)

        #----------------- Giving new values to the optimizer -------------------------
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        outputs['energy'] = data["AircraftParameters"]["mission_energy"]
        outputs['span'] = data["Wing"]["span"]
        outputs['MTOM'] = data["AircraftParameters"]["MTOM"]
        self.outerloop_counter += 1


        #---------------------- Give updates on the design -----------------------------

        print(f"\nUpdates on Design Variables\n-----------------------------------")
        print(f"Aspect ratio = {inputs['AR'][0]}")
        print(f"Mission Energy= {outputs['energy'][0]/3.6e6} [KwH]")





prob = om.Problem()
prob.model.add_subsystem('Integrated_design',VTOLOptimization())
# Initial values for the optimization TODO: Improve initial values
prob.model.set_input_defaults('Integrated_design.AR', 8.4)

# Define constraints TODO: Probably better to define them in a central file, like constants
prob.model.add_constraint('Integrated_design.MTOM', upper=3175.)
prob.model.add_constraint('Integrated_design.span', lower= 6, upper= 14.)
#prob.model.add_constraint("Integrated_design.AR", upper= 8.5)


prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'COBYLA'
#prob.driver.opt_settings['maxiter'] = 15


prob.model.add_design_var('Integrated_design.AR', lower = 5, upper = 15)

prob.model.add_objective('Integrated_design.energy')

prob.setup()
prob.run_driver()
