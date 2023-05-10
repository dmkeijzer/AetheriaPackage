import openmdao.api as om
from Test_model import test_function
import numpy as np
import matplotlib.pyplot as plt


prob = om.Problem()
prob.model.add_subsystem('cruise', test_function(), promotes_inputs=['h', 'V'])

prob.model.set_input_defaults('h', 300)
prob.model.set_input_defaults('V', 40)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'COBYLA'

prob.model.add_design_var('h', lower = 305)
prob.model.add_design_var('V', lower = 0)

prob.model.add_objective('cruise.D/L')

prob.setup()
prob.run_driver()

print(prob.get_val('cruise.h'), prob.get_val('cruise.V'))

if __name__ == "__main__":

    model = om.Group()
    model.add_subsystem('cruise', test_function(), promotes_inputs=['h', 'V'])

    prob = om.Problem(model)
    prob.setup()

    prob.set_val('cruise.h', 305)

    for V in range(40, 60):
        prob['cruise.V'] = V
        prob.run_model()
        print(prob['cruise.D/L'], V)


# class parabola(om.ExplicitComponent):
#
#     def setup(self):
#
#         self.add_input('x', val = 0.0)
#
#         self.add_output('f', val = 0.0)
#
#     def setup_partials(self):
#
#         self.declare_partials('*', '*', method = 'fd')
#
#     def compute(self, inputs, outputs):
#
#         x = inputs['x']
#
#         outputs['f'] = (x**2 + 4*x + 4)
#
#
# if __name__ == "__main__":
#
#     model = om.Group()
#     model.add_subsystem('para', parabola())
#
#     prob = om.Problem(model)
#     prob.setup()
#
#     prob.set_val('para.x', 1.0)
#
#     prob.run_model()
#     print(prob['para.f'])
#
# prob = om.Problem()
# prob.model.add_subsystem('para', parabola(), promotes_inputs=['x'])
#
# prob.model.add_subsystem('const', om.ExecComp('g = x'), promotes_inputs=['x'])
#
# prob.model.set_input_defaults('x', 3.0)
#
# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'COBYLA'
#
# prob.model.add_design_var('x', lower = -10, upper = 10)
# prob.model.add_objective('para.f')
#
# prob.model.add_constraint('const.g', lower = -3, upper = 3)
#
# prob.setup()
# prob.run_driver()
#
# print(prob.get_val('para.x'), prob.get_val('para.f'))