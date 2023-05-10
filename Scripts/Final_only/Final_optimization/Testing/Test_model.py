import numpy as np
import openmdao.api as om
from Aero_tools import ISA
from Final_optimization.constants_final import g


class cruise_drag:
    def __init__(self, h, W, V, S, A, e, CD0):
        self.h = h
        self.W = W
        self.V = V
        self.S = S
        self.A = A
        self.e = e
        self.CD0 = CD0

    def d_l(self):
        # Get the density
        rho = ISA(self.h).density()

        CL  = 2*self.W/(rho*self.V*self.V*self.S)

        k = 1/(np.pi*self.A*self.e)

        return self.CD0/CL + k*CL


class test_function(om.ExplicitComponent):
    def setup(self):

        self.add_input('h')
        self.add_input('W', val = 3000*g)
        self.add_input('S', val = 14)
        self.add_input('CD0', val = 0.05)
        self.add_input('V', val = 50)
        self.add_input('A', val = 10)
        self.add_input('e', val = 0.85)

        self.add_output('D/L')

    def setup_partials(self):

        self.declare_partials('*', '*', method = 'fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        h = inputs['h']
        V = inputs['V']
        W = inputs['W']
        S = inputs['S']
        CD0 = inputs['CD0']
        A = inputs['A']
        e = inputs['e']

        drag = cruise_drag(h, W, V, S, A, e, CD0)

        outputs['D/L'] = drag.d_l()



