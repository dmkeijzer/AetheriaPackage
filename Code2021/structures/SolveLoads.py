from Geometry import Stringer, WingBox, WingStructure
from Equilibrium import PointLoad, Moment, RunningLoad, EquilibriumEquation, DistributedMoment
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from MathFunctions import StepFunction
import rainflow
import numpy as np
import pandas as pd


class Engines:
    def __init__(self, ThrustHover, ThrustCruise, positions: list[float, int], weight):
        self.n, self.Thover, self.Tcruise = len(positions), ThrustHover, ThrustCruise
        self.pos = positions
        self.w = weight
    __repr__ = __str__ = lambda self: "Engines(" + ', '.join(f"{k}={self.__dict__[k]}" for k in self.__dict__) + ")"

class WingLoads:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.wing, self.acp = [None]*2
        self.Fx, self.Fy, self.Fz, self.Mx, self.My, self.Mz = [None]*6
        self.RFx, self.RFy, self.RFz, self.RMx, self.RMy, self.RMz = [None]*6
        self.VFx, self.VMy, self.VMz = [None]*3
        self.ViVx, self.ViMy = [None]*2
        
        self.box = WingBox(self.tsk, self.tsp, self.frac, self.toc)
        self.box.StrPlacement(self.nStrT, self.nStrB, self.strGeo, self.strType)
        self.wing = WingStructure(self.span, self.taper, self.cr, self.box)
        self.acp = (self.xac - 0.45) * self.mac # Redefine
        
    def __setitem__(self, key, item):
        self.__dict__[key] = item

    __getitem__ = lambda self, key: self.__dict__[key]
    
    mass = lambda self, material: self.span * self.wing(0).Area() * material.rho * 2

    __repr__ = __str__ = lambda self: "Wingloads(" + ', '.join(f"{k}={self.__dict__[k]}" for k in self.__dict__) + ")"

    def equilibriumCruise(self, dragDistr, liftDistr, macDistr, wingWeight):
        acp = self.acp
        Thrusts = [PointLoad([-self.engines.Tcruise, -self.engines.w, 0], [-0.45*self.wing(p).b/self.frac, 0, p]) \
                   for p in self.engines.pos] # Redefine x, y
        Drag = RunningLoad([dragDistr[1], [0]*len(dragDistr[1])], dragDistr[0], axis=2, poa=(acp, 0))
        Lift = RunningLoad([[0]*len(liftDistr[1]), liftDistr[1]], liftDistr[0], axis=2, poa=(acp, 0))
        Mac = DistributedMoment(values=[[0, 0, mom] for mom in macDistr[1]], positions=macDistr[0])
        weights = RunningLoad(values=[[0]*2, [-wingWeight / self.span]*2], positions=[0, self.span/2], axis=2)
        eqn = EquilibriumEquation(kloads=[Lift, weights, Drag, Mac] + Thrusts,
                                 ukloads=[PointLoad([1, 0, 0], [0, 0, 0]), PointLoad([0, 1, 0], [0, 0, 0]),
                                          PointLoad([0, 0, 1], [0, 0, 0]), Moment([1, 0, 0]),
                                          Moment([0, 1, 0]), Moment([0, 0, 1])])
#         print(Lift.force()*4, Drag.force(), 3024.8012022968796*9.81)
        eqn.SetupEquation()
        self.RFx, self.RFy, self.RFz, self.RMx, self.RMy, self.RMz = solved = eqn.SolveEquation()
        return solved
    
    def equilibriumVTO(self, wingWeight, ground = False):
        Thrusts = [PointLoad([(-self.engines.Thover if not ground else 0) + self.engines.w, 0, 0],
                             [-0.45*self.wing(p).b/self.frac, 0, p]) for p in self.engines.pos] # Redefine x, y
        weights = RunningLoad(values=[[wingWeight / self.span]*2, [0]*2], positions=[0, self.span/2], axis=2)
        eqtn = EquilibriumEquation(kloads=[weights] + Thrusts,
                                  ukloads=[PointLoad([1, 0, 0], [0, 0, 0]), Moment([0, 1, 0]), Moment([0, 0, 1])])
        eqtn.SetupEquation()
        self.VFx, self.VMy, self.VMz = eqtn.SolveEquation()
        return self.VFx, self.VMy, self.VMz
    
    def internalLoads(self, dragDistr, liftDistr, macDistr, wingWeight):
        if not any([self.RFx, self.RFy, self.RFz, self.RMx, self.RMy, self.RMz]):
            self.equilibriumCruise(dragDistr, liftDistr, macDistr, wingWeight)

        lin = LinearRegression()
        poly = PolynomialFeatures(degree=len(liftDistr))
        X = poly.fit_transform(dragDistr[0].reshape(-1, 1))
        lin.fit(X, np.array([dragDistr[1], liftDistr[1], macDistr[1]]).T)
        dragcoef, liftcoef, momcoef = lin.coef_
        interd, interl, interm = lin.intercept_
    
        drag = StepFunction([[dragcoef[i], 0, i] for i in range(len(dragcoef))] + [[interd, 0, 0]])
        lift = StepFunction([[liftcoef[i], 0, i] for i in range(len(liftcoef))] + [[interl, 0, 0]])
        Mac = StepFunction([[momcoef[i], 0, i] for i in range(len(momcoef))] + [[interm, 0, 0]])
        Thrust = StepFunction([[-self.engines.Tcruise, p, 0] for p in self.engines.pos])
        ThrustW = StepFunction([[-self.engines.w, p, 0] for p in self.engines.pos])
        MThrust = StepFunction([[0.45*self.wing(p).b * self.engines.w / self.frac, p, 0] for p in self.engines.pos])
    
        wgt = StepFunction([[-wingWeight / self.span, 0, 0]])
        self.Vy = -(lift + wgt).integral(self.RFy) - ThrustW
        self.Vx = -(Thrust + drag.integral(self.RFx))
        self.T = -(Mac + lift * self.acp).integral(self.RMz) - MThrust
        self.My, self.Mx = self.Vx.integral(self.RMy), self.Vy.integral(-self.RMx)
        return lift, wgt

    def internalLoadsVTO(self, wingWeight, ground = False):
        wgt = StepFunction([[wingWeight / self.span, 0, 0]]) # Fx
        Thrusts = StepFunction([[(-self.engines.Thover if not ground else 0)+self.engines.w, p, 0] for p in self.engines.pos]) # Fx
        self.ViVx = Vx = -(Thrusts + wgt.integral(self.VFx))
        self.ViMy = My = Vx.integral(self.VMy)
        return Vx, My
    
    def stressesCruise(self, point=0):
        root = self.wing(point)
        # q, tau, o
        coordinates = [[x, root.h/2] for x in np.linspace(-root.b/2, root.b/2, 1000)] + [[root.b/2, y] for y in np.linspace(root.h/2, -root.h/2, 1000)] \
        + [[x, -root.h/2] for x in np.linspace(-root.b/2, root.b/2, 1000)] + [[-root.b/2, y] for y in np.linspace(root.h/2, -root.h/2, 1000)]
        
        x, y = np.array(coordinates).T
        sigma = root.o(x, y, self.Mx(point), self.My(point))
        tau = np.array([root.tau(ix, iy, self.Vx(point), self.Vy(point), self.T(point)) for ix, iy in coordinates])
        
        return np.array(coordinates), sigma, tau, np.sqrt(3*tau**2 + sigma**2)

    def stressesVTO(self, point=0):
        root = self.wing(point)
        # q, tau, o
        coordinates = [[x, root.h/2] for x in np.linspace(-root.b/2, root.b/2, 1000)] + [[root.b/2, y] for y in np.linspace(root.h/2, -root.h/2, 1000)] \
        + [[x, -root.h/2] for x in np.linspace(-root.b/2, root.b/2, 1000)] + [[-root.b/2, y] for y in np.linspace(root.h/2, -root.h/2, 1000)]
        
        x, y = np.array(coordinates).T
        sigma = root.o(x, y, Mx = 0, My = self.ViMy(point))
        tau = np.array([root.tau(ix, iy, Vx = self.ViVx(point), Vy = 0, T = 0) for ix, iy in coordinates])
        return np.array(coordinates), sigma, tau, np.sqrt(3*tau**2 + sigma**2)
    
    def LugLoad(self):
        eqf = self.RMz / self.wing(0).b
        FmaxN = self.RFy / 2 + eqf
        eqmx = self.RMy / self.wing(0).b
        FmaxT = eqmx
        return [FmaxN / 2, FmaxT / 2]
    
    @staticmethod
    def extreme(coord, arr):
        h, l = np.argmax(arr), np.argmin(arr)
        return ([coord[l], arr[l]], [coord[h], arr[h]])

class Lug:
    def __init__(self, a, c, d, t, mat=7075):
        self.a, self.c, self.d, self.t = a, c, d, t
        self.b, self.m = (6.0, -0.020803428405001143) if mat == 7075 else (5.929411764705884, -0.011764705882352944)
        self.density = 2810
        self.mat = mat
        self.k1, self.k2 = (a * d / (c ** 2)) ** 0.5, (10 / d) ** 0.2

    Kt = lambda self: 3.8 * (self.c / self.a) ** 0.2 * (self.c / self.d) ** 0.5
    
    def stress(self, F):
        reflug = ReferenceLug(self.t, self.mat)
        Sref = reflug.stress(F)
        N = reflug.SN(Sref)
        theta = 0.25 * np.log10(N) - 0.5 if 1e3 <= N <= 1e6 else 1
        return Sref * (1 + theta * (self.k1 * self.k2 - 1))
    
    Ka = lambda self, alpha: alpha * (0.007 * self.c / self.a - 0.008) + 1
    
    mass = lambda self, L: 1e-9 * self.t * self.density * (np.pi * ((self.a + self.d / 2) ** 2 - self.d ** 2 / 4) + L * (2 * self.c + self.d))
    
    Fatigue = lambda self, F, alpha=0: self.SN(self.Kt() * self.Ka(alpha) * self.stress(F))

    SN = lambda self, S: 10 ** (self.b + S * self.m)

    __repr__ = __str__ = lambda self: "Lug(" + ', '.join(f"{k}={self.__dict__[k]}" for k in self.__dict__) + ")"


class ReferenceLug(Lug):
    def __init__(self, t, mat=7075):
        super().__init__(*([10]*3 + [t, mat]))
    
    stress = lambda self, F, alpha=0: F /  (2 * self.c * self.t)


class Fatigue:
    def __init__(self, Sground, Stakeoff, Scruise, airTime, turnOver, takeOffTime, mat):
        self.Sg, self.Sto, self.Scr, self.tAir, self.tTO = Sground, Stakeoff, Scruise, airTime, takeOffTime
        self.tot = turnOver
        self.cyc, self.df, self.ts = [None]*3
        self.mat = mat

    def determineCycle(self):
        self.ts = np.linspace(0, self.tAir + self.tot, 1000)
        gag, tg = [], self.tot / 2
        for t in self.ts:
            acoustic = 0.05 * np.sum(np.sin(2*np.pi * t / np.arange(0.001/3600, 0.01/3600, 0.001/3600))) # noise in the given frequency range
            turbulence = 0.1 * np.sum(np.sin(2*np.pi * t / np.arange(0.1/3600, 10/3600, 0.1/3600))) # same here
            taxi = 0.1 * np.sum(np.sin(2*np.pi * t / np.arange(0.05/3600, 1/3600, 0.05/3600))) # and here, all frequencies taken from the textbook - Schijve Ch 9
            if t < tg or t > self.tAir + tg:
                gag.append(self.Sg + (taxi if tg > t > 0.5 * tg or self.tAir + tg < t < self.tAir + 1.5 * tg else 0))
            elif tg + 0.5 <= t <= self.tAir + tg - 0.5:
                gag.append(self.Scr + acoustic + (turbulence if tg + 1 > t or t > self.tAir + tg - 1 else 0))
            else:
                gag.append(self.Sto + acoustic + turbulence)
        self.cyc = np.array(gag)
        return self.ts, self.cyc

    def getCycles(self):
        df = pd.DataFrame(data=list(rainflow.extract_cycles(self.cyc)),
                              columns='dS, Sm, count, ti, tf'.split(', '))
        df['Smax'] = df['Sm'] + df['dS'] / 2
        df['Smin'] = df['Sm'] - df['dS'] / 2
        df = df[df['Smax'] != 0]
        self.df = df[df['count'] != 0]
        return self.df
    
    def MinersRule(self):
        Ns = np.array([self.mat.BasquinLaw(dS) for dS in self.df.dS])
        nCycles = self.df['count'] / Ns
        return 1 / nCycles.sum()
    
    def CrackGrowth(self, a0, w, Nflights):
        length = a0
        step = 1000
        for j in range(0, Nflights, step):
            da = min(step, Nflights - j) * self.mat.ParisFatigueda(length, w,
                                self.df['Smax'].values, self.df['Smin'].values,
                                                                   self.df['count'].values).sum()
            if length >= w / 2:
                break
            length += da
        return length, j
    
