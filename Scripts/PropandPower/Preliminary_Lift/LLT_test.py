"""
https://github.com/montagdude/weissinger
"""


import numpy as np
from math import *
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
# INPUTS
alpha   = 9         # Geometric angle of attack at root, degrees
span    = 8.57       # Wing span
root    = 0.875        # Root chord
tip     = 0.35         # Tip chord
sweep   = 0        # Sweep of quarter-chord, degrees
washout = 0        # Downward twist at tip, degrees
npoints = 10        # Number of points to evaluate on wing half

# WING
def slope(y2, y1, x2, x1): return (y2 - y1) / (x2 - x1)

class Wing:
    """ Class for swept, tapered, twisted wing """

    def __init__(self, span, root, tip, sweep, washout, tandem):
        self.span = span
        self.root = root
        self.tip = tip
        self.sweep = sweep
        self.washout = washout
        self.tandem = tandem
        self.area = None
        self.aspect_ratio = None
        self.cbar = None
        self.xroot = []
        self.yroot = None
        self.xtip = []
        self.ytip = None

        self.compute_geometry()

    def compute_geometry(self):
        """ Computes area, aspect ratio, MAC """

        self.area = 0.5*(self.root + self.tip)*self.span
        self.aspect_ratio = self.span**2./self.area
        self.compute_mac()

    def compute_mac(self):
        """ Computes mean aerodynamic chord """

        if not self.yroot: self.compute_coordinates()

        mt = slope(self.xtip[0], self.xroot[0], self.ytip, self.yroot)
        mb = slope(self.xtip[1], self.xroot[1], self.ytip, self.yroot)
        bt = self.xroot[0]
        bb = self.xroot[1]

        self.cbar = 2./self.area * (1./3.*self.ytip**3.*(mb-mt)**2. +
                                    self.ytip**2.*(mb-mt)*(bb-bt) +
                                    self.ytip*(bb-bt)**2.)

    def compute_coordinates(self):
        """ Computes root and tip x and y coordinates """

        self.yroot = 0.
        self.ytip = self.span/2.
        self.xroot = [0., self.root]
        xrootc4 = self.root/4.
        xtipc4 = xrootc4 + self.span/2.*tan(self.sweep*pi/180.)
        self.xtip = [xtipc4 - 0.25*self.tip, xtipc4 + 0.75*self.tip]

    def plot(self, ax):

        if not self.yroot: self.compute_coordinates()

        x = [self.xroot[0], self.xtip[0], self.xtip[1], self.xroot[1], \
             self.xtip[1], self.xtip[0], self.xroot[0]]
        y = [self.yroot,    self.ytip,    self.ytip,    self.yroot, \
             -self.ytip,   -self.ytip,    self.yroot]
        xrng = max(x) - min(x)
        yrng = self.span

        ax.plot(y, x, 'k')
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_xlim(-self.ytip-yrng/7., self.ytip+yrng/7.)
        ax.set_ylim(min(x)-xrng/7., max(x)+xrng/7.)
        ax.set_aspect('equal', 'datalim')
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.annotate("Area: {:.4f}\nAR: {:.4f}\nMAC: {:.4f}".format(self.area,
                                                                   self.aspect_ratio, self.cbar), xy=(0.02,0.95),
                    xycoords='axes fraction', verticalalignment='top',
                    bbox=dict(boxstyle='square', fc='w', ec='m'), color='m')

# WEISSINGER

eps = 1E-10

def l_function(lam, spc, y, n):
    """ Weissinger-L function, formulation by De Young and Harper.
        lam: sweep angle of quarter-chord (radians)
        spc: local span/chord
        y: y/l = y*
        n: eta/l = eta* """

    if abs(y-n) < eps:
        weissl = tan(lam)

    else:
        yp = abs(y)
        if n < 0.:
            weissl = sqrt((1.+spc*(yp+n)*tan(lam))**2. + spc**2.*(y-n)**2.) / \
                     (spc*(y-n) * (1.+spc*(yp+y)*tan(lam))) - 1./(spc*(y-n)) + \
                     2.*tan(lam) * sqrt((1.+spc*yp*tan(lam))**2. \
                                        + spc**2.*y**2.) / \
                     ((1.+spc*(yp-y)*tan(lam)) * (1.+spc*(yp+y)*tan(lam)))
        else:
            weissl = -1./(spc*(y-n)) + sqrt((1.+spc*(yp-n)*tan(lam))**2. + \
                                            spc**2.*(y-n)**2.) / \
                     (spc*(y-n) * (1.+spc*(yp-y)*tan(lam)))

    return weissl

def downwash_fore(c,y,Cl, x_h, z_h, V_inf):
    c = c[0:11]
    Circ = 0.5*Cl*V_inf*c
    w = []
    for i in range(len(cl)):
        r_avg = np.sqrt(x_h**2 + z_h**2)
        thetas = np.arctan2(r_avg, y- y[i])
        Circwsin = Circ*np.sin(thetas)
        f = interp1d(thetas, Circwsin)
        integral = quad(f,thetas[0],thetas[-1])
        w.append(integral[0]*(1/(np.pi*4*r_avg)))
    w_final = np.array(w)
    cang = np.arctan2(z_h,x_h)
    a_w = np.arctan2(np.cos(cang)*w_final,np.sin(cang)*w_final + V_inf) * 180/pi
    return a_w

def weissinger_l(wing, al, al_fore, m):
    """ Weissinger-L method for a swept, tapered, twisted wing.
        wing.span: span
        wing.root: chord at the root
        wing.tip: chord at the tip
        wing.sweep: quarter-chord sweep (degrees)
        wing.washout: twist of tip relative to root, +ve down (degrees)
        al: angle of attack (degrees) at the root
        m: number of points along the span (an odd number).

        Returns:
        y: vector of points along span
        cl: local 2D lift coefficient cl
        ccl: cl * local chord (proportional to sectional lift)
        al_i: local induced angle of attack
        CL: lift coefficient for entire wing
        CDi: induced drag coefficient for entire wing """

    if wing.tandem == 'fore':
        # Convert angles to radians
        lam = wing.sweep*pi/180.
        tw = -wing.washout*pi/180.
        al = al*pi/180. # Fore wing -> single AOA input

        # Initialize solution arrays
        O = m+2
        phi   = np.zeros((m))
        y     = np.zeros((m))
        c     = np.zeros((m))
        spc   = np.zeros((m))
        twist = np.zeros((m))
        theta = np.zeros((O))
        n     = np.zeros((O))
        rhs   = np.zeros((m,1))
        b     = np.zeros((m,m))
        g     = np.zeros((m,m))
        A     = np.zeros((m,m))

        # Compute phi, y, chord, span/chord, and twist on full span
        for i in range(m):
            phi[i]   = (i+1)*pi/float(m+1)                   #b[v,v] goes to infinity at phi=0
            y[i]     = cos(phi[i])                           #y* = y/l
            c[i]     = wing.root + (wing.tip-wing.root)*y[i] #local chord
            spc[i]   = wing.span/c[i]                        #span/(local chord)
            twist[i] = abs(y[i])*tw                          #local twist

        # Compute theta and n
        for i in range(O):
            theta[i] = (i+1)*pi/float(O+1)
            n[i]     = cos(theta[i])
        n0 = 1.
        phi0 = 0.
        nO1 = -1.
        phiO1 = pi

        # Construct the A matrix, which is the analog to the 2D lift slope
        # print("Calculating aerodynamics ...")
        for j in range(m):
            # print("Point " + str(j+1) + " of " + str(m))
            rhs[j,0] = al + twist[j]

            for i in range(m):
                if i == j: b[j,i] = float(m+1)/(4.*sin(phi[j]))
                else: b[j,i] = sin(phi[i]) / (cos(phi[i])-cos(phi[j]))**2. * \
                               (1. - (-1.)**float(i-j))/float(2*(m+1))

                g[j,i] = 0.
                Lj0 = l_function(lam, spc[j], y[j], n0)
                LjO1 = l_function(lam, spc[j], y[j], nO1)
                fi0 = 0.
                fiO1 = 0.
                for mu in range(m):
                    fi0 += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*phi0)
                    fiO1 += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*phiO1)

                for r in range(O):
                    Ljr = l_function(lam, spc[j], y[j], n[r])
                    fir = 0.
                    for mu in range(m):
                        fir += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*theta[r])
                    g[j,i] += Ljr*fir;
                g[j,i] = -1./float(2*(O+1)) * ((Lj0*fi0 + LjO1*fiO1)/2. + g[j,i])

                if i == j: A[j,i] = b[j,i] + wing.span/(2.*c[j])*g[j,i]
                else: A[j,i] = wing.span/(2.*c[j])*g[j,i] - b[j,i]

        # Scale the A matrix
        A *= 1./wing.span

        # Calculate ccl
        ccl = np.linalg.solve(A, rhs)

        # Add a point at the tip where the solution is known
        y = np.hstack((np.array([1.]), y))
        ccl = np.hstack((np.array([0.]), ccl[:,0]))
        c = np.hstack((np.array([wing.tip]), c))
        twist = np.hstack((np.array([tw]), twist))

        # Return only the right-hand side (symmetry)
        nrhs = int((m+1)/2)+1    # Explicit int conversion needed for Python3
        y = y[0:nrhs]
        ccl = ccl[0:nrhs]

        # Sectional cl and induced angle of attack
        cl = np.zeros(nrhs) # Lift Distribution
        al_i = np.zeros(nrhs)
        for i in range(nrhs):
            cl[i] = ccl[i]/c[i]
            al_e = cl[i]/(2.*pi)
            al_i[i] = al + twist[i] - al_e
        al_i = al_i * 180 / pi
        # Integrate to get CL and CDi
        CL = 0.
        CDi = 0.
        area = 0.
        for i in range(1,nrhs):
            dA = 0.5*(c[i]+c[i-1]) * (y[i-1]-y[i])
            dCL = 0.5*(cl[i-1]+cl[i]) * dA
            dCDi = sin(0.5*(al_i[i-1]+al_i[i])) * dCL
            CL += dCL
            CDi += dCDi
            area += dA
        CL /= area
        CDi /= area
        # print('test cl', cl, len(cl))
        # print('test al_i', al_i, len(al_i))

    elif wing.tandem == 'hind':
        # Convert angles to radians
        lam = wing.sweep*pi/180.
        tw = -wing.washout*pi/180.

        al = al *pi/180. # Fore Wing AOA
        al_fore = al_fore *pi/180. # Fore Wing induced AOA array
        al = - al_fore + al

        # Initialize solution arrays
        O = m+2
        phi   = np.zeros((m))
        y     = np.zeros((m))
        c     = np.zeros((m))
        spc   = np.zeros((m))
        twist = np.zeros((m))
        theta = np.zeros((O))
        n     = np.zeros((O))
        rhs   = np.zeros((m,1))
        b     = np.zeros((m,m))
        g     = np.zeros((m,m))
        A     = np.zeros((m,m))

        # Compute phi, y, chord, span/chord, and twist on full span
        for i in range(m):
            phi[i]   = (i+1)*pi/float(m+1)                   #b[v,v] goes to infinity at phi=0
            y[i]     = cos(phi[i])                           #y* = y/l
            c[i]     = wing.root + (wing.tip-wing.root)*y[i] #local chord
            spc[i]   = wing.span/c[i]                        #span/(local chord)
            twist[i] = abs(y[i])*tw                          #local twist

        # Compute theta and n
        for i in range(O):
            theta[i] = (i+1)*pi/float(O+1)
            n[i]     = cos(theta[i])
        n0 = 1.
        phi0 = 0.
        nO1 = -1.
        phiO1 = pi

        # Construct the A matrix, which is the analog to the 2D lift slope
        # print("Calculating aerodynamics ...")
        for j in range(0.5*m+1):
            # print('m',m)
            # print("Point " + str(j+1) + " of " + str(m))
            rhs[j,0] = al[j] + twist[j]

            for i in range(m):
                if i == j: b[j,i] = float(m+1)/(4.*sin(phi[j]))
                else: b[j,i] = sin(phi[i]) / (cos(phi[i])-cos(phi[j]))**2. * \
                               (1. - (-1.)**float(i-j))/float(2*(m+1))

                g[j,i] = 0.
                Lj0 = l_function(lam, spc[j], y[j], n0)
                LjO1 = l_function(lam, spc[j], y[j], nO1)
                fi0 = 0.
                fiO1 = 0.
                for mu in range(m):
                    fi0 += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*phi0)
                    fiO1 += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*phiO1)

                for r in range(O):
                    Ljr = l_function(lam, spc[j], y[j], n[r])
                    fir = 0.
                    for mu in range(m):
                        fir += 2./float(m+1) * (mu+1)*sin((mu+1)*phi[i])*cos((mu+1)*theta[r])
                    g[j,i] += Ljr*fir;
                g[j,i] = -1./float(2*(O+1)) * ((Lj0*fi0 + LjO1*fiO1)/2. + g[j,i])

                if i == j: A[j,i] = b[j,i] + wing.span/(2.*c[j])*g[j,i]
                else: A[j,i] = wing.span/(2.*c[j])*g[j,i] - b[j,i]

        # Scale the A matrix
        A *= 1./wing.span

        # Calculate ccl
        ccl = np.linalg.solve(A, rhs)

        # Add a point at the tip where the solution is known
        y = np.hstack((np.array([1.]), y))
        ccl = np.hstack((np.array([0.]), ccl[:,0]))
        c = np.hstack((np.array([wing.tip]), c))
        twist = np.hstack((np.array([tw]), twist))

        # Return only the right-hand side (symmetry)
        nrhs = int((m+1)/2)+1    # Explicit int conversion needed for Python3
        y = y[0:nrhs]
        ccl = ccl[0:nrhs]

        # Sectional cl and induced angle of attack
        cl = np.zeros(nrhs) # Lift Distribution
        al_i = np.zeros(nrhs)
        for i in range(nrhs):
            cl[i] = ccl[i]/c[i]
            al_e = cl[i]/(2.*pi)
            al_i[i] = al + twist[i] - al_e
        al_i = al_i * 180 / pi
        # Integrate to get CL and CDi
        CL = 0.
        CDi = 0.
        area = 0.
        for i in range(1,nrhs):
            dA = 0.5*(c[i]+c[i-1]) * (y[i-1]-y[i])
            dCL = 0.5*(cl[i-1]+cl[i]) * dA
            dCDi = sin(0.5*(al_i[i-1]+al_i[i])) * dCL
            CL += dCL
            CDi += dCDi
            area += dA
        CL /= area
        CDi /= area



    return y*wing.span/2., cl, ccl, al_i, CL, CDi, c

# RUN _WEISSINGER

def create_plot(wing, y, cl, ccl, al_i, CL, CDi):
    """ Plots lift distribution and wing geometry """

    # Mirror to left side for plotting
    npt = y.shape[0]
    y = np.hstack((y, np.flipud(-y[0:npt-1])))
    cl = np.hstack((cl, np.flipud(cl[0:npt-1])))
    ccl = np.hstack((ccl, np.flipud(ccl[0:npt-1])))
    fig, axarr = plt.subplots(2, sharex=True)

    axarr[0].plot(y, cl, 'r', y, ccl/wing.cbar, 'b')
    axarr[0].set_xlabel('y')
    axarr[0].set_ylabel('Sectional lift coefficient')
    axarr[0].legend(['Cl', 'cCl / MAC'], numpoints=1)
    axarr[0].grid()
    axarr[0].annotate("CL: {:.4f}\nCDi: {:.5f}".format(CL,CDi), xy=(0.02,0.95),
                      xycoords='axes fraction', verticalalignment='top',
                      bbox=dict(boxstyle='square', fc='w', ec='m'), color='m')

    wing.plot(axarr[1])
    plt.show()

if __name__ == "__main__":

    wing = Wing(span, root, tip, sweep, washout, 'fore')

    y, cl, ccl, al_i, CL, CDi, c = weissinger_l(wing, alpha, 0, 2*npoints-1)

    wing2 = Wing(span, root, tip, sweep, washout, 'hind')
    y2, cl2, ccl2, al_i2, CL2, CDi2, c2 = weissinger_l(wing2, alpha, downwash_fore(c, y, cl, 6, 1.25, 1), 2*npoints-1) # alpha, al_i

    # print("{:<6}".format("Area: ") + str(wing.area))
    # print("{:<6}".format("AR: ") + str(wing.aspect_ratio))
    # print("{:<6}".format("MAC: ") + str(wing.cbar))
    # print("{:<6}".format("CL: ") + str(CL))
    # print("{:<6}".format("CDi: ") + str(CDi))

    create_plot(wing, y, cl, ccl, al_i,CL, CDi)
    create_plot(wing2, y2, cl2, ccl2, al_i2, CL2, CDi2)
