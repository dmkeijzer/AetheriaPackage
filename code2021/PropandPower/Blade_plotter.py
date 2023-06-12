import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as sp_int


class PlotBlade:
    def __init__(self, chords, pitchs, radial_coords, R, xi_0, airfoil_name='naca4412', tc_ratio=0.12):
        """
        :param chords: Array with chords, from root to tip [m]
        :param pitchs: Array with pitch angles, from root to tip [rad]
        :param radial_coords: Radial coordinates per station [m]
        :param R: Radius of propeller [m]
        :param xi_0: Hub ratio [-]
        :param airfoil_name: String with ile name of the airfoil to use, by default NACA4412 [-]
        :param tc_ratio: Thickness to chord ratio of the airfoil, by default 12% for NACA4412 [-]
        """
        self.chords = chords
        self.pitchs = pitchs
        self.radial_coords = radial_coords
        self.R = R
        self.xi_0 = xi_0
        self.airfoil_name = airfoil_name
        self.tc_ratio = tc_ratio

    def load_airfoil(self):
        file = open('code2021/PropandPower/'+self.airfoil_name)

        airfoil = file.readlines()

        # Close file
        file.close()

        # List to save formatted coordinates
        airfoil_coord = []

        for line in airfoil:
            # Separate variables inside file
            a = line.split()

            new_line = []
            for value in a:
                new_line.append(float(value))

            # Set c/4 to be the origin
            new_line[0] -= 0.25
            airfoil_coord.append(new_line)

        airfoil_coord = np.array(airfoil_coord)
        airfoil_coord = airfoil_coord.T

        return airfoil_coord

    def plot_blade(self):
        # Create figures
        fig, axs = plt.subplots(2, 1)
        axs[0].axis('equal')

        # Plot side view of the airfoil cross-sections
        for i in range(len(self.chords)):
            # Scale the chord length and thickness
            x_coords = self.load_airfoil()[0] * self.chords[i]
            y_coords = self.load_airfoil()[1] * self.chords[i]

            # New coordinates after pitch
            x_coords_n = []
            y_coords_n = []

            # Apply pitch
            for j in range(len(x_coords)):
                # Transform coordinates with angle
                x_coord_n = np.cos(self.pitchs[i]) * x_coords[j] + np.sin(self.pitchs[i]) * y_coords[j]
                y_coord_n = -np.sin(self.pitchs[i]) * x_coords[j] + np.cos(self.pitchs[i]) * y_coords[j]

                # Save new coordinates
                x_coords_n.append(x_coord_n)
                y_coords_n.append(y_coord_n)

            # Plot the cross section

            axs[0].plot(x_coords_n, y_coords_n)
        axs[0].hlines(0, -0.2, 0.3, label='Disk plane', colors='k', linewidths=0.75)

        y_mins = []
        y_maxs = []
        for i in range(len(self.chords)):
            chord_len = self.chords[i]
            # Plot chord at its location, align half chords
            y_maxs.append(chord_len/4)
            y_mins.append(-3*chord_len/4)

        # # Interpolate for smooth distribution
        # y_max_fun = sp_int.CubicSpline(self.radial_coords, y_maxs, extrapolate=True)
        # y_min_fun = sp_int.CubicSpline(self.radial_coords, y_mins, extrapolate=True)

        # Polinomial regression for smooth distribution
        coef_y_max_fun = np.polynomial.polynomial.polyfit(self.radial_coords, y_maxs, 5)
        coef_y_min_fun = np.polynomial.polynomial.polyfit(self.radial_coords, y_mins, 5)

        y_max_fun = np.polynomial.polynomial.Polynomial(coef_y_max_fun)
        y_min_fun = np.polynomial.polynomial.Polynomial(coef_y_min_fun)

        # Plot
        axs[1].axis('equal')

        # Plot actual points
        axs[1].scatter(self.radial_coords, y_maxs)
        axs[1].scatter(self.radial_coords, y_mins)

        # Plot smooth distribution  TODO: revise
        radius = np.linspace(self.xi_0*self.R, self.R, 200)
        axs[1].plot(radius, y_min_fun(radius))
        axs[1].plot(radius, y_max_fun(radius))

        axs[0].legend()
        plt.show()

    def plot_3D_blade(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.set_aspect('equal')

        # Plot airfoil blade in 3D
        for i in range(len(self.chords)):
            # Scale the chord length and thickness
            x_coords = self.load_airfoil()[0] * self.chords[i]
            y_coords = self.load_airfoil()[1] * self.chords[i]

            # New coordinates after pitch
            x_coords_n = []
            y_coords_n = []

            blade_plot = np.empty(3)

            # Apply pitch
            for j in range(len(x_coords)):
                # Transform coordinates with angle
                x_coord_n = np.cos(self.pitchs[i]) * x_coords[j] + np.sin(self.pitchs[i]) * y_coords[j]
                y_coord_n = -np.sin(self.pitchs[i]) * x_coords[j] + np.cos(self.pitchs[i]) * y_coords[j]

                # Save new coordinates
                x_coords_n.append(x_coord_n)
                y_coords_n.append(y_coord_n)

                # Save coordinates of each point
                point = [x_coord_n, y_coord_n, self.radial_coords[i]]
                blade_plot = np.vstack((blade_plot, point))

            ax.plot3D(x_coords_n, y_coords_n, self.radial_coords[i], color='k')

        # ax.plot3D(blade_plot[:][0], blade_plot[:][1], blade_plot[:][2], color='k')


        # Trick to set 3D axes to equal scale, obtained from:
        # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

        # Just to get max X, Y, and Z
        X = np.array([self.chords[0], self.chords[-1]])
        Y = np.array([self.chords[0]*self.tc_ratio, self.chords[-1]*self.tc_ratio])
        Z = np.array([0, self.radial_coords[-1]])

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        plt.show()
