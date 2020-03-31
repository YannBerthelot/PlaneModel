import numpy as np
import math
from math import cos, sin
import matplotlib.pyplot as plt
import pandas as pd
from numpy import arcsin
from numpy.linalg import norm


class FlightModel:
    def __init__(self):
        # CONSTANTS
        self.g = 9.81
        self.m = 73500
        self.RHO = 1.225
        self.S_front = 12
        self.S_wings = 120
        self.C_x_min = 0.08
        self.C_z_max = 1.5
        self.THRUST_MAX = 113000 * 2
        self.DELTA_T = 1
        self.V_1 = 150
        self.V_1_ok = False
        # DYNAMICS
        self.A = [(0), (0)]
        self.V = [(0), (0)]
        self.Pos = [(0), (0)]
        self.theta = 0

        # LISTS_FOR_PLOT
        self.lift_vec = [[], []]
        self.P_vec = []
        self.T_vec = [[], []]
        self.drag_vec = [[], []]
        self.A_vec = [[], []]
        self.V_vec = [[], []]
        self.Pos_vec = [[], []]
        self.alt_factor_vec = []
        self.alpha_vec = []
        self.gamma_vec = []
        self.theta_vec = []

        # KPIS
        self.max_alt = 0
        self.max_A = 0
        self.min_A = 0
        self.max_V = 0
        self.min_V = 0

    def drag(self, S, V, C):
        print("S", S, "C", C, "V", V)
        return 0.5 * self.RHO * S * C * np.power(V, 2)

    def C_x(self, alpha):
        return (np.degrees(alpha) * 0.02) ** 2 + self.C_x_min

    def C_z(self, alpha):
        if np.degrees(alpha) < 15:
            print("alpha", np.degrees(alpha))
            return (np.degrees(alpha) / 15) * self.C_z_max
        elif np.degrees(alpha) < 20:
            return 1.5 - ((np.degrees(alpha) - 15) / 15) * 1.5
        else:
            ##Décrochage
            print("décrochage")
            return 0

    def gamma(self):
        if norm(self.V) > 0:
            gamma = arcsin(self.V[1] / norm(self.V))
            return gamma
        else:
            return 0

    def alpha(self, gamma):
        alpha = self.theta - gamma
        return alpha

    def S_x(self, alpha):
        return cos(alpha) * self.S_front + sin(alpha) * self.S_wings

    def S_z(self, alpha):
        return sin(alpha) * self.S_front + cos(alpha) * self.S_wings

    def check_colisions(self):
        return self.Pos[1] <= 0

    def compute_acceleration(self, thrust):
        V = norm(self.V)
        gamma = self.gamma()
        alpha = self.alpha(gamma)

        # Compute P
        P = self.m * self.g
        # Compute surfaces according to alpha

        # Compute total Drag
        drag = self.drag(self.S_x(alpha), V, self.C_x(alpha))

        # Compute lift
        lift = self.drag(self.S_z(alpha), V, self.C_z(alpha))
        print("lift", lift)
        # Compute sum of forces for each axis
        # Z-Axis
        lift_z = abs(cos(self.theta) * lift)
        drag_z = -sin(gamma) * drag
        thrust_z = sin(self.theta) * thrust
        F_z = lift_z + drag_z + thrust_z - P
        print("F_Z", F_z, "dragz", drag_z, "thrust_z", thrust_z, "lift_z", lift_z)

        # X-Axis
        lift_x = -sin(self.theta) * lift
        drag_x = -abs(cos(gamma) * drag)
        thrust_x = cos(self.theta) * thrust
        F_x = lift_x + drag_x + thrust_x

        print("F_x", F_x, "lift_x", lift_x, "thurst_x", thrust_x)

        if self.check_colisions() and F_z <= 0:
            print("colision")
            F_z = 0
            # energy = 0.5 * self.m * V ** 2
            self.V[1] = 0
            self.Pos[1] = 0
        # Compute Acceleration using sum of forces = m.a
        A = [F_x / self.m, F_z / self.m]

        self.lift_vec[0].append(lift_x)
        self.lift_vec[1].append(lift_z)
        self.P_vec.append(P)
        self.T_vec[1].append(thrust_z)
        self.T_vec[0].append(thrust_x)
        self.drag_vec[0].append(drag_x)
        self.drag_vec[1].append(drag_z)
        self.alpha_vec.append(np.degrees(alpha))
        self.gamma_vec.append(np.degrees(gamma))
        self.theta_vec.append(np.degrees(self.theta))

        return A

    def compute_dyna(self, thrust):
        # Update acceleration, speed and position
        self.A = self.compute_acceleration(thrust)
        self.V = [self.V[i] + self.A[i] * self.DELTA_T for i in range(2)]
        self.Pos = [self.Pos[i] + self.V[i] * self.DELTA_T for i in range(2)]

        # Update plot lists
        self.A_vec[0].append(self.A[0])
        self.A_vec[1].append(self.A[1])
        self.V_vec[0].append(self.V[0])
        self.V_vec[1].append(self.V[1])
        self.Pos_vec[0].append(self.Pos[0])
        self.Pos_vec[1].append(self.Pos[1])

    def print_kpis(self):
        print("max alt", self.max_alt)
        print("max A", self.max_A)
        print("min A", self.min_A)
        print("max V", self.max_V)
        print("min V", self.min_V)

    def compute_episodes(self, thrust, theta, num_episodes):
        self.theta = np.radians(theta)
        # Change alpha from rad to deg
        for i in range(num_episodes):
            print("i", i)
            # if self.Pos[1] < 4000:
            #     if self.V[0] > self.V_1:
            #         self.V_1_ok = True
            #     if self.V_1_ok:
            #         self.theta = np.radians(15)
            #     else:
            #         self.theta = np.radians(0)
            # else:
            #     self.theta = np.radians(5)
            # Apply the atitude factor to the thrust
            thrust_modified = thrust
            self.compute_dyna(thrust_modified)
        self.plot_graphs()
        self.max_alt = max(self.Pos_vec[1])
        self.max_A = [max(self.A_vec[0]), max(self.A_vec[1])]
        self.min_A = [min(self.A_vec[0]), min(self.A_vec[1])]
        self.max_V = [max(self.V_vec[0]), max(self.V_vec[1])]
        self.min_V = [min(self.V_vec[0]), min(self.V_vec[1])]
        self.print_kpis()

    def altitude_factor(self):
        alt = self.Pos[1]
        a = 1 / (math.exp(alt / 100000))
        self.alt_factor_vec.append(a ** 0.7)
        a = 1
        return max(0, min(1, a ** (0.7)))

    def plot_graphs(self):
        # Z-axis
        Force_vec_z = [element * self.m for element in self.A_vec[1]]

        y = pd.Series(self.drag_vec[1])
        x = pd.Series(self.V_vec[1])
        plt.scatter(x, y)
        plt.title("Drag z vs Vz")
        plt.autoscale()
        plt.legend()
        plt.show()

        ax = pd.Series(self.alpha_vec).plot(label="Alpha")
        pd.Series(self.gamma_vec).plot(label="Gamma", ax=ax)
        pd.Series(self.theta_vec).plot(label="Theta", ax=ax)
        plt.title("Angles")
        plt.autoscale()
        plt.legend()
        plt.show()

        ax = pd.Series(self.lift_vec[1]).plot(label="Lift z")
        pd.Series(self.P_vec).plot(label="P", ax=ax)
        pd.Series(self.T_vec[1]).plot(label="Thrust z", ax=ax)
        pd.Series(self.drag_vec[1]).plot(label="Drag z", ax=ax)
        pd.Series(Force_vec_z).plot(label="Total z", ax=ax)
        plt.title("Z-axis")
        plt.autoscale()
        plt.legend()
        plt.show()
        # X-axis
        Force_vec_x = [element * self.m for element in self.A_vec[0]]
        ax = pd.Series(self.T_vec[0]).plot(label="Thrust x")
        pd.Series(self.drag_vec[0]).plot(label="Drag x", ax=ax)
        pd.Series(self.lift_vec[0]).plot(label="Lift x", ax=ax)
        pd.Series(Force_vec_x).plot(label="Total x", ax=ax)
        plt.title("Xaxis")
        plt.autoscale()
        plt.legend()
        plt.show()

        color = "tab:red"

        ax = pd.Series(self.A_vec[0]).plot(label="A_x")
        ax.set_ylabel("X")
        ax2 = ax.twinx()
        ax2.set_ylabel("Y", color=color)
        pd.Series(self.A_vec[1]).plot(ax=ax2, label="A_y", color=color)
        plt.title("Acceleration")
        plt.legend()
        plt.show()

        ax = pd.Series(self.V_vec[0]).plot(label="V_x")
        ax.set_ylabel("X")
        ax2 = ax.twinx()
        ax2.set_ylabel("Y", color=color)
        pd.Series(self.V_vec[1]).plot(ax=ax2, label="V_y", color=color)
        plt.title("Speed")
        plt.legend()
        plt.show()

        ax = pd.Series(self.Pos_vec[0]).plot(label="Pos_x")
        ax.set_ylabel("X")
        ax2 = ax.twinx()
        ax2.set_ylabel("Y", color=color)
        pd.Series(self.Pos_vec[1]).plot(ax=ax2, label="Pos_y", color=color)
        plt.title("Position")
        plt.legend()
        plt.show()

        ax = pd.Series(self.alt_factor_vec).plot(label="Altitude factor")
        ax.set_ylabel("Alt factor")
        ax2 = ax.twinx()
        ax2.set_ylabel("Alt", color=color)
        pd.Series(self.Pos_vec[1]).plot(ax=ax2, label="Altitude", color=color)
        plt.title("Altitude vs Alt factor")
        plt.legend()
        plt.show()

        ax = pd.Series(self.A_vec[1]).plot(label="Acceleration")
        ax.set_ylabel("Acceleration")
        ax2 = ax.twinx()
        ax2.set_ylabel("Alt", color=color)
        pd.Series(self.Pos_vec[1]).plot(ax=ax2, label="Altitude", color=color)
        plt.title("Acceleration vs Altitude")
        plt.legend()
        plt.show()


Model = FlightModel()
Model.compute_episodes(113000 * 2, 10, 2500)

