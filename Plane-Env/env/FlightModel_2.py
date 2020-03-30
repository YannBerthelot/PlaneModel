import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


class FlightModel:
    def __init__(self):
        # CONSTANTS
        self.g = 9.81
        self.m = 73500
        self.RHO = 1.225
        self.S_x = 12
        self.S_z = 120
        self.C_x = 0.03
        self.C_z = 0.5
        self.THRUST_MAX = 113000 * 2
        self.DELTA_T = 1

        # DYNAMICS
        self.A = [0, 0]
        self.V = [0, 0]
        self.Pos = [0, 0]

        # LISTS_FOR_PLOT
        self.lift_vec = []
        self.P_vec = []
        self.T_vec = [[], []]
        self.drag_vec = [[], []]
        self.A_vec = [[], []]
        self.V_vec = [[], []]
        self.Pos_vec = [[], []]
        self.alt_factor_vec = []

        # KPIS
        self.max_alt = 0
        self.max_A = 0
        self.min_A = 0
        self.max_V = 0
        self.min_V = 0

    def drag(self, S, V, C):
        return 0.5 * self.RHO * S * C * (V ** 2)

    def check_colisions(self):
        return self.Pos[1] <= 0

    def compute_acceleration(self, thrust, alpha):
        V_x = self.V[0]
        V_z = self.V[1]

        # Compute P
        P = self.m * self.g
        # Compute surfaces according to alpha
        S_x = math.cos(alpha) * self.S_x + math.sin(alpha) * self.S_z
        S_z = math.sin(alpha) * self.S_x + math.cos(alpha) * self.S_z
        # Compute horizontal and vertical Drags
        drag_x = self.drag(S_x, V_x, self.C_x)
        drag_z = self.drag(S_z, V_z, self.C_z)
        # Compute lift (same formula as draft but the speed is the one perpendicular to the surface)
        lift = self.drag(S_x, V_x, self.C_z)

        # Compute sum of forces for each axis
        # Z-Axis
        F_z = abs(lift) - abs(P) + abs(thrust) * math.sin(alpha) - drag_z
        if self.check_colisions() and F_z <= 0:
            F_z = 0
            energy = 0.5 * self.m * self.V[1] ** 2
            self.V[1] = 0
            self.Pos[1] = 0

        # X-Axis
        F_x = abs(thrust) * math.cos(alpha) - abs(drag_x)

        # Compute Acceleration using sum of forces = m.a
        A = [F_x / self.m, F_z / self.m]

        self.lift_vec.append(lift)
        self.P_vec.append(P)
        self.T_vec[1].append(thrust * math.sin(alpha))
        self.T_vec[0].append(thrust * math.cos(alpha))
        self.drag_vec[0].append(drag_x)
        self.drag_vec[1].append(drag_z)

        return A

    def compute_dyna(self, thrust, alpha):
        # Update acceleration, speed and position
        self.A = self.compute_acceleration(thrust, alpha)
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

    def compute_episodes(self, thrust, alpha, num_episodes):
        # Change alpha from rad to deg
        alpha = alpha / 180

        for i in range(num_episodes):
            # Apply the atitude factor to the thrust
            thrust_modified = thrust * self.altitude_factor()

            self.compute_dyna(thrust_modified, alpha)
        self.plot_graphs()
        self.max_alt = max(self.Pos_vec[1])
        self.max_A = [max(self.A_vec[0]), max(self.A_vec[1])]
        self.min_A = [min(self.A_vec[0]), min(self.A_vec[1])]
        self.max_V = [max(self.V_vec[0]), max(self.V_vec[1])]
        self.min_V = [min(self.V_vec[0]), min(self.V_vec[1])]
        self.print_kpis()

    def altitude_factor(self):
        alt = self.Pos[1]
        a = 1 / (math.exp(alt / 60000))
        self.alt_factor_vec.append(a)
        return max(0, min(1, a ** (0.7)))

    def plot_graphs(self):
        # Z-axis
        Force_vec_z = [element * self.m for element in self.A_vec[1]]

        ax = pd.Series(self.lift_vec).plot(label="Lift")
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
Model.compute_episodes(113000 * 2, 0, 750)

