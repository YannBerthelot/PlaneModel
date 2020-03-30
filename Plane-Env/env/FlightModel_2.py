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
        self.C_x = 0.09
        self.C_z = 0.2
        self.THRUST_MAX = 113000 * 2
        self.DELTA_T = 1

        # DYNAMICS
        self.A = [0, 0]
        self.V = [0, 0]
        self.Pos = [0, 0]

        # LISTS_FOR_PLOT
        self.lift_vec = []
        self.P_vec = []
        self.T_z_vec = []
        self.T_x_vec = []
        self.drag_x_vec = []
        self.drag_z_vec = []
        self.A_x_vec = []
        self.A_z_vec = []
        self.V_z_vec = []
        self.V_x_vec = []
        self.Pos_x_vec = []
        self.Pos_z_vec = []

    def drag(self, S, V, C):
        return 0.5 * self.RHO * S * C * (V ** 2)

    def check_colisions(self):
        if self.Pos[1] <= 0:
            return True
        else:
            return False

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

        # X-Axis
        F_x = abs(thrust) * math.cos(alpha) - abs(drag_x)

        # Compute Acceleration using sum of forces = m.a
        A = [F_x / self.m, F_z / self.m]

        self.lift_vec.append(lift)
        self.P_vec.append(P)
        self.T_z_vec.append(thrust * math.sin(alpha))
        self.T_x_vec.append(thrust * math.cos(alpha))
        self.drag_x_vec.append(drag_x)
        self.drag_z_vec.append(drag_z)

        return A

    def compute_dyna(self, thrust, alpha):
        # Update acceleration, speed and position
        self.A = self.compute_acceleration(thrust, alpha)
        self.V = [self.V[i] + self.A[i] * self.DELTA_T for i in range(2)]
        self.Pos = [self.Pos[i] + self.V[i] * self.DELTA_T for i in range(2)]

        # Update plot lists
        self.A_x_vec.append(self.A[0])
        self.A_z_vec.append(self.A[1])
        self.V_x_vec.append(self.V[0])
        self.V_z_vec.append(self.V[1])
        self.Pos_x_vec.append(self.Pos[0])
        self.Pos_z_vec.append(self.Pos[1])

    def compute_episodes(self, thrust, alpha, num_episodes):
        for i in range(num_episodes):
            print("A", self.A)
            print("V", self.V)
            print("Pos", self.Pos)
            self.compute_dyna(thrust, alpha)
        print(self.lift_vec)
        self.plot_forces()

    def plot_forces(self):
        # Z-axis
        Force_vec_x = self.A_x_vec * self.m
        Force_vec_z = self.A_z_vec * self.m
        ax = pd.Series(self.lift_vec).plot(label="Lift")
        pd.Series(self.P_vec).plot(label="P", ax=ax)
        pd.Series(self.T_z_vec).plot(label="Thrust z", ax=ax)
        pd.Series(self.drag_z_vec).plot(label="Drag z", ax=ax)
        pd.Series(Force_vec_z).plot(label="Total z", ax=ax)
        plt.title("Z-axis")
        plt.legend()
        plt.show()
        # X-axis
        ax = pd.Series(self.T_x_vec).plot(label="Thrust x")
        pd.Series(self.drag_x_vec).plot(label="Drag x", ax=ax)
        pd.Series(Force_vec_x).plot(label="Total x", ax=ax)
        plt.title("Xaxis")
        plt.legend()
        plt.show()


Model = FlightModel()
Model.compute_episodes(1000000, 0, 10)

