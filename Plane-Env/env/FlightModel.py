import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


class FlightModel:
    def __init__(self):
        # CONSTANTS
        self.g = (9.81,)
        self.m = (73500,)
        self.RHO = (1.225,)
        self.S_x = (12,)
        self.S_z = (120,)
        self.C_x = (0.09,)
        self.C_z = (0.2,)
        self.THRUST_MAX = 113000 * 2

        # DYNAMICS
        self.A = [0, 0]
        self.V = [0, 0]
        self.Pos = [0, 0]

        # VARIABLES


dynamics = {"A": [0, 0], "V": [0, 0], "Pos": [0, 0]}

variables = {"thrust": 0, "alpha": 0}

plot_variables = {
    "Lift_vec": [],
    "P_vec": [],
    "T_vec_z": [],
    "Drag_vec": [],
    "T_vec_x": [],
    "Drag_vec_x": [],
    "A_vec_x": [],
    "A_vec_z": [],
    "V_vec_x": [],
    "V_vec_z": [],
    "Pos_vec_x": [],
    "Pos_vec_z": [],
    "Force_vec_x": [],
    "Force_vec_z": [],
}


def drag(S, V, C):
    return 0.5 * CONSTANTS["rho"] * S * C * (V ** 2)


def compute_acceleration(thrust, alpha, V):
    V_x = V[0]
    V_z = V[1]
    C_x = CONSTANTS["C_x"]
    C_z = CONSTANTS["C_z"]

    # Compute P
    P = CONSTANTS["m"] * CONSTANTS["g"]
    # Compute surfaces according to alpha
    S_x = math.cos(alpha) * CONSTANTS["S_x"] + math.sin(alpha) * CONSTANTS["S_z"]
    S_z = math.sin(alpha) * CONSTANTS["S_x"] + math.cos(alpha) * CONSTANTS["S_z"]
    # Compute horizontal and vertical Drags
    drag_x = drag(S_x, V_x, C_x)
    drag_z = drag(S_z, V_z, C_z)
    # Compute lift (same formula as draft but the speed is the one perpendicular to the surface)
    lift = drag(S_x, V_x, C_z)

    # Compute sum of forces for each axis
    # Z-Axis
    F_z = abs(lift) - abs(P) + abs(thrust) * math.sin(alpha) - drag_z
    # X-Axis
    F_x = abs(thrust) * math.cos(alpha) - abs(drag_x)

    # Compute Acceleration using sum of forces = m.a
    A = [F_x / CONSTANTS["m"], F_z / CONSTANTS["m"]]

    forces = {
        "lift": lift,
        "P": P,
        "thrust_x": thrust * math.sin(alpha),
        "thrust_y": thrust * math.cos(alpha),
        "drag_x": drag_x,
        "drag_z": drag_z,
    }
    return A, forces


def compute_dynamics(dyna_vec, thrust, alpha):
    Acceleration = dyna_vec[0]
    V = dyna_vec[1]
    Pos = dyna_vec[2]
    Force, Force_vars = compute_force(Thrust, V, alpha)
    Acceleration = [f * (1.0 / m) for f in Force]
    Lift_vec.append(Force_vars[0])
    P_vec.append(Force_vars[1])
    T_vec.append(Force_vars[2])
    Drag_vec.append(Force_vars[3])
    T_vec_x.append(Force_vars[4])
    Drag_vec_x.append(Force_vars[5])
    Force_vec_x.append(Force[0])
    Force_vec_y.append(Force[1])
    A_vec_x.append(Acceleration[0])
    A_vec_y.append(Acceleration[1])
    V = [V[i] + Acceleration[i] * delta_t for i in range(2)]

    Pos = [Pos[i] + V[i] * delta_t for i in range(2)]
    if True:
        if Pos[1] <= 0:
            Pos[1] = -1
            if V[1] < 0:
                V[1] = -1
    V_vec_x.append(V[0])
    V_vec_y.append(V[1])
    Pos_vec_x.append(Pos[0])
    Pos_vec_y.append(Pos[1])
    dyna_vec = [Acceleration, V, Pos]
    Force_vec = [Lift_vec, P_vec, T_vec, Drag_vec, T_vec_x, Drag_vec_x]
    Dyna_plot_vec = (
        A_vec_x,
        A_vec_y,
        V_vec_x,
        V_vec_y,
        Pos_vec_x,
        Pos_vec_y,
        Force_vec_x,
        Force_vec_y,
    )
    return Dyna_plot_vec, dyna_vec, Force_vec


print(compute_acceleration(100, 100, dynamics["V"]))

