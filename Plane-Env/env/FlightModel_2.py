import numpy as np
import math
from math import cos, sin
import matplotlib.pyplot as plt
import pandas as pd
from numpy import arcsin
from numpy.linalg import norm


class FlightModel:
    def __init__(self):
        """
        CONSTANTS
        Constants used throughout the model
        """
        self.g = 9.81  # gravity vector
        self.m = 73500  # mass
        self.RHO = 1.225  # air density
        self.S_front = 12  # Frontal surface
        self.S_wings = 120  # Wings surface
        self.C_x_min = 0.08  # Drag coefficient
        self.C_z_max = 1.5  # Lift coefficient
        self.THRUST_MAX = 113000 * 2  # Max thrust
        self.DELTA_T = 0.1  # Timestep size
        self.V_1 = 150  # V1 for takeoff
        self.V_1_ok = False  # Has V1 been reached

        """
        DYNAMICS
        Acceleration, speed, position and theta.
        """
        self.A = [(0), (0)]  # Acceleration vector
        self.V = [(0), (0)]  # Speed Vector
        self.Pos = [(0), (0)]  # Position vector
        self.theta = 0  # Angle between the plane's axis and the ground

        """
        LISTS FOR PLOT:
        Lists initialization to store values in order to monitor them through graphs
        """
        self.lift_vec = [[], []]  # Store lift values for each axis
        self.P_vec = []  # Store P values
        self.T_vec = [[], []]  # Store thrust values for both axis
        self.drag_vec = [[], []]  # Store drag values
        self.A_vec = [[], []]  # Store Acceleration values
        self.V_vec = [[], []]  # Store Speed values
        self.Pos_vec = [[], []]  # Store position values
        self.alt_factor_vec = []  # Store altitude factor values
        self.C_vec = [[], []]
        self.S_vec = [[], []]
        self.alpha_vec = []
        # Store angle alpha values (angle between the speed vector and the plane)
        self.gamma_vec = []
        # Store angla gamma values (angle bweteen the ground and speed vector)
        self.theta_vec = []
        # Store angle theta values (angle between the plane's axis and the ground)

        """
        KPIS:
        Useful information to print at the end of an episode
        """
        self.max_alt = 0  # Record maximal altitude reached
        self.max_A = 0  # Record maximal acceleration
        self.min_A = 0  # Record minimal acceleration
        self.max_V = 0  # Record maximal speed
        self.min_V = 0  # Record minimal speed

    def drag(self, S, V, C):
        """
        Compute the drag using:
        S : Surface peripendicular to the drag direction
        V : Speed colinear with the drag direction
        C : Coefficient of drag/lift regarding the drag direction
        RHO : air density
        F = 1/2 * S * C * V^2
        """
        # print("S", S, "C", C, "V", V)
        return 0.5 * self.RHO * S * C * np.power(V, 2)

    def C_x(self, alpha):
        """
        Compute the drag coefficient depending on alpha (the higher alpha the higher the drag)
        """

        return (np.degrees(alpha) * 0.02) ** 2 + self.C_x_min

    def C_z(self, alpha):
        """
        Compute the lift coefficient depending on alpha (the higher the alpha, the higher the lift until stall)
        """
        # return self.C_z_max
        if abs(np.degrees(alpha)) < 15:
            # Quadratic evolution  from C_z = 0 for 0 degrees and reaching a max value of C_z = 1.5 for 15 degrees
            return abs((np.degrees(alpha) / 15) * self.C_z_max)
        elif abs(np.degrees(alpha)) < 20:
            # Quadratic evolution  from C_z = 1.5 for 15 degrees to C_2 ~ 1.2 for 20 degrees.
            return abs(1 - ((abs(np.degrees(alpha)) - 15) / 15) * self.C_z_max)
        else:
            ##if alpha > 20 degrees : Stall => C_z = 0
            print("stall")
            return 0

    def gamma(self):
        """
        Compute gamma (the angle between ground and the speed vector) using trigonometry. 
        sin(gamma) = V_z / V -> gamma = arcsin(V_z/V)
        """
        if norm(self.V) > 0:
            gamma = arcsin(self.V[1] / norm(self.V))
            return gamma
        else:
            return 0

    def alpha(self, gamma):
        """
        Compute alpha (the angle between the plane's axis and the speed vector).
        alpha = theta - gamma 
        """
        alpha = self.theta - gamma
        return alpha

    def S_x(self, alpha):
        """
        update the value of the surface orthogonal to the speed vector depending on alpha by projecting the x and z surface.
        S_x = cos(alpha)*S_front + sin(alpha) * S_wings
        """
        alpha = abs(alpha)
        return cos(alpha) * self.S_front + sin(alpha) * self.S_wings

    def S_z(self, alpha):
        """
        update the value of the surface colinear to the speed vector depending on alpha by projecting the x and z surface.
        S_x = sin(alpha)*S_front + cos(alpha) * S_wings
        !IMPORTANT!
        The min allows the function to be stable, I don't understand why yet.
        """
        alpha = abs(alpha)
        return (sin(alpha) * self.S_front) + (cos(alpha) * self.S_wings)

    def check_colisions(self):
        """
        Check if the plane is touching the ground
        """
        return self.Pos[1] <= 0

    def compute_acceleration(self, thrust):
        """
        Compute the acceleration for a timestep based on the thrust by using Newton's second law : F = m.a <=> a = F/m with F the resultant of all forces
        applied on the oject, m its mass and a the acceleration o fthe object.
        Variables used: 
        - P [Weight] in kg
        - V in m/s
        - gamma in rad
        - alpha in rad
        - S_x  in m^2
        - S_y in m^2
        - C_x (no units)
        - C_z (no units)
        On the vertical axis (z):
        F_z = Lift_z(alpha) * cos(theta) + Thrust * sin(theta) - Drag_z(alpha) * sin(gamma)  - P
        
        On the horizontal axis(x):
        F_x = Thrust_x  * cos(theta) - Drag_x(alpha) * cos(gamma) - Lift_x(alpha) * sin(theta)
        """
        # Compute the magnitude of the speed vector
        V = norm(self.V)
        # Compute gamma based on speed
        gamma = self.gamma()
        # Compute alpha based on gamma and theta
        alpha = self.alpha(gamma)

        # Compute P
        P = self.m * self.g

        # Compute Drag magnitude
        S_x = self.S_x(alpha)
        S_z = self.S_z(alpha)
        C_x = self.C_x(alpha)
        C_z = self.C_z(alpha)

        drag = self.drag(S_x, V, C_x)
        # print("Cx", C_x, "C_z", C_z, "S_x", S_x, "S_z", S_z, "Alpha", np.degrees(alpha))

        # Compute lift magnitude
        lift = self.drag(S_z, V, C_z)

        # Newton's second law
        # Z-Axis
        # Project onto Z-axis
        lift_z = cos(self.theta) * lift
        drag_z = -sin(gamma) * drag
        thrust_z = sin(self.theta) * thrust
        # Compute the sum
        F_z = lift_z + drag_z + thrust_z - P
        # print("F_Z", F_z, "dragz", drag_z, "thrust_z", thrust_z, "lift_z", lift_z)

        # X-Axis
        # Project on X-axis
        lift_x = -sin(self.theta) * lift
        drag_x = -abs(cos(gamma) * drag)
        thrust_x = cos(self.theta) * thrust
        # Compute the sum
        F_x = lift_x + drag_x + thrust_x

        # print("F_x", F_x, "lift_x", lift_x, "thurst_x", thrust_x)

        # Check if we are on the ground, if so prevent from going underground by setting  vertical position and vertical speed to 0.
        if self.check_colisions() and F_z <= 0:
            print("colision")
            F_z = 0
            # energy = 0.5 * self.m * V ** 2
            self.V[1] = 0
            self.Pos[1] = 0

        # Compute Acceleration using a = F/m
        A = [F_x / self.m, F_z / self.m]

        # Append all the interesting values to their respective lists for monitoring
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
        self.S_vec[0].append(S_x)
        self.S_vec[1].append(S_z)
        self.C_vec[0].append(C_x)
        self.C_vec[1].append(C_z)

        return A

    def compute_dyna(self, thrust):
        """
        Compute the dynamcis : Acceleration, Speed and Position
        Speed(t+1) = Speed(t) + Acceleration(t) * Delta_t
        Position(t+1) = Position(t) + Speed(t) * Delta_t
        """
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
        """
        Print interesting values : max alt, max and min acceleration, max and min speed.
        """
        print("max alt", self.max_alt)
        print("max A", self.max_A)
        print("min A", self.min_A)
        print("max V", self.max_V)
        print("min V", self.min_V)

    def compute_episodes(self, thrust, theta, num_episodes):
        """
        Compute the dynamics of the the plane over a given numbero f episodes based on thrust and theta values
        Variables : Thrust in N, theta in degrees, number of episodes (no unit)
        """
        # switch theta from degrees to radians and store it in the class
        self.theta = np.radians(theta)
        # Change alpha from rad to deg
        for i in range(num_episodes):
            #To be used for autopilot, removed while debugging
            if self.Pos[1] < 8000:
                thrust_factor = 1
                if self.V[0] > self.V_1:
                    self.V_1_ok = True
                if self.V_1_ok:
                    if self.theta < np.radians(10):
                        self.theta += 0.01
                else:
                    self.theta = np.radians(0)
            else:
                thrust_factor = 0.8
                if self.theta>np.radians(5):
                    self.theta += -0.01
            #Apply the atitude factor to the thrust

            thrust_modified = thrust * self.altitude_factor() *thrust_factor

            # Compute the dynamics for the episode
            self.compute_dyna(thrust_modified)

        # Plot interesting graphs after all episodes have ended.
        self.plot_graphs()
        self.max_alt = max(self.Pos_vec[1])
        self.max_A = [max(self.A_vec[0]), max(self.A_vec[1])]
        self.min_A = [min(self.A_vec[0]), min(self.A_vec[1])]
        self.max_V = [max(self.V_vec[0]), max(self.V_vec[1])]
        self.min_V = [min(self.V_vec[0]), min(self.V_vec[1])]
        self.print_kpis()

    def altitude_factor(self):
        """
        WIP
        Compute the reducting in reactor's power with rising altitude. Not ready yet
        """
        alt = self.Pos[1]
        a = 1 / (math.exp(alt / 60000))
        self.alt_factor_vec.append(a ** 0.7)
        return max(0, min(1, a ** (0.7)))

    def plot_graphs(self):
        """
        Plot interesting graphs over timesteps :
        -Vertical drag against Vertical speed
        -Alpha, Gamma and Theta
        -Vertical forces
        -Horizontal forces
        -Acceleration (x and z)
        -Speed (x and z)
        -Position (x and z)
        -(WIP) Altitude factor vs Altitude
        """

        # y = pd.Series(self.A_vec[1])
        # x = pd.Series(self.alpha_vec)
        # plt.scatter(x, y)
        # plt.title("A_z vs alpha")
        # plt.legend()
        # plt.show()

        # y = pd.Series(self.V_vec[1])
        # x = pd.Series(self.alpha_vec)
        # plt.scatter(x, y)
        # plt.title("V_z vs alpha")
        # plt.legend()
        # plt.show()

        # y = pd.Series(self.C_vec[1])
        # x = pd.Series(self.alpha_vec)
        # plt.scatter(x, y)
        # plt.title("C_z vs alpha")
        # plt.legend()
        # plt.show()

        # y = pd.Series(self.C_vec[0])
        # x = pd.Series(self.alpha_vec)
        # plt.scatter(x, y)
        # plt.title("C_x vs alpha")
        # plt.legend()
        # plt.show()

        # y = pd.Series(self.S_vec[1])
        # x = pd.Series(self.alpha_vec)
        # plt.scatter(x, y)
        # plt.title("S_z vs alpha")
        # plt.xlim((0, 120))
        # plt.legend()
        # plt.show()

        # y = pd.Series(self.S_vec[0])
        # x = pd.Series(self.alpha_vec)
        # plt.scatter(x, y)
        # plt.title("S_x vs alpha")
        # plt.xlim((0, 120))
        # plt.legend()
        # plt.show()

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

        y = pd.Series(self.alt_factor_vec)
        x = pd.Series(self.Pos_vec[1])
        plt.scatter(x, y)
        plt.title("Altitude vs Alt factor")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Create Model
    model = FlightModel()
    # Run simulation over number of episodes, with thrust and theta
    thrust = 113000 * 2  # 2 Reactors of 113kN each
    theta = 10
    number_episodes = 500000
    model.compute_episodes(thrust, theta, number_episodes)
