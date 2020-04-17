import math
from math import cos, sin, ceil, floor
import numpy as np
from numpy import arcsin
from numpy.linalg import norm
from .graph_utils import plot_duo, plot_multiple, plot_xy

# from .AnimatePlane import animate_plane
# from ..utils import write_to_txt

# from .test_animation import animate_plane


class FlightModel:
    def __init__(self):
        """
        CONSTANTS
        Constants used throughout the model
        """
        self.g = 9.81  # gravity vector in m.s-2
        self.m = 73500  # mass in kg
        self.RHO = 1.225  # air density in kg.m-3
        self.S_front = 12.6  # Frontal surface in m2
        self.S_wings = 122.6  # Wings surface in m2
        self.C_x_min = 0.095  # Drag coefficient
        self.C_z_max = 0.9  # Lift coefficient
        self.THRUST_MAX = 120000 * 2  # Max thrust in Newtons
        self.DELTA_T = 0.1  # Timestep size in seconds
        self.V_R = 77  # VR for takeoff (R is for Rotate)
        self.MAX_SPEED = 250  # Max speed bearable by the plane
        self.flaps_factor = 1.5  # Lift improvement due to flaps
        self.SFC = 17.5 / 1000  # Specific Fuel Consumption in kg/(N.s)
        self.fuel_mass = 23860 / 1.25  # Fuel mass at take-off in kg
        self.M_critic = 0.78  # Critical Mach Number
        self.critical_energy = 1323000 # Maximal acceptable kinetic energy at landing in Joules
        
        '''
        VARIABLES
        '''
        self.V_R_ok = False  # Has VR been reached
        self.crashed = False

        """
        DYNAMICS
        Acceleration, speed, position, theta and misc variable that will evolve at every timestep.
        """
        self.A = [(0), (0)]  # Acceleration vector
        self.V = [(0), (0)]  # Speed Vector
        self.Pos = [(0), (0)]  # Position vector
        self.theta = 0  # Angle between the plane's axis and the ground
        self.thrust_modified = 0  # Thrust after the influence of altitude factor
        self.M = 0  # Mach number

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
        self.C_vec = [[], []]  # Store the coefficient values
        self.S_vec = [[], []]  # Store the reference surface values
        self.Fuel_vec = []
        self.thrust_vec = []
        # Store angle alpha values (angle between the speed vector and the plane)
        self.alpha_vec = []
        # Store angla gamma values (angle bweteen the ground and speed vector)
        self.gamma_vec = []
        # Store angle theta values (angle between the plane's axis and the ground)
        self.theta_vec = []

        """
        KPIS:
        Useful information to print at the end of an episode
        """
        self.max_alt = 0  # Record maximal altitude reached
        self.max_A = 0  # Record maximal acceleration
        self.min_A = 0  # Record minimal acceleration
        self.max_V = 0  # Record maximal speed
        self.min_V = 0  # Record minimal speed

        """
        ACTIONS:
        Action vec for RL stocking thrust and theta values
        """
        self.timestep = 0  # init timestep
        self.timestep_max = 1000  # Max number of timestep per episode
        # Represent the actions : couples of thrust and theta.
        self.action_vec = [
            [thrust, theta] for thrust in range(5, 11) for theta in range(0, 15)
        ]

        """
        OBSERVATIONS
        States vec for RL stocking position and velocity
        """
        self.obs = [self.Pos[0], self.Pos[1], self.V[0], self.V[1]]

    def fuel_consumption(self):
        """
        Compute the fuel mass variation at each timestep based on the thrust.
        Update the remaining fuell mass and plane mass accordingly
        """
        fuel_variation = self.SFC * self.DELTA_T * self.thrust_modified / 1000
        self.fuel_mass += -fuel_variation
        self.m += -fuel_variation
        self.Fuel_vec.append(self.fuel_mass)

    def drag(self, S, V, C):
        """
        Compute the drag using:
        S : Surface peripendicular to the drag direction
        V : Speed colinear with the drag direction
        C : Coefficient of drag/lift regarding the drag direction
        RHO : air density
        F = 1/2 * S * C * V^2
        """
        return 0.5 * self.RHO * self.altitude_factor() * S * C * np.power(V, 2)

    def Mach_Cx(self, Cx):
        """
        Compute the drag coefficient based on Mach Number and drag coefficient at M =0
        """
        if self.M < self.M_critic:
            return Cx / math.sqrt(1 - (self.M ** 2))
        else:
            return Cx * 15 * (self.M - self.M_critic) + Cx / math.sqrt(
                1 - (self.M_critic ** 2)
            )

    def Mach_Cz(self, Cz):
        """
        Compute the lift coefficient based on Mach Number and lift coefficient at M =0
        """
        M_d = self.M_critic + (1 - self.M_critic) / 4
        if self.M <= self.M_critic:
            return Cz
        elif self.M <= M_d:
            return Cz + 0.1 * (self.M - self.M_critic)
        else:
            maximal = Cz + 0.1 * (M_d - self.M_critic)
            return maximal - 0.8 * (self.M - M_d)

    def C_x(self, alpha):
        """
        Compute the drag coefficient at M = 0 depending on alpha (the higher alpha the higher the drag)
        """
        alpha = alpha + np.radians(0)
        C_x = (np.degrees(alpha) * 0.02) ** 2 + self.C_x_min
        return self.Mach_Cx(C_x)

    def C_z(self, alpha):
        """
        Compute the lift coefficient at M=0 depending on alpha (the higher the alpha, the higher the lift until stall)
        """
        alpha = alpha + np.radians(5)
        # return self.C_z_max
        sign = np.sign(np.degrees(alpha))
        if abs(np.degrees(alpha)) < 15:
            # Quadratic evolution  from C_z = 0 for 0 degrees and reaching a max value of C_z = 1.5 for 15 degrees
            C_z = sign * abs((np.degrees(alpha) / 15) * self.C_z_max)
        elif abs(np.degrees(alpha)) < 20:
            # Quadratic evolution  from C_z = 1.5 for 15 degrees to C_2 ~ 1.2 for 20 degrees.
            C_z = sign * abs((1 - ((abs(np.degrees(alpha)) - 15) / 15)) * self.C_z_max)
        else:
            ##if alpha > 20 degrees : Stall => C_z = 0
            C_z = 0
        C_z = self.Mach_Cz(C_z)
        return C_z

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
        
        if self.Pos[1] > 122:
            self.flaps_factor = 1
        else:
            self.flaps_factor = 1.7

        drag = self.drag(S_x, V, C_x) * self.flaps_factor

        # Compute lift magnitude
        lift = self.drag(S_z, V, C_z) * self.flaps_factor

        # Newton's second law
        # Z-Axis
        # Project onto Z-axis
        lift_z = cos(self.theta) * lift
        drag_z = -sin(gamma) * drag
        thrust_z = sin(self.theta) * thrust
        # Compute the sum
        F_z = lift_z + drag_z + thrust_z - P

        # X-Axis
        # Project on X-axis
        lift_x = -sin(self.theta) * lift
        drag_x = -abs(cos(gamma) * drag)
        thrust_x = cos(self.theta) * thrust
        # Compute the sum
        F_x = lift_x + drag_x + thrust_x

        # Check if we are on the ground, if so prevent from going underground by setting  vertical position and vertical speed to 0.
        if self.check_colisions() and F_z <= 0:
            F_z = 0
            energy = 0.5 * self.m * self.V[1] ** 2
            if energy > self.critical_energy:
                self.crashed = True
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
        self.M = self.V[0] / 343
        # self.V[0] = 240
        self.Pos = [self.Pos[i] + self.V[i] * self.DELTA_T for i in range(2)]
        # if self.V[0] > self.MAX_SPEED:
        #     self.V[0] = self.MAX_SPEED

        # Update plot lists
        self.A_vec[0].append(self.A[0])
        self.A_vec[1].append(self.A[1])
        self.V_vec[0].append(self.V[0])
        self.V_vec[1].append(self.V[1])
        self.Pos_vec[0].append(self.Pos[0])
        self.Pos_vec[1].append(self.Pos[1])

    def altitude_factor(self):
        """
        Compute the reducting in reactor's power with rising altitude.
        """
        alt = self.Pos[1]
        a = 1 / (math.exp(alt / 7500))
        
        return max(0, min(1, a ** (0.7)))

    def print_kpis(self):
        """
        Print interesting values : max alt, max and min acceleration, max and min speed.
        """
        print("max alt", self.max_alt)
        print("max A", self.max_A)
        print("min A", self.min_A)
        print("max V", self.max_V)
        print("min V", self.min_V)
        print("max x", max(self.Pos_vec[0]))

    def compute_episode(
        self,
        thrust,
        theta,
        thrust_cruise,
        number_timesteps,
        theta_takeoff,
        theta_cruise,
        graphs=False,
        kpis=False,
        save_figs=False,
    ):
        """
        Compute the dynamics of the the plane over a given numbero f episodes based on thrust and theta values
        Variables : Thrust is a %, theta in degrees, number of episodes (no unit)
        This is made for plotting, testing and debugging purposes and will not be used by RL agent.
        """
        # switch theta from degrees to radians and store it in the class
        self.theta = np.radians(theta)
        # Change alpha from rad to deg
        counter_cruise = 0
        counter_to = 0
        thrust_cruise_variation = thrust_cruise-thrust
        theta_cruise_variation = np.radians(theta_cruise-theta_takeoff)
        theta_takeoff_variation = theta_takeoff-self.theta
        for i in range(number_timesteps):
            # To be used for autopilot, removed while debugging
            self.fuel_consumption()
            # Apply the atitude factor to the thrust
            if self.Pos[1] > 3000:
                if counter_cruise < 600:
                    counter_cruise += 1
                    thrust += thrust_cruise_variation/600
                    self.theta += theta_cruise_variation/600
            elif self.V[0] > self.V_R:
                if counter_to < 100:
                    counter_to += 1
                    self.theta += theta_takeoff_variation/100
                self.theta = np.radians(theta_takeoff)

            if self.fuel_mass <= 0:
                thrust = 0
            self.thrust_modified = thrust * self.altitude_factor() * self.THRUST_MAX
            self.alt_factor_vec.append(self.altitude_factor())
            self.thrust_vec.append(thrust * self.THRUST_MAX)
            # Compute the dynamics for the episode
            self.compute_dyna(self.thrust_modified)
                
            if self.Pos[1] > 122:
                self.flaps_factor = 1
            else:
                self.flaps_factor = 1.7

            # if self.Pos[1] > 25:
            #     break
            # print("step:", i, " Pos ", self.Pos)

        # Plot interesting graphs after all episodes have ended.
        if graphs:
            self.plot_graphs(save_figs)
        self.max_alt = max(self.Pos_vec[1])
        self.max_A = [max(self.A_vec[0]), max(self.A_vec[1])]
        self.min_A = [min(self.A_vec[0]), min(self.A_vec[1])]
        self.max_V = [max(self.V_vec[0]), max(self.V_vec[1])]
        self.min_V = [min(self.V_vec[0]), min(self.V_vec[1])]
        if kpis:
            self.print_kpis()

    def compute_timestep(self, action):

        """
        Compute the dynamics of the the plane over a given numbero f episodes based on thrust and theta values
        Variables : Thrust in N, theta in degrees, number of episodes (no unit)
        This will be used by the RL environment.
        """
        # switch theta from degrees to radians and store it in the class
        # self.theta = np.radians(5)
        action_vec = self.action_vec[action]
        thrust_factor = action_vec[0] / 10
        self.theta = np.radians(action_vec[1])
        # print('timestep',self.timestep)
        self.timestep += 1
        self.fuel_consumption()
        # Apply the atitude factor to the thrust

        thrust_modified = thrust_factor * self.altitude_factor() * self.THRUST_MAX



        

        # Compute the dynamics for the episode
        self.compute_dyna(thrust_modified)
        self.obs = [
            ceil(self.Pos[0]),
            ceil(self.Pos[1]),
            ceil(self.V[0]),
            floor(self.V[1]),
        ]
        return self.obs

    

    def plot_graphs(self,save_figs=False,path=None):
        """
        Plot interesting graphs over timesteps :
        -Vertical drag against Vertical speed
        -Alpha, Gamma and Theta
        -Vertical forces
        -Horizontal forces
        -Acceleration (x and z)
        -Speed (x and z)
        -Position (x and z)
        -Altitude factor vs Altitude
        """

        # Cz = [self.C_z(np.radians(alpha)) for alpha in range(-5,15)]
        # alpha = [alpha for alpha in range(-5,15)]
        # Series = [alpha, Cz]
        # xlabel = "Angle of attack (°)"
        # ylabel = "Lift coefficient"
        # title = "Lift coefficient vs angle of attack"
        # plot_xy(Series, xlabel, ylabel, title, save_fig=save_figs,path=path)

        # Cx = [self.C_x(np.radians(alpha)) for alpha in range(-5,15)]
        # alpha = [alpha for alpha in range(-5,15)]
        # Series = [alpha, Cx]
        # xlabel = "Angle of attack (°)"
        # ylabel = "Drag coefficient"
        # title = "Drag coefficient vs angle of attack"
        # plot_xy(Series, xlabel, ylabel, title, save_fig=save_figs,path=path)


        Series = [self.Fuel_vec]
        labels = ["Remaining fuel (kg)"]
        xlabel = "time (s)"
        ylabel = "Remaining fuel (kg)"
        title = "Remaining fuel vs time"
        plot_multiple(Series, labels, xlabel, ylabel, title, save_fig=save_figs,path=path)

        # Series = [self.alpha_vec, self.C_vec[0]]
        # xlabel = "Angle of attack (°)"
        # ylabel = "Drag coefficient"
        # title = "Drag coefficient vs angle of attack"
        # plot_xy(Series, xlabel, ylabel, title, save_fig=save_figs)

        # Series = [self.alpha_vec, self.C_vec[1]]
        # xlabel = "Angle of attack (°)"
        # ylabel = "Lift coefficient"
        # title = "Lift coefficient vs angle of attack"
        # plot_xy(Series, xlabel, ylabel, title, save_fig=save_figs)


        # Series = [self.alpha_vec, self.S_vec[1]]
        # xlabel = "Angle of attack (°)"
        # ylabel = "Vertical reference surface (m2)"
        # title = "Vertical reference surface vs angle of attack"
        # plot_xy(Series, xlabel, ylabel, title, save_fig=save_figs,path=path)

        # Series = [self.alpha_vec, self.S_vec[0]]
        # xlabel = "Angle of attack (°)"
        # ylabel = "Horizontal reference surface (m2)"
        # title = "Horizontal reference surface vs angle of attack"
        # plot_xy(Series, xlabel, ylabel, title, save_fig=save_figs,path=path)

        # Series = [self.V_vec[1], self.drag_vec[1]]
        # xlabel = "Vertical velocity (m/s)"
        # ylabel = "Vertical drag intensity (N)"
        # title = "Vertical velocity vs Vertical drag intensity"
        # plot_xy(Series, xlabel, ylabel, title, save_fig=save_figs,path=path)

        #Angles
        Force_vec_z = [element * self.m for element in self.A_vec[1]]
        Series = [self.alpha_vec, self.gamma_vec, self.theta_vec]
        labels = ["Alpha", "Gamma", "Theta"]
        xlabel = "time (s)"
        ylabel = "Angle values (°)"
        title = "Angles vs time"
        plot_multiple(Series, labels, xlabel, ylabel, title, save_fig=save_figs,path=path)

        #Z-axis
        Force_vec_z = [element * self.m for element in self.A_vec[1]]
        Series = [self.lift_vec[1],self.P_vec, self.T_vec[1], self.drag_vec[1], Force_vec_z]
        labels = ["Lift z", "P", "Thrust z", "Drag z", "Total z"]
        xlabel = "time (s)"
        ylabel = "Force intensity (N)"
        title = "Vertical forces vs time"
        plot_multiple(Series, labels, xlabel, ylabel, title, save_fig=save_figs,path=path)

        # X-axis
        Force_vec_x = [element * self.m for element in self.A_vec[0]]
        Series = [self.T_vec[0],self.drag_vec[0],self.lift_vec[0],Force_vec_x]
        labels = ["Thrust x", "Drag x","Lift x","Total x"]
        xlabel = "time (s)"
        ylabel = "Force intensity (N)"
        title = "Horizontal forces vs time"
        plot_multiple(Series, labels, xlabel, ylabel, title, save_fig=save_figs,path=path)
        
        Series = [self.A_vec[0],self.A_vec[1]]
        labels = ["Horizontal acceleration", "Vertical acceleration"]
        xlabel = "time (s)"
        ylabel = "Acceleration (m.s-2)"
        title = "Acceleration vs time"
        plot_duo(Series, labels, xlabel, ylabel, title, save_fig=save_figs,path=path)

        Series = [self.V_vec[0],self.V_vec[1]]
        labels = ["Horizontal velocity", "Vertical velocity"]
        xlabel = "time (s)"
        ylabel = "Velociy (m.s-1)"
        title = "Velocity vs time"
        plot_duo(Series, labels, xlabel, ylabel, title, save_fig=save_figs,path=path)

        Series = [self.Pos_vec[0],self.Pos_vec[1]]
        labels = ["Horizontal position", "Vertical position"]
        xlabel = "time (s)"
        ylabel = "Distance from origin (m)"
        title = "Position vs time"
        plot_duo(Series, labels, xlabel, ylabel, title, save_fig=save_figs,path=path)

        # Series = [self.Pos_vec[1], self.alt_factor_vec]
        # xlabel = "Altitude (m)"
        # ylabel = "Altitude factor"
        # title = "Altitude factor vs altitude"
        # plot_xy(Series, xlabel, ylabel, title, save_fig=save_figs,path=path)

        Series = [self.C_vec[0],self.C_vec[1]]
        labels = ["Drag coefficient", "Lift coefficient"]
        xlabel = "time (s)"
        ylabel = "Coefficient (no unit)"
        title = "Drag and lift coefficients vs time"
        plot_multiple(Series, labels, xlabel, ylabel, title, save_fig=save_figs,path=path)

    def _animate_plane(self):
        animate_plane(self.Pos_vec, self.theta_vec)


if __name__ == "__main__":
    # Create Model

    # Run simulation over number of episodes, with thrust and theta
    def max_speed_study():
        thrust = 1  # 100% power
        theta = 0
        number_timesteps = 50000
        theta_takeoff = 10
        theta_cruise = 2.5
        range_vec = []
        for i in range(10, 11):
            thrust_cruise = 0.65
            model = FlightModel()
            model.compute_episode(
                thrust,
                theta,
                thrust_cruise,
                number_timesteps,
                theta_takeoff,
                theta_cruise,
                graphs=True,
                kpis=True,
                save_figs=True,
            )
            range_vec.append(model.Pos[0])
        print(model.V[0])
    max_speed_study()

    def TO_angle_vs_TO_dist_study():
        thrust = 1  # 100% power
        theta = 0
        number_timesteps = 1000
        dic_results = {}
        dic_results_2 = {}
        theta_takeoff = 5
        thrust_cruise = 1
        theta_cruise = 3
        TO_length = 5000
        for theta_takeoff in range(0, 16):
            dic_x = {}
            for thrust_val in range(10, 11):
                model = FlightModel()
                thrust = thrust_val / 10
                model.compute_episode(
                    thrust,
                    theta,
                    thrust_cruise,
                    number_timesteps,
                    theta_takeoff,
                    theta_cruise,
                    graphs=False,
                    kpis=True,
                    save_figs=True
                )
                dic_x[thrust] = int(max(model.Pos_vec[0]))
            if min(dic_x.values()) < TO_length:
                dic_results[theta] = dic_x
                dic_results_2[theta_takeoff] = min(dic_x.values())
        print(list(dic_results_2.values()))
        distances = list(dic_results_2.values())
        angle_values = list(dic_results_2.keys())
        angle_values = [angle-0 for angle in angle_values]
        Series = [angle_values, distances ]
        xlabel = "Take-off pitch (°)"
        ylabel = "Take-off distance (m)"
        title = "Take-off distance vs take-off pitch"
        plot_xy(Series, xlabel, ylabel, title, save_fig=True)


    #TO_angle_vs_TO_dist_study()
    # write_to_txt(environment)
    # animate_plane()

