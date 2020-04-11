import arcade
import os
import math


def read_txt():
    cur_path = os.path.dirname(__file__)

    new_path = os.path.join(cur_path, "positions.txt")
    raw_pos = open(new_path, "r").read()
    Pos_vec = eval(raw_pos)

    new_path = os.path.join(cur_path, "angles.txt")
    raw_ang = open(new_path, "r").read()
    theta_vec = eval(raw_ang)

    return Pos_vec, theta_vec


def animate_plane():
    Pos_vec, theta_vec = read_txt()
    y_vec = Pos_vec[1]
    x_vec = Pos_vec[0]

    SPRITE_SCALING = 0.1

    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 1000
    SCREEN_TITLE = "Move Sprite by Angle Example"

    MOVEMENT_SPEED = 5
    ANGLE_SPEED = 5

    class Player(arcade.Sprite):
        """ Player class """

        def __init__(self, image, scale):
            """ Set up the player """

            # Call the parent init
            super().__init__(image, scale)
            self.i = 0

            # Create a variable to hold our speed. 'angle' is created by the parent
            self.speed = 0

        def update(self):
            # Convert angle in degrees to radians.
            angle_rad = math.radians(self.angle)

            # Rotate the ship
            self.angle = theta_vec[self.i]
            # print("i", self.i, "x_vec", x_vec)

            x_scale_factor = 0.1
            y_scale_factor = 1

            # Use math to find our change based on our speed and angle
            self.center_x = 100 + x_vec[self.i] * x_scale_factor

            self.center_y = y_vec[self.i] * y_scale_factor + 300
            delta = 1
            if self.i + delta < len(x_vec):
                self.i += delta

    class MyGame(arcade.Window):
        """
        Main application class.
        """

        def __init__(self, width, height, title):
            """
            Initializer
            """

            # Call the parent class initializer
            super().__init__(width, height, title)

            # Set the working directory (where we expect to find files) to the same
            # directory this .py file is in. You can leave this out of your own
            # code, but it is needed to easily run the examples using "python -m"
            # as mentioned at the top of this program.
            file_path = os.path.dirname(os.path.abspath(__file__))
            os.chdir(file_path)

            # Variables that will hold sprite lists
            self.player_list = None

            # Set up the player info
            self.player_sprite = None

            # Set the background color
            arcade.set_background_color(arcade.color.WHITE)

        def setup(self):
            """ Set up the game and initialize the variables. """

            # Sprite lists
            self.player_list = arcade.SpriteList()

            # Set up the plane
            self.player_sprite = Player("plane.png", SPRITE_SCALING)
            self.player_sprite.center_x = SCREEN_WIDTH / 2
            self.player_sprite.center_y = SCREEN_HEIGHT / 2
            self.player_list.append(self.player_sprite)

        def on_draw(self):
            """
            Render the screen.
            """

            # This command has to happen before we start drawing
            arcade.start_render()

            # Draw all the sprites.
            self.player_list.draw()
            arcade.draw_line(
                0, 300, SCREEN_WIDTH, 300, arcade.color.BLACK, 2,
            )

        def on_update(self, delta_time):
            """ Movement and game logic """
            # Call update on all sprites
            self.player_list.update()

    window = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    animate_plane()
