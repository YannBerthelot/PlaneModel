import os


def write_to_txt(environment):
    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath("env/positions.txt", cur_path)
    positions = environment.FlightModel.Pos_vec
    text_file = open(new_path, "w")
    n = text_file.write(str(positions))
    text_file.close()
    angles = environment.FlightModel.theta_vec
    text_file = open("env/angles.txt", "w")
    n = text_file.write(str(angles))
    text_file.close()
