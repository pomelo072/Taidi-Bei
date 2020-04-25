import pandas as pd


def Input_Raw_File(param, param1, param2):
    r_data_d = pd.read_csv(param)
    r_data_y = pd.read_csv(param1)
    r_data_b = pd.read_csv(param2)
    return r_data_d, r_data_y, r_data_b

