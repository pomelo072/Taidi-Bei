import pandas as pd


def Input_Raw_File(param, param1, param2):
    r_data_d = pd.read_csv(param, encoding='GBK')
    r_data_y = pd.read_csv(param1, encoding='GBK')
    r_data_b = pd.read_csv(param2, encoding='GBK')
    return r_data_d, r_data_y, r_data_b

