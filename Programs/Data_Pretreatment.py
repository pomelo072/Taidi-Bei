import pandas as pd
import numpy as np


# data_list = [p_ZBGJ_WFLR, p_YYGJ, p_TTM, p_JZC, p_XJJE, total_GB, rate_SRZZ, year_listed, snew_share, YZ_YJ, CZ_CD]
def Day_Data_Extraction(data):
    group_data = data.groupby(by=['股票编号', '年', '月'])[['每股盈余公积金', '每股收益TTM值', '每股净资产',
                                                      '每股现金流量净额']].agg(lambda x: x.mode()).reset_index()
    return group_data


def Year_Data_Extraction(data):
    group_data = data.groupby(by=['股票编号', '年份（年份）'])[['实收资本(或股本)', '营业总收入同必增长(%)']].agg()
    return None

def Is_snew_share(x):
    if '次新股' in x:
        return 1
    else:
        return 0


def Is_YZ_YJ(x):
    if '预增' in x:
        return 1
    elif '预减' in x:
        return -1
    else:
        return 0


def Is_CZ_CD(x):
    if '超涨' in x:
        return 1
    elif '超跌' in x:
        return -1
    else:
        return 0


