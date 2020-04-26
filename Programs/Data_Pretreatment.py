import pandas as pd
import numpy as np


# data_list = [p_ZBGJ_WFLR, p_YYGJ, p_TTM, p_JZC, p_XJJE, total_GB, rate_SRZZ, year_listed, snew_share, YZ_YJ, CZ_CD]
# 日数据提取
def Day_Data_Extraction(data):
    # 分组
    group_data = data.groupby(by=['股票编号', '年', '月'])
    # 求均值
    temp_data = group_data['收盘价'].agg(np.mean).reset_index()
    # 取1月1日前1个月数据
    fanal_data = temp_data.loc[temp_data['月'] == 12, :]
    return fanal_data


def Year_Data_Extraction(data):
    right_data = data.loc[:, ['股票编号', '年份（年末）', '每股现金流量净额(元/股)', '实收资本(或股本)',
                              '每股收益(期末摊薄，元/股)', '每股净资产(元/股)', '营业总收入同必增长(%)']]
    # 求ZBGJ和WFLR的和
    left_data = data.loc[:, ['股票编号', '年份（年末）', '每股资本公积(元/股)', '每股未分配利润(元/股)']]
    left_data['每股资本公积(元/股)+每股未分配利润(元/股)'] = left_data[['每股资本公积(元/股)', '每股未分配利润(元/股)']].sum(axis=1)
    # 合并两表
    fanal_data = pd.merge(left_data, right_data, on=['股票编号', '年份（年末）'])
    return fanal_data


# 次新股判断
def Is_snew_share(x):
    if '次新股' in x:
        return 1
    else:
        return 0


# 预增减判断
def Is_YZ_YJ(x):
    if '预增' in x:
        return 1
    elif '预减' in x:
        return -1
    else:
        return 0


# 超涨跌判断
def Is_CZ_CD(x):
    if '超涨' in x:
        return 1
    elif '超跌' in x:
        return -1
    else:
        return 0


# 基础数据提取
def Basic_Data_Extraction(data):
    group_data = data.groupby(by='股票编号')
    fanal_data = data.loc[:, ['股票编号', '上市年限']]
    # 以下错误
    fanal_data['次新股'] = group_data['所属概念板块'].agg(Is_snew_share)
    fanal_data['预增或预减'] = group_data['所属概念板块'].agg(Is_YZ_YJ)
    fanal_data['超涨或超跌'] = group_data['所属概念板块'].agg(Is_CZ_CD)
    return fanal_data
