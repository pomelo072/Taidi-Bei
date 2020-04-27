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
    temp_data = pd.merge(left_data, right_data, on=['股票编号', '年份（年末）'])
    SZ_temp_data = data.loc[data['年份（年末）'] >= 6, ['股票编号', '年份（年末）', '每股送转']].fillna(0.0)
    SZ_data = SZ_temp_data.groupby('股票编号')['每股送转'].agg(np.mean).reset_index()
    fanal_data = pd.merge(temp_data, SZ_data.loc[:, ['股票编号', '每股送转']], on='股票编号')
    return fanal_data


# 基础数据提取
def Basic_Data_Extraction(data):
    group_data = data.groupby(by='股票编号')
    fanal_data = data.loc[:, ['股票编号', '上市年限']]
    # 以下错误
    fanal_data['次新股'] = group_data['所属概念板块'].apply(Is_snew_share())
    fanal_data['预增或预减'] = group_data['所属概念板块'].apply(Is_YZ_YJ())
    fanal_data['超涨或超跌'] = group_data['所属概念板块'].apply(Is_CZ_CD())
    return fanal_data
