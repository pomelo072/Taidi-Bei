import pandas as pd
import numpy as np


# 日数据提取
def Day_Data_Extraction(data):
    # 分组
    group_data = data.groupby(by=['股票编号', '年', '月'])
    # 求均值
    temp_data = group_data['收盘价'].agg(np.mean).reset_index()
    # 取1月1日前1个月数据
    fanal_data = temp_data.loc[temp_data['月'] == 12, :]
    fanal_data = fanal_data.loc[:, ['股票编号', '年', '收盘价']]
    fanal_data = fanal_data.rename(columns={'年': '年份（年末）', '收盘价': '交易日平均价'})
    return fanal_data


# 年数据提取
def Year_Data_Extraction(data_y, data_b):
    # 关键数据提取
    right_data = data_y.loc[:, ['股票编号', '年份（年末）', '每股现金流量净额(元/股)', '实收资本(或股本)',
                                '每股收益(期末摊薄，元/股)', '每股净资产(元/股)', '营业总收入同必增长(%)']]
    # 求每股资本公积金和每股未分配利润的和
    left_data = data_y.loc[:, ['股票编号', '年份（年末）', '每股资本公积(元/股)', '每股未分配利润(元/股)']]
    left_data['每股资本公积(元/股)+每股未分配利润(元/股)'] = left_data[['每股资本公积(元/股)', '每股未分配利润(元/股)']].sum(axis=1)
    # 合并两表
    temp_data = pd.merge(left_data, right_data, on=['股票编号', '年份（年末）'])
    # 送转比例处理
    data_rate = data_y.loc[:, ['股票编号', '每股送转']].fillna(0.0)
    data_r = []
    for i in range(data_rate.shape[0]):
        if i % 7 != 0:
            x = (data_rate.iloc[i - 1, 1] + data_rate.iloc[i, 1]) / 2
            data_r.append(x)
        else:
            x = data_rate.iloc[i, 1]
            data_r.append(x)
    dis1 = pd.DataFrame({'id': list(range(data_rate.shape[0])), 'averge': data_r})
    temp_data['近两年送转比例'] = dis1['averge']
    # 上市时间处理
    SS_data = data_b.loc[:, ['股票编号', '上市年限']]
    fanal_SS_data = pd.merge(temp_data, SS_data, on='股票编号')
    for i in range(fanal_SS_data.shape[0]):
        d = fanal_SS_data.iloc[i, 1]
        fanal_SS_data.iloc[i, 11] = fanal_SS_data.iloc[i, 11] - 7 + d
    fanal_SS_data = fanal_SS_data.rename(columns={'上市年限': '上市时间'})
    # 是否高送转数据处理
    SZ_data_y = data_y.loc[:, ['股票编号', '年份（年末）', '是否高转送']]
    SZ_data_list = SZ_data_y.values.tolist()
    for i in range(SZ_data_y.shape[0]):
        if SZ_data_list[i][1] == 7:
            SZ_data_list[i][2] = 0
            continue
        SZ_data_list[i][2] = SZ_data_list[i + 1][2]
    fanal_SZ_data = pd.DataFrame(SZ_data_list, columns=['股票编号', '年份（年末）', '是否高转送'])
    fanal_data = pd.merge(fanal_SS_data, fanal_SZ_data, on=['股票编号', '年份（年末）'])
    return fanal_data


# 基础数据提取
def Basic_Data_Extraction(data):
    # 填充缺失转换数据
    temp_data = data.loc[:, ['股票编号', '所属概念板块']].fillna('')
    temp = temp_data.values.tolist()

    # 预增或预减判断函数
    def YZYJ_func(t_data):
        YZYJ = []
        for i in t_data:
            temp_str = i[1]
            if '预增' in temp_str:
                YZYJ.append(1)
            elif '预减' in temp_str:
                YZYJ.append(-1)
            else:
                YZYJ.append(0)
        YZYJ_DF = pd.DataFrame({'股票编号': range(1, data.shape[0] + 1), '预增或预减': YZYJ})
        return YZYJ_DF

    # 超涨或超跌判断函数
    def CZCD_func(t_data):
        CZCD = []
        for i in t_data:
            temp_str = i[1]
            if '超涨' in temp_str:
                CZCD.append(1)
            elif '超跌' in temp_str:
                CZCD.append(-1)
            else:
                CZCD.append(0)
        CZCD_DF = pd.DataFrame({'股票编号': range(1, data.shape[0] + 1), '超涨或超跌': CZCD})
        return CZCD_DF

    # 次新股判断函数
    def CXG_func(t_data):
        CXG = []
        for i in t_data:
            temp_str = i[1]
            if '次新股' in temp_str:
                CXG.append(1)
            else:
                CXG.append(0)
        CXG_DF = pd.DataFrame({'股票编号': range(1, data.shape[0] + 1), '次新股': CXG})
        return CXG_DF
    # 合并数据
    fanal_data = data.loc[:, '股票编号']
    fanal_data = pd.merge(pd.merge(YZYJ_func(temp), CZCD_func(temp), on='股票编号'), CXG_func(temp), on='股票编号')
    return fanal_data


# 数据合并
def Data_Merge(data1, data2, on_key):
    M_data = pd.merge(data1, data2, on=on_key)
    return M_data


# 数据清洗
def Data_Wash(data):
    # 去除上市时间小于0时的数据
    fanal_data = data.loc[data['上市时间'] >= 0, :]
    # 去除每股资本公积+每股未分配利润小于0的数据
    fanal_data = fanal_data.loc[fanal_data['每股资本公积(元/股)+每股未分配利润(元/股)'] > 0, :]
    # 去除多余列
    fanal_data.drop(['每股资本公积(元/股)', '每股未分配利润(元/股)'], axis=1, inplace=True)
    return fanal_data


# 均值插值
def Data_Interpolate(data):
    for column in list(data.columns[data.isnull().sum() > 0]):
        mean_val = data[column].mean()
        data[column].fillna(mean_val, inplace=True)
    return data


# 数据标准差标准化
def Data_Standardization(data):
    def stand_sca(d):
        new_d = (d - d.mean()) / d.std()
        return new_d

    for column in list(data.columns):
        if column in ['股票编号', '年份（年末）', '预增或预减', '超涨或超跌', '次新股', '是否高转送']:
            continue
        else:
            data[column] = stand_sca(data[column])
    return data
