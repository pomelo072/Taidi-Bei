# 数据预处理模块
from Programs import Data_Pretreatment as DP
# 文件处理模块
from Programs import File_Manager as FM
# 模型预测模块
from Programs import Model_Build as MB

if __name__ == '__main__':
    # 导入原始数据
    raw_data_day, raw_data_year, raw_data_basic = FM.Input_Raw_File(u'./Data/Day.csv',
                                                                    u'./Data/Year.csv',
                                                                    u'./Data/Basic.csv')
    # 提取日数据项
    # data_day = DP.Day_Data_Extraction(raw_data_day)
    # data_day.to_csv('./data_day.csv', encoding='GBK')
    # 提取年数据项
    # data_year = DP.Year_Data_Extraction(raw_data_year)
    # data_year.to_csv('./data_year.csv', encoding='GBK')
    # # 提出基础数据项
    data_basic = DP.Basic_Data_Extraction(raw_data_basic)
    data_basic.to_csv('./data_basic.csv', encoding='GBK')
    # # 数据合并
    # data = DP.Data_Merge(DP.DataMerge(data_day, data_year, '年'), data_basic, '股票编号')
    # # 数据清洗
    # data = DP.Data_Wash(data)
    # # 数据标准化
    # data = DP.Data_Standardization(data)
