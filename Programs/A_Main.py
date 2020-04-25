# 数据预处理模块
from Programs import Data_Pretreatment as DP
# 文件处理模块
from Programs import File_Manager as FM
# 模型预测模块
from Programs import Model_Build as MB


if __name__ == '__main__':
    # 导入原始数据
    raw_data_day, raw_data_year, raw_data_basic = FM.Input_Raw_File('X', 'Y', 'Z')
    # 提取日数据项
    data_day = DP.Day_Data_Extraction(raw_data_day)
    # 提取年数据项
    data_year = DP.Year_Data_Extraction(raw_data_year)
    # 提出基础数据项
    data_basic = DP.Basic_Data_Extraction(raw_data_basic)
    # 数据合并
    data = DP.Data_Merge(DP.DataMerge(data_day, data_year, '年'), data_basic, '股票编号')
    # 数据清洗
    data = DP.Data_Wash(data)
    # 数据标准化
    data = DP.Data_Standardization(data)
