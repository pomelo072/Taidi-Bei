from Programs import Data_Pretreatment as DP
from Programs import File_Manager as FM
from Programs import Model_Build as MB


raw_data_day, raw_data_year, raw_data_basic = FM.Input_Raw_File('X', 'Y', 'Z')

p_ZBGJ_WFLR, p_YYGJ, p_LCSY, p_JZC, p_XJJE = DP.Day_Data_Extraction(raw_data_day)

total_GB, rate_SRZZ = DP.Year_Data_Extraction(raw_data_year)

year_listed, snew_share, YZ_YJ, CZ_CD = Basic_Data_Extraction(raw_data_basic)

data_list = [p_ZBGJ_WFLR, p_YYGJ, p_LCSY, p_JZC, p_XJJE, total_GB, rate_SRZZ, year_listed, snew_share, YZ_YJ, CZ_CD]


for i in data_list:
    pass
