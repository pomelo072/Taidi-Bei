# Taidi-Bei

## 因子池

| 因子类型 | 名称·                         | 说明                     |变量名        |
| -------- | -----------------------------| ------------------------ |------------ |
| 因变量    | 是否高送转                    | 送转比例0-1，当>0.5为高送转|   Is_GSZ    |
| 自变量    | 每股资本公积金+每股未分配利润   |                          | p_ZBGJ_WFLR |
|          | 每股盈余公积金                 |                          | p_YYGJ      |
|          | 每股留存收益                   |                          | p_LCSY      |
|          | 每股净资产                     |                          | p_JZC       |
|          | 每股现金流净额                 |                          | p_XJJE       |
|....|....|....|....|
|          | 实收资本(或股本)              |                          |   total_GB    |
|          | 营业总收入同必增长(%)          |                          |   rate_SRZZ   |
|....|....|....|....|
|          | 交易日平均价                  |        日数据计算                  |   average_price    |
|          | 近两年送转比率                | 最近两年送转比率平均值（年数据计算）  |   transform   |
|....|....|....|....|
|          | **上市天数**                   | 上市时间（年）            |    year_listed     |
|          | 次新股                         | 是为1，否为0             |     snew_share      |
|          | 预增或预减                      | 预增为1，预减-1，其他为0 |     YZ_YJ            |
|          | 超涨或超跌                      | 超涨为1，超跌为-1，其他为0  |  CZ_CD            |
