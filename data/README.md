# 数据说明

本项目的数据文件 `rfm_analysis_results.csv` 不随仓库上传，仅在此说明数据结构。

## 数据集字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| User_ID | int | 用户ID |
| Age | int | 年龄 |
| Gender | str | 性别 |
| Location | str | 地区 |
| Income | float | 收入 |
| Interests | str | 兴趣标签 |
| Last_Login_Days_Ago | int | 最近登录距今天数 |
| Purchase_Frequency | int | 购买频率 |
| Average_Order_Value | float | 平均客单价 |
| Total_Spending | float | 总消费金额 |
| Product_Category_Preference | str | 偏好品类 |
| Time_Spent_on_Site_Minutes | float | 网站停留时间(分钟) |
| Pages_Viewed | int | 浏览页数 |
| Newsletter_Subscription | int | 是否订阅邮件 (0/1) |

## 数据规模

- 行数: 1000
- 列数: 14

## 获取数据

数据为模拟生成的电商用户行为数据，可用于复现项目分析流程。
