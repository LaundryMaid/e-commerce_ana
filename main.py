#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

deps = os.path.join(os.path.dirname(__file__), '.deps')
if os.path.isdir(deps):
    sys.path.insert(0, deps)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

input_path = 'data/rfm_analysis_results.csv'
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)


# 1. 数据加载与预处理
raw_cols = ['User_ID', 'Age', 'Gender', 'Location', 'Income', 'Interests',
            'Last_Login_Days_Ago', 'Purchase_Frequency', 'Average_Order_Value',
            'Total_Spending', 'Product_Category_Preference', 'Time_Spent_on_Site_Minutes',
            'Pages_Viewed', 'Newsletter_Subscription']
df = pd.read_csv(input_path, encoding='utf-8-sig', usecols=raw_cols)
df['Newsletter_Subscription'] = df['Newsletter_Subscription'].astype(bool)
print(f"数据规模: {df.shape[0]} 行 x {df.shape[1]} 列")


# 2. 特征工程
def min_max_normalize(vector, reverse=False):
    """最小-最大标准化, 返回0-100分"""
    v = np.array(vector)
    min_val, max_val = np.nanmin(v), np.nanmax(v)
    if max_val == min_val:
        return np.full_like(v, 50, dtype=float)
    if reverse:
        return ((max_val - v) / (max_val - min_val)) * 100
    return ((v - min_val) / (max_val - min_val)) * 100


# 意向分 I
df['Time_Spent_Norm'] = min_max_normalize(df['Time_Spent_on_Site_Minutes'])
df['Pages_Viewed_Norm'] = min_max_normalize(df['Pages_Viewed'])
df['I_Score'] = 0.5 * df['Time_Spent_Norm'] + 0.5 * df['Pages_Viewed_Norm']

# 阻力系数
df['Friction'] = df['Pages_Viewed'] / (df['Purchase_Frequency'] + 1)

# 忠诚度 L
def calc_loyalty(subscribed, login_days):
    if subscribed and login_days < 7:
        return 3
    elif (not subscribed) and login_days < 7:
        return 2
    return 1
df['L_Score'] = df.apply(lambda x: calc_loyalty(x['Newsletter_Subscription'], x['Last_Login_Days_Ago']), axis=1)

# 收入水平分箱
income_33 = np.percentile(df['Income'].dropna(), 33)
income_66 = np.percentile(df['Income'].dropna(), 66)
df['Income_Level'] = pd.cut(df['Income'], bins=[-np.inf, income_33, income_66, np.inf],
                            labels=['Low', 'Medium', 'High'])

# 兴趣匹配
df['Interest_Match'] = (df['Interests'] == df['Product_Category_Preference']).astype(int)


# 3. 探索性分析
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
df['Total_Spending'].hist(bins=50, ax=axes[0,0], color='#0B5CAD', alpha=0.7, edgecolor='black')
axes[0,0].axvline(df['Total_Spending'].median(), color='red', linestyle='dashed', linewidth=1)
axes[0,0].set_title('Total_Spending Distribution')
axes[0,0].set_xlabel('Total Spending')
axes[0,0].set_ylabel('Frequency')

sns.scatterplot(data=df, x='Income', y='Total_Spending', hue='Income_Level',
                palette={'Low':'#FF6B6B', 'Medium':'#4ECDC4', 'High':'#0B5CAD'},
                alpha=0.5, s=20, ax=axes[0,1])
axes[0,1].set_title('Income vs Total_Spending')
axes[0,1].legend(title='Income Level', loc='upper left')

sns.boxplot(data=df, x='Newsletter_Subscription', y='Last_Login_Days_Ago', ax=axes[1,0], color='#FFD166')
for patch in axes[1,0].patches:
    patch.set_alpha(0.3)
axes[1,0].set_title('Login Days by Subscription')

df['I_Score'].hist(bins=30, ax=axes[1,1], color='#4ECDC4', alpha=0.7, edgecolor='black')
axes[1,1].set_title('Intent Score Distribution')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_distributions.png'), dpi=150)
plt.close()

# 相关性矩阵
numeric_cols = ['Age', 'Income', 'Last_Login_Days_Ago', 'Purchase_Frequency',
                'Average_Order_Value', 'Total_Spending', 'Time_Spent_on_Site_Minutes',
                'Pages_Viewed', 'I_Score', 'Friction']
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150)
plt.close()


# 4. RFM-I 模型
df['R_Score'] = min_max_normalize(df['Last_Login_Days_Ago'], reverse=True)
df['F_Score'] = min_max_normalize(df['Purchase_Frequency'])
df['M_Score'] = min_max_normalize(df['Total_Spending'])
df['RFM_Score'] = 0.2 * df['R_Score'] + 0.3 * df['F_Score'] + 0.5 * df['M_Score']
df['I_Weight'] = df['I_Score'] / 500
df['Final_Score'] = df['RFM_Score'] * (1 + df['I_Weight'])


# 5. 用户分层
friction_q60 = df['Friction'].quantile(0.6)

def classify_user_func(r, f, m, i, income, friction, l):
    if r > 60 and f > 60 and m > 60:
        base = "重要价值用户"
    elif r > 60 and m > 60 and f < 40:
        base = "重要发展用户"
    elif r < 40 and f > 60 and m > 60:
        base = "重要保持用户"
    elif r < 40 and m > 60:
        base = "重要挽留用户"
    elif r > 60 and f > 40 and m < 40:
        base = "一般发展用户"
    elif r > 60 and f < 40 and m < 40:
        base = "一般维持用户"
    elif r < 40 and f < 40 and m < 40:
        base = "低价值用户"
    else:
        base = "一般用户"

    # RFI 修正规则
    if base in ["低价值用户", "一般维持用户"] and income == "High" and i > 60:
        return "高潜沉睡用户"
    if income == "High" and i > 70 and f < 40 and m < 50:
        return "纠结土豪"
    if base in ["一般维持用户", "一般发展用户"] and i > 80 and friction > friction_q60:
        return "犹豫型潜力用户"
    if income == "Low" and i > 70 and f < 40:
        return "隐形活跃者"
    if income == "High" and r < 40 and m > 50:
        return "高潜流失客"
    if m > 70 and f > 70 and i > 60:
        return "核心VIP"
    if income == "Low" and i < 40 and m < 40:
        return "羊毛党/低值"
    return base

df['User_Segment'] = df.apply(lambda x: classify_user_func(
    x['R_Score'], x['F_Score'], x['M_Score'], x['I_Score'],
    x['Income_Level'], x['Friction'], x['L_Score']), axis=1)

segment_stats = df.groupby('User_Segment').agg(
    用户数=('User_ID', 'count'),
    平均消费=('Total_Spending', 'mean'),
    平均购买频率=('Purchase_Frequency', 'mean'),
    平均意向分=('I_Score', 'mean'),
    平均收入=('Income', 'mean'),
    平均综合分=('Final_Score', 'mean')
).reset_index()
segment_stats['占比'] = segment_stats['用户数'] / df.shape[0] * 100
segment_stats = segment_stats.sort_values('用户数', ascending=False)
for col in ['平均消费', '平均购买频率', '平均意向分', '平均收入', '平均综合分']:
    segment_stats[col] = segment_stats[col].round(2)
print("\n分层统计:")
print(segment_stats.to_string(index=False))


# 6. 用户画像可视化
segments = df['User_Segment'].unique()
metrics = ['R_Score', 'F_Score', 'M_Score', 'I_Score']
segment_means = df.groupby('User_Segment')[metrics].mean()

n_segments = len(segments)
cmap_name = 'tab10' if n_segments <= 10 else 'gist_rainbow'
cmap = plt.colormaps[cmap_name].resampled(n_segments)
colors_border = [cmap(i) for i in range(n_segments)]
colors_fill = [(r, g, b, 0.3) for r, g, b, _ in colors_border]

n_cols = 5
n_rows = (n_segments + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), subplot_kw=dict(polar=True))
axes = axes.flatten()

for idx, seg in enumerate(segments):
    ax = axes[idx]
    values = segment_means.loc[seg, metrics].values
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values = np.append(values, values[0])
    angles += angles[:1]
    ax.plot(angles, values, color=colors_border[idx], linewidth=2, linestyle='-')
    ax.fill(angles, values, color=colors_fill[idx], alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(['25', '50', '75'])
    ax.set_title(seg, size=12, pad=20)
    ax.grid(True)

for idx in range(n_segments, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'segment_radar.png'), dpi=120)
plt.close()

# 分层分布柱状图
seg_counts = df['User_Segment'].value_counts().reset_index()
seg_counts.columns = ['User_Segment', 'count']
seg_counts['percentage'] = seg_counts['count'] / df.shape[0] * 100
seg_counts = seg_counts.sort_values('count', ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.barh(seg_counts['User_Segment'], seg_counts['count'], color='#0B5CAD', alpha=0.8)
for bar, cnt, pct in zip(bars, seg_counts['count'], seg_counts['percentage']):
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{cnt} ({pct:.1f}%)', va='center', size=9)
plt.xlabel('用户数')
plt.ylabel('用户分层')
plt.title('用户分层分布')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'segment_distribution.png'), dpi=150)
plt.close()


# 7. ROI 测算
total_budget = 10000
coupon_cost = 10
aov = df['Average_Order_Value'].mean()

core_natural = 0.25
core_with_coupon = 0.30
potential_natural = 0.01
potential_with_coupon = 0.20
rfm20_natural = 0.20
rfm20_with_coupon = 0.30

core_lift = core_with_coupon - core_natural
potential_lift = potential_with_coupon - potential_natural
rfm20_lift = rfm20_with_coupon - rfm20_natural

# 核心用户防流失: 发券降低年流失率
core_churn_no_coupon = 0.15
core_churn_with_coupon = 0.05
core_retained_lift = core_churn_no_coupon - core_churn_with_coupon

# A模型: 传统RFM
top20_threshold = df['RFM_Score'].quantile(0.80)
rfm_top20 = df[df['RFM_Score'] >= top20_threshold]
target_a_count = min(len(rfm_top20), int(total_budget // coupon_cost))
cost_a = target_a_count * coupon_cost
incremental_revenue_a = target_a_count * rfm20_lift * aov
roi_a = (incremental_revenue_a - cost_a) / cost_a * 100 if cost_a > 0 else 0

# B模型: 优化RFI
core_users = df[df['User_Segment'].isin(['核心VIP', '重要价值用户'])]
potential_users = df[df['User_Segment'].isin(['纠结土豪', '高潜沉睡用户', '犹豫型潜力用户', '高潜流失客', '隐形活跃者'])]

core_count = min(len(core_users), int(total_budget // coupon_cost))
core_annual_value = aov * core_users['Purchase_Frequency'].mean()
core_retention_revenue = core_count * core_retained_lift * core_annual_value
core_incremental_revenue = core_count * core_lift * aov
core_revenue = core_retention_revenue + core_incremental_revenue

remaining_budget = total_budget - core_count * coupon_cost
potential_count = min(len(potential_users), int(remaining_budget // coupon_cost))
potential_revenue = potential_count * potential_lift * aov

target_b_count = core_count + potential_count
cost_b = target_b_count * coupon_cost
incremental_revenue_b = core_revenue + potential_revenue
roi_b = (incremental_revenue_b - cost_b) / cost_b * 100 if cost_b > 0 else 0

print(f"\n总预算: {total_budget}元 | 券面额: {coupon_cost}元 | AOV: {aov:.0f}元")
print(f"\n【A模型 - 传统RFM】")
print(f"  目标用户: {target_a_count} | 成本: {cost_a:.0f}元 | 收益: {incremental_revenue_a:.0f}元 | ROI: {roi_a:.1f}%")
print(f"\n【B模型 - 优化RFI】")
print(f"  核心用户: {core_count}人 (防流失{core_retention_revenue:.0f} + 增量{core_incremental_revenue:.0f})")
print(f"  潜力用户: {potential_count}人 (增量{potential_revenue:.0f})")
print(f"  成本: {cost_b:.0f}元 | 收益: {incremental_revenue_b:.0f}元 | ROI: {roi_b:.1f}%")

# ROI 对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

strategies = ['A模型', 'B模型']
rois = [roi_a, roi_b]
colors = ['#FF6B6B', '#0B5CAD']
bars = axes[0].bar(strategies, rois, color=colors, alpha=0.8, edgecolor='black')
for bar, roi in zip(bars, rois):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{roi:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[0].set_title('ROI: A模型 vs B模型')
axes[0].set_ylabel('ROI (%)')
axes[0].set_ylim(0, max(rois)*1.2 + 10)

marginal_data = {
    'Strategy': ['A模型', 'A模型', 'B模型', 'B模型'],
    'User_Type': ['核心用户', '潜力用户', '核心用户', '潜力用户'],
    'Marginal_Revenue': [incremental_revenue_a, 0, core_revenue, potential_revenue]
}
df_marginal = pd.DataFrame(marginal_data)
df_marginal_pivot = df_marginal.pivot(index='Strategy', columns='User_Type', values='Marginal_Revenue').fillna(0)
df_marginal_pivot = df_marginal_pivot[['核心用户', '潜力用户']]

df_marginal_pivot.plot(kind='bar', stacked=True, ax=axes[1], color=['#0B5CAD', '#4ECDC4'], alpha=0.8)
axes[1].set_title('收益构成对比')
axes[1].set_ylabel('边际收益 (元)')
axes[1].legend(title='用户类型', loc='upper left')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roi_comparison.png'), dpi=150)
plt.close()


# 8. 保存结果
df.to_csv(os.path.join(output_dir, 'rfm_analysis_results.csv'), index=False, encoding='utf-8-sig')
print(f"\n结果已保存至: {output_dir}")