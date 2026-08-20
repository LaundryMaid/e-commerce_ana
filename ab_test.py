#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B 测试 — 三模型对比 (A模型 / B模型 / C模型)

阶段1: 原始虚高版B (年消费防流失价值)
阶段2: 修正后无惩罚
阶段3: 修正后有惩罚
"""

import os
import sys

deps = os.path.join(os.path.dirname(__file__), '.deps')
if os.path.isdir(deps):
    sys.path.insert(0, deps)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

input_path = 'data/rfm_analysis_results.csv'
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)


def min_max_normalize(vector, reverse=False):
    v = np.array(vector, dtype=float)
    min_val, max_val = np.nanmin(v), np.nanmax(v)
    if max_val == min_val:
        return np.full_like(v, 50, dtype=float)
    if reverse:
        return ((max_val - v) / (max_val - min_val)) * 100
    return ((v - min_val) / (max_val - min_val)) * 100


raw_cols = ['User_ID', 'Age', 'Gender', 'Location', 'Income', 'Interests',
            'Last_Login_Days_Ago', 'Purchase_Frequency', 'Average_Order_Value',
            'Total_Spending', 'Product_Category_Preference', 'Time_Spent_on_Site_Minutes',
            'Pages_Viewed', 'Newsletter_Subscription']
df = pd.read_csv(input_path, encoding='utf-8-sig', usecols=raw_cols)
df['Newsletter_Subscription'] = df['Newsletter_Subscription'].astype(bool)

df['Time_Spent_Norm'] = min_max_normalize(df['Time_Spent_on_Site_Minutes'])
df['Pages_Viewed_Norm'] = min_max_normalize(df['Pages_Viewed'])
df['I_Score'] = 0.5 * df['Time_Spent_Norm'] + 0.5 * df['Pages_Viewed_Norm']
df['Friction'] = df['Pages_Viewed'] / (df['Purchase_Frequency'] + 1)

def calc_loyalty(subscribed, login_days):
    if subscribed and login_days < 7:
        return 3
    elif (not subscribed) and login_days < 7:
        return 2
    return 1
df['L_Score'] = df.apply(lambda x: calc_loyalty(x['Newsletter_Subscription'], x['Last_Login_Days_Ago']), axis=1)

income_33 = np.percentile(df['Income'].dropna(), 33)
income_66 = np.percentile(df['Income'].dropna(), 66)
df['Income_Level'] = pd.cut(df['Income'], bins=[-np.inf, income_33, income_66, np.inf],
                            labels=['Low', 'Medium', 'High'])

df['R_Score'] = min_max_normalize(df['Last_Login_Days_Ago'], reverse=True)
df['F_Score'] = min_max_normalize(df['Purchase_Frequency'])
df['M_Score'] = min_max_normalize(df['Total_Spending'])
df['RFM_Score'] = 0.2 * df['R_Score'] + 0.3 * df['F_Score'] + 0.5 * df['M_Score']

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


# 参数
total_budget = 1000
coupon_cost = 10
aov = df['Average_Order_Value'].mean()

core_lift = 0.05
potential_lift = 0.19
rfm20_lift = 0.10

core_users = df[df['User_Segment'].isin(['核心VIP', '重要价值用户'])]
potential_users = df[df['User_Segment'].isin(['纠结土豪', '高潜沉睡用户', '犹豫型潜力用户', '高潜流失客'])]

# 惩罚机制: 核心占比0%→60%时, 潜力收益损失从50%线性递减至5%
penalty_start = 0.50
penalty_end = 0.05
penalty_full_ratio = 0.60

def calc_penalty(ratio_pct):
    if ratio_pct >= penalty_full_ratio * 100:
        return penalty_end
    frac = ratio_pct / (penalty_full_ratio * 100)
    return penalty_start - frac * (penalty_start - penalty_end)


# 模型计算
def calc_rfi_model(core_ratio, use_penalty=True):
    """修正版 B/C 模型"""
    core_count = min(len(core_users), int(total_budget * core_ratio // coupon_cost))
    remaining_budget = total_budget - core_count * coupon_cost
    potential_count = min(len(potential_users), int(remaining_budget // coupon_cost))
    count = core_count + potential_count
    cost = count * coupon_cost
    penalty = calc_penalty(core_ratio * 100) if use_penalty else 0
    core_rev = core_count * core_lift * aov
    pot_rev = potential_count * potential_lift * aov * (1 - penalty)
    revenue = core_rev + pot_rev
    roi = (revenue - cost) / cost * 100 if cost > 0 else 0
    return {
        'core_count': core_count, 'potential_count': potential_count,
        'cost': cost, 'penalty': penalty,
        'core_rev': core_rev, 'pot_rev': pot_rev,
        'revenue': revenue, 'roi': roi
    }

def calc_rfi_original():
    """原始虚高版B: 按年消费算防流失价值"""
    core_count = min(len(core_users), int(total_budget // coupon_cost))
    remaining_budget = total_budget - core_count * coupon_cost
    potential_count = min(len(potential_users), int(remaining_budget // coupon_cost))
    count = core_count + potential_count
    cost = count * coupon_cost
    core_annual_value = aov * core_users['Purchase_Frequency'].mean()
    core_retention_rev = core_count * 0.10 * core_annual_value  # 原假设: 挽回率10% × 年消费
    core_incremental_rev = core_count * core_lift * aov
    core_rev = core_retention_rev + core_incremental_rev
    pot_rev = potential_count * potential_lift * aov
    revenue = core_rev + pot_rev
    roi = (revenue - cost) / cost * 100 if cost > 0 else 0
    return {
        'core_count': core_count, 'potential_count': potential_count,
        'cost': cost, 'penalty': 0,
        'core_rev': core_rev, 'pot_rev': pot_rev,
        'revenue': revenue, 'roi': roi,
        'retention_rev': core_retention_rev,
        'incremental_rev': core_incremental_rev
    }

# A模型: 传统RFM Top20%
top20_threshold = df['RFM_Score'].quantile(0.80)
rfm_top20 = df[df['RFM_Score'] >= top20_threshold]
count_a = min(len(rfm_top20), int(total_budget // coupon_cost))
cost_a = count_a * coupon_cost
revenue_a = count_a * rfm20_lift * aov
roi_a = (revenue_a - cost_a) / cost_a * 100 if cost_a > 0 else 0

# 各场景计算
b_original = calc_rfi_original()
b_nopen = calc_rfi_model(0.80, use_penalty=False)
c_nopen = calc_rfi_model(0.50, use_penalty=False)
b_pen = calc_rfi_model(0.80, use_penalty=True)
c_pen = calc_rfi_model(0.50, use_penalty=True)


# 打印结果
print("=" * 50)
print("A/B 测试结果")
print("=" * 50)
print(f"数据规模: {len(df)} 行 | AOV: {aov:.2f}")
print(f"核心用户池: {len(core_users)} | 潜力用户池: {len(potential_users)}")
print(f"预算: {total_budget}元 | 券面额: {coupon_cost}元\n")

print("【阶段1: 原始虚高版B】")
print(f"  ROI={b_original['roi']:+.1f}%  收益={b_original['revenue']:+.0f}  成本={b_original['cost']}")

print("\n【阶段2: 修正无惩罚】")
print(f"  A模型:  ROI={roi_a:+.1f}%  收益={revenue_a:+.0f}  成本={cost_a}")
print(f"  B模型:  ROI={b_nopen['roi']:+.1f}%  收益={b_nopen['revenue']:+.0f}  成本={b_nopen['cost']}")
print(f"  C模型:  ROI={c_nopen['roi']:+.1f}%  收益={c_nopen['revenue']:+.0f}  成本={c_nopen['cost']}")

print("\n【阶段3: 修正有惩罚】")
print(f"  A模型:  ROI={roi_a:+.1f}%  收益={revenue_a:+.0f}  成本={cost_a}")
print(f"  B模型:  ROI={b_pen['roi']:+.1f}%  收益={b_pen['revenue']:+.0f}  成本={b_pen['cost']}  (罚{b_pen['penalty']:.0%})")
print(f"  C模型:  ROI={c_pen['roi']:+.1f}%  收益={c_pen['revenue']:+.0f}  成本={c_pen['cost']}  (罚{c_pen['penalty']:.0%})")


# 绘图: 3x2 布局
fig, axes = plt.subplots(3, 2, figsize=(18, 20))

# 第一排: 原始虚高 vs 修正
orig_labels = ['B模型_原\n(虚高)', 'B模型\n(修正无惩罚)', 'C模型\n(修正无惩罚)']
orig_colors = ['#9B59B6', '#4ECDC4', '#0B5CAD']

rois_orig = [b_original['roi'], b_nopen['roi'], c_nopen['roi']]
bars = axes[0, 0].bar(orig_labels, rois_orig, color=orig_colors, alpha=0.85, edgecolor='black', width=0.5)
for bar, roi in zip(bars, rois_orig):
    ypos = bar.get_height() + max(abs(r) for r in rois_orig) * 0.04
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{roi:+.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0, 0].set_title('ROI: 原始虚高 vs 修正', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('ROI (%)', fontsize=12)
axes[0, 0].grid(axis='y', alpha=0.3)

x_orig = np.arange(len(orig_labels))
width_orig = 0.5
l1_orig = [b_original['core_rev'], b_nopen['core_rev'], c_nopen['core_rev']]
l2_orig = [b_original['pot_rev'], b_nopen['pot_rev'], c_nopen['pot_rev']]
bottom_orig = np.zeros(len(orig_labels))
axes[0, 1].bar(x_orig, l1_orig, width_orig, bottom=bottom_orig, label='核心收益', color='#4ECDC4', alpha=0.85, edgecolor='black')
bottom_orig += np.array(l1_orig)
axes[0, 1].bar(x_orig, l2_orig, width_orig, bottom=bottom_orig, label='潜力收益', color='#FF6B6B', alpha=0.85, edgecolor='black')
for i, (v, b) in enumerate(zip(l1_orig, np.zeros(3))):
    if abs(v) > 30:
        axes[0, 1].text(x_orig[i], b + v / 2, f'{int(v):+,}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
for i, (v, b) in enumerate(zip(l2_orig, bottom_orig)):
    if abs(v) > 30:
        axes[0, 1].text(x_orig[i], b + v / 2, f'{int(v):+,}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
for i, cost in enumerate([b_original['cost'], b_nopen['cost'], c_nopen['cost']]):
    axes[0, 1].text(x_orig[i], -max(abs(b_original['revenue']), abs(b_nopen['revenue']), abs(c_nopen['revenue'])) * 0.04,
                    f'成本:{int(cost):,}', ha='center', va='top', fontsize=9, color='#555')
axes[0, 1].set_title('收益构成: 原始 vs 修正', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('收益 (元)', fontsize=12)
axes[0, 1].set_xticks(x_orig)
axes[0, 1].set_xticklabels(orig_labels, fontsize=10)
axes[0, 1].legend(loc='upper right', fontsize=9)
axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
axes[0, 1].grid(axis='y', alpha=0.3)

# 第二排: 无惩罚
model_labels = ['A模型', 'B模型', 'C模型']
bar_colors = ['#FF6B6B', '#4ECDC4', '#0B5CAD']

rois_nopen = [roi_a, b_nopen['roi'], c_nopen['roi']]
bars = axes[1, 0].bar(model_labels, rois_nopen, color=bar_colors, alpha=0.85, edgecolor='black', width=0.5)
for bar, roi in zip(bars, rois_nopen):
    ypos = bar.get_height() + max(abs(r) for r in rois_nopen) * 0.04
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{roi:+.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1, 0].set_title('无惩罚 ROI 对比', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('ROI (%)', fontsize=12)
axes[1, 0].grid(axis='y', alpha=0.3)

x = np.arange(len(model_labels))
width = 0.5
l1_nopen = [revenue_a, b_nopen['core_rev'], c_nopen['core_rev']]
l2_nopen = [0, b_nopen['pot_rev'], c_nopen['pot_rev']]
bottom = np.zeros(len(model_labels))
axes[1, 1].bar(x, l1_nopen, width, bottom=bottom, label='核心收益', color='#4ECDC4', alpha=0.85, edgecolor='black')
bottom += np.array(l1_nopen)
axes[1, 1].bar(x, l2_nopen, width, bottom=bottom, label='潜力收益', color='#FF6B6B', alpha=0.85, edgecolor='black')
for i, (v, b) in enumerate(zip(l1_nopen, np.zeros(3))):
    if abs(v) > 30:
        axes[1, 1].text(x[i], b + v / 2, f'{int(v):+,}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
for i, (v, b) in enumerate(zip(l2_nopen, bottom)):
    if abs(v) > 30:
        axes[1, 1].text(x[i], b + v / 2, f'{int(v):+,}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
for i, cost in enumerate([cost_a, b_nopen['cost'], c_nopen['cost']]):
    axes[1, 1].text(x[i], -max(abs(revenue_a), abs(b_nopen['revenue']), abs(c_nopen['revenue'])) * 0.04,
                    f'成本:{int(cost):,}', ha='center', va='top', fontsize=9, color='#555')
axes[1, 1].set_title('无惩罚收益构成', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('收益 (元)', fontsize=12)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(model_labels, fontsize=10)
axes[1, 1].legend(loc='upper right', fontsize=9)
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
axes[1, 1].grid(axis='y', alpha=0.3)

# 第三排: 有惩罚
rois_pen = [roi_a, b_pen['roi'], c_pen['roi']]
bars = axes[2, 0].bar(model_labels, rois_pen, color=bar_colors, alpha=0.85, edgecolor='black', width=0.5)
for bar, roi in zip(bars, rois_pen):
    ypos = bar.get_height() + max(abs(r) for r in rois_pen) * 0.04
    axes[2, 0].text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{roi:+.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
axes[2, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[2, 0].set_title('有惩罚 ROI 对比', fontsize=13, fontweight='bold')
axes[2, 0].set_ylabel('ROI (%)', fontsize=12)
axes[2, 0].grid(axis='y', alpha=0.3)

l1_pen = [revenue_a, b_pen['core_rev'], c_pen['core_rev']]
l2_pen = [0, b_pen['pot_rev'], c_pen['pot_rev']]
bottom = np.zeros(len(model_labels))
axes[2, 1].bar(x, l1_pen, width, bottom=bottom, label='核心收益', color='#4ECDC4', alpha=0.85, edgecolor='black')
bottom += np.array(l1_pen)
axes[2, 1].bar(x, l2_pen, width, bottom=bottom, label='潜力收益(扣罚后)', color='#FF6B6B', alpha=0.85, edgecolor='black')
for i, (v, b) in enumerate(zip(l1_pen, np.zeros(3))):
    if abs(v) > 30:
        axes[2, 1].text(x[i], b + v / 2, f'{int(v):+,}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
for i, (v, b) in enumerate(zip(l2_pen, bottom)):
    if abs(v) > 30:
        axes[2, 1].text(x[i], b + v / 2, f'{int(v):+,}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
for i, cost in enumerate([cost_a, b_pen['cost'], c_pen['cost']]):
    axes[2, 1].text(x[i], -max(abs(revenue_a), abs(b_pen['revenue']), abs(c_pen['revenue'])) * 0.04,
                    f'成本:{int(cost):,}', ha='center', va='top', fontsize=9, color='#555')
axes[2, 1].set_title('有惩罚收益构成', fontsize=13, fontweight='bold')
axes[2, 1].set_ylabel('收益 (元)', fontsize=12)
axes[2, 1].set_xticks(x)
axes[2, 1].set_xticklabels(model_labels, fontsize=10)
axes[2, 1].legend(loc='upper right', fontsize=9)
axes[2, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
axes[2, 1].grid(axis='y', alpha=0.3)

fig.suptitle('A/B/C 三模型 ROI 对比\n阶段1: 原始虚高 → 阶段2: 修正(无惩罚) → 阶段3: 最终(有惩罚)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_file = os.path.join(output_dir, 'ab_test_comparison.png')
plt.savefig(out_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n对比图已保存: {out_file}")