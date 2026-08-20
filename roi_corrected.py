#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心/潜力预算配比敏感性分析

场景1: 无惩罚机制
场景2: 有惩罚机制 (50%→5%线性递减至60%核心占比)
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

core_users = df[df['User_Segment'].isin(['核心VIP', '重要价值用户'])]
potential_users = df[df['User_Segment'].isin(['纠结土豪', '高潜沉睡用户', '犹豫型潜力用户', '高潜流失客'])]

# 惩罚机制
penalty_start = 0.50
penalty_end = 0.05
penalty_full_ratio = 0.60

def calc_penalty(ratio_pct):
    if ratio_pct >= penalty_full_ratio * 100:
        return penalty_end
    frac = ratio_pct / (penalty_full_ratio * 100)
    return penalty_start - frac * (penalty_start - penalty_end)

ratios = list(range(0, 101, 10))


# 场景1: 无惩罚
roi_list_np = []
revenue_list_np = []

for ratio in ratios:
    cc = min(len(core_users), int(total_budget * ratio / 100 // coupon_cost))
    rb = total_budget - cc * coupon_cost
    pc = min(len(potential_users), int(rb // coupon_cost))
    rev = cc * core_lift * aov + pc * potential_lift * aov
    cost = (cc + pc) * coupon_cost
    r = (rev - cost) / cost * 100 if cost > 0 else 0
    roi_list_np.append(r)
    revenue_list_np.append(rev)

max_idx_np = roi_list_np.index(max(roi_list_np))
min_idx_np = roi_list_np.index(min(roi_list_np))


# 场景2: 有惩罚
roi_list_p = []
revenue_list_p = []
penalty_list = []

for ratio in ratios:
    cc = min(len(core_users), int(total_budget * ratio / 100 // coupon_cost))
    rb = total_budget - cc * coupon_cost
    pc = min(len(potential_users), int(rb // coupon_cost))
    core_rev = cc * core_lift * aov
    pen = calc_penalty(ratio)
    pot_rev = pc * potential_lift * aov * (1 - pen)
    rev = core_rev + pot_rev
    cost = (cc + pc) * coupon_cost
    r = (rev - cost) / cost * 100 if cost > 0 else 0
    roi_list_p.append(r)
    revenue_list_p.append(rev)
    penalty_list.append(pen * 100)

max_idx_p = roi_list_p.index(max(roi_list_p))
min_idx_p = roi_list_p.index(min(roi_list_p))


# 打印
print("=" * 50)
print("预算配比敏感性分析")
print("=" * 50)
print(f"预算: {total_budget}元 | 券面额: {coupon_cost}元 | AOV: {aov:.0f}元\n")

print("【无惩罚】")
print(f"  最优: 核心{ratios[max_idx_np]}% → ROI {roi_list_np[max_idx_np]:.1f}%")
for i, ratio in enumerate(ratios):
    print(f"  核心{ratio:3d}%: ROI {roi_list_np[i]:+6.1f}%  收益 {revenue_list_np[i]:+8.0f}")

print(f"\n【有惩罚】")
print(f"  最优: 核心{ratios[max_idx_p]}% → ROI {roi_list_p[max_idx_p]:.1f}%")
for i, ratio in enumerate(ratios):
    print(f"  核心{ratio:3d}%: ROI {roi_list_p[i]:+6.1f}%  收益 {revenue_list_p[i]:+8.0f}  (罚{penalty_list[i]:.0f}%)")


# 绘图: 场景1
fig1, ax1 = plt.subplots(figsize=(14, 7))

x = np.arange(len(ratios))
bar_width = 0.55
bar_colors_np = ['#4ECDC4' if r >= 0 else '#FF6B6B' for r in roi_list_np]
bars1 = ax1.bar(x, revenue_list_np, bar_width, color=bar_colors_np, alpha=0.75,
                edgecolor='black', label='收益 (元)')
for bar, rev in zip(bars1, revenue_list_np):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(revenue_list_np) * 0.015,
             f'{int(rev)}', ha='center', va='bottom', fontsize=7, color='#333')

ax1.set_xlabel('核心用户预算占比 (%)')
ax1.set_ylabel('收益 (元)')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{r}%' for r in ratios], fontsize=9)
ax1.grid(axis='y', alpha=0.3)

ax1r = ax1.twinx()
ax1r.plot(x, roi_list_np, color='#0B5CAD', marker='o', linewidth=2.5,
          markersize=7, label='ROI (%)', zorder=5)
for xi, yi in zip(x, roi_list_np):
    offset = 3 if yi >= 0 else -6
    va = 'bottom' if yi >= 0 else 'top'
    ax1r.annotate(f'{yi:.1f}%', (xi, yi), textcoords="offset points",
                  xytext=(0, offset * 4), ha='center', va=va,
                  fontsize=7, fontweight='bold', color='#0B5CAD')
ax1r.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax1r.set_ylabel('ROI (%)', color='#0B5CAD')

best_x1 = x[max_idx_np]
best_y1 = roi_list_np[max_idx_np]
ax1r.plot(best_x1, best_y1, 'o', color='#FFD166', markersize=14,
          markeredgecolor='black', markeredgewidth=2, zorder=10)
ax1r.annotate(f'最优: {best_y1:.1f}%', (best_x1, best_y1),
              textcoords="offset points", xytext=(8, 10), ha='left',
              fontsize=9, fontweight='bold', color='#B8860B',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD', edgecolor='#B8860B'))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

ax1.set_title('预算配比对ROI与收益的影响 — 无惩罚', fontsize=12, fontweight='bold')
plt.tight_layout()
out1 = os.path.join(output_dir, 'roi_ratio_no_penalty.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图1已保存: {out1}")


# 绘图: 场景2
fig2, ax2 = plt.subplots(figsize=(14, 7))

bar_colors_p = ['#4ECDC4' if r >= 0 else '#FF6B6B' for r in roi_list_p]
bars2 = ax2.bar(x, revenue_list_p, bar_width, color=bar_colors_p, alpha=0.75,
                edgecolor='black', label='实际收益(扣罚后)')
ax2.bar(x, revenue_list_np, bar_width, color='none',
        edgecolor='#999', linewidth=0.8, linestyle='--',
        label='无惩罚收益(参考)', zorder=1)

for bar, rev in zip(bars2, revenue_list_p):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(revenue_list_p) * 0.015,
             f'{int(rev)}', ha='center', va='bottom', fontsize=7, color='#333')

ax2.set_xlabel('核心用户预算占比 (%)')
ax2.set_ylabel('收益 (元)')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{r}%' for r in ratios], fontsize=9)
ax2.grid(axis='y', alpha=0.3)

ax2r = ax2.twinx()
ax2r.plot(x, roi_list_p, color='#0B5CAD', marker='o', linewidth=2.5,
          markersize=7, label='ROI (%)', zorder=5)
for xi, yi in zip(x, roi_list_p):
    offset = 3 if yi >= 0 else -6
    va = 'bottom' if yi >= 0 else 'top'
    ax2r.annotate(f'{yi:.1f}%', (xi, yi), textcoords="offset points",
                  xytext=(0, offset * 4), ha='center', va=va,
                  fontsize=7, fontweight='bold', color='#0B5CAD')
ax2r.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax2r.set_ylabel('ROI (%)', color='#0B5CAD')

ax2r.plot(x, penalty_list, color='#999', marker='s', linewidth=1.5,
          markersize=5, linestyle=':', alpha=0.6, label='惩罚率 (%)', zorder=4)

best_x2 = x[max_idx_p]
best_y2 = roi_list_p[max_idx_p]
ax2r.plot(best_x2, best_y2, 'o', color='#FFD166', markersize=14,
          markeredgecolor='black', markeredgewidth=2, zorder=10)
ax2r.annotate(f'最优: {best_y2:.1f}%', (best_x2, best_y2),
              textcoords="offset points", xytext=(8, 10), ha='left',
              fontsize=9, fontweight='bold', color='#B8860B',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD', edgecolor='#B8860B'))

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

ax2.set_title('预算配比对ROI与收益的影响 — 有惩罚', fontsize=12, fontweight='bold')
plt.tight_layout()
out2 = os.path.join(output_dir, 'roi_ratio_with_penalty.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {out2}")


# 绘图: 两场景对比
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

ax3a.plot(x, roi_list_np, color='#0B5CAD', marker='o', linewidth=2, markersize=6, label='无惩罚ROI')
ax3a.plot(x, roi_list_p, color='#FF6B6B', marker='s', linewidth=2, markersize=6, label='有惩罚ROI')
ax3a.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax3a.set_xlabel('核心用户预算占比 (%)')
ax3a.set_ylabel('ROI (%)')
ax3a.set_title('ROI 对比')
ax3a.set_xticks(x)
ax3a.set_xticklabels([f'{r}%' for r in ratios], fontsize=8)
ax3a.legend(fontsize=9)
ax3a.grid(alpha=0.3)

ax3b.plot(x, revenue_list_np, color='#0B5CAD', marker='o', linewidth=2, markersize=6, label='无惩罚收益')
ax3b.plot(x, revenue_list_p, color='#FF6B6B', marker='s', linewidth=2, markersize=6, label='有惩罚收益')
ax3b.fill_between(x, revenue_list_p, revenue_list_np,
                  alpha=0.15, color='gray', label='惩罚损失')
ax3b.set_xlabel('核心用户预算占比 (%)')
ax3b.set_title('收益对比')
ax3b.set_xticks(x)
ax3b.set_xticklabels([f'{r}%' for r in ratios], fontsize=8)
ax3b.legend(fontsize=9)
ax3b.grid(alpha=0.3)

plt.suptitle('惩罚机制对ROI与收益的影响对比', fontsize=13, fontweight='bold')
plt.tight_layout()
out3 = os.path.join(output_dir, 'roi_ratio_comparison.png')
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {out3}")