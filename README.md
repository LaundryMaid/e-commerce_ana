# 电商用户分层与优惠券 ROI 分析

RFM-I电商用户分层，结合 A/B 测试和敏感性分析优化优惠券投放策略。

## 项目结构

```
e-commerce_ana/
├── main.py              # 数据预处理 + RFM-I 建模 + 用户分层 + ROI 测算
├── ab_test.py           # A/B 测试: A/B/C 三模型对比
├── roi_corrected.py     # 核心/潜力预算配比敏感性分析
├── code_explanation.txt # 代码说明文档
├── data/                # 数据与输出目录
│   └── README.md        # 数据集字段说明 (不含原始数据)
└── .gitignore
```

## 分析流程

1. **main.py** — 数据加载 → 特征工程(意向分/阻力系数/忠诚度) → RFM-I 建模 → 用户分层(15类) → ROI 测算
2. **ab_test.py** — 三模型对比(A:传统RFM / B:80-20 / C:50-50)，展示从原始虚高到修正的完整过程
3. **roi_corrected.py** — 0%~100% 核心预算占比敏感性分析，含无惩罚和有惩罚两种场景

## 核心方法

- **RFM-I 模型**: 在 RFM 基础上加入意向分(I_Score)，识别"高意向低转化"的潜力用户
- **分层差异化定价**: 核心用户转化提升 5%，潜力用户 19%
- **防流失价值**: 按单次 AOV × 挽回率 × 可信度折扣计算(非年消费额)
- **惩罚机制**: 核心占比 0%→60% 时，潜力收益损失从 50% 线性递减至 5%

## 运行

```bash
pip install pandas numpy matplotlib seaborn

python main.py
python ab_test.py
python roi_corrected.py
```

输出结果保存至 `data/` 目录。
