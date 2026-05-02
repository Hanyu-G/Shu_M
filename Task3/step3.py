import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 设定工作目录为当前脚本所在路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------
# 1. 全局配置与路径设定
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PRIORITY_FILE = 'priority_results.csv'
OPT_STRATEGY_FILE = os.path.join('..', 'Task2', 'Task2_Final_Optimization_Strategies.csv')
INFO_FILE = 'info.npy'
OUTPUT_IMAGE = 'result.png'

print("=" * 75)
print("Task 3: 典型工况最优控制策略与敏感性差异对比分析")
print("=" * 75)

# ---------------------------------------------------------
# 2. 数据加载与预处理
# ---------------------------------------------------------
df_priority = pd.read_csv(PRIORITY_FILE)
df_opt = pd.read_csv(OPT_STRATEGY_FILE)
info = np.load(INFO_FILE, allow_pickle=True).item()

cond_low = info['low']
cond_high = info['high']

# 提取边际贡献指标
row_low = df_priority[df_priority['工况'] == cond_low].iloc[0]
row_high = df_priority[df_priority['工况'] == cond_high].iloc[0]

M_U_low, M_T_low = row_low['M_U'], row_low['M_T']
M_U_high, M_T_high = row_high['M_U'], row_high['M_T']

# 提取最优控制参数
opt_low = df_opt[df_opt['Condition_Name'] == cond_low].iloc[0]
opt_high = df_opt[df_opt['Condition_Name'] == cond_high].iloc[0]

# ---------------------------------------------------------
# 3. 核心可视化绘图模块
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x_positions = np.arange(4)
bar_width = 0.35
field_labels = ['E1', 'E2', 'E3', 'E4']

# --- 子图 1：各电场最优电压对比 ---
axes[0, 0].bar(x_positions - bar_width/2, opt_low[['U1_kV', 'U2_kV', 'U3_kV', 'U4_kV']], 
               bar_width, label=f'{cond_low}\n(C_in={row_low["C_in"]} g/Nm$^3$)', color='steelblue')
axes[0, 0].bar(x_positions + bar_width/2, opt_high[['U1_kV', 'U2_kV', 'U3_kV', 'U4_kV']], 
               bar_width, label=f'{cond_high}\n(C_in={row_high["C_in"]} g/Nm$^3$)', color='darkorange')
axes[0, 0].set_title('各电场最优电压对比', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('电压 (kV)')
axes[0, 0].set_xticks(x_positions)
axes[0, 0].set_xticklabels(field_labels)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(axis='y', alpha=0.3)

# --- 子图 2：各电场最优振打周期对比 ---
axes[0, 1].bar(x_positions - bar_width/2, opt_low[['T1_s', 'T2_s', 'T3_s', 'T4_s']], 
               bar_width, label=cond_low, color='steelblue')
axes[0, 1].bar(x_positions + bar_width/2, opt_high[['T1_s', 'T2_s', 'T3_s', 'T4_s']], 
               bar_width, label=cond_high, color='darkorange')
axes[0, 1].set_title('各电场最优振打周期对比', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('振打周期 (s)')
axes[0, 1].set_xticks(x_positions)
axes[0, 1].set_xticklabels(field_labels)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(axis='y', alpha=0.3)

# --- 子图 3：目标函数边际贡献对比 ---
x_margin = np.arange(2)
axes[1, 0].bar(x_margin - bar_width/2, [M_U_low, M_T_low], bar_width, label=cond_low, color='steelblue')
axes[1, 0].bar(x_margin + bar_width/2, [M_U_high, M_T_high], bar_width, label=cond_high, color='darkorange')
axes[1, 0].set_title('目标函数边际贡献对比', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('M (%)')
axes[1, 0].set_xticks(x_margin)
axes[1, 0].set_xticklabels(['电压 M_U', '振打 M_T'])
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(axis='y', alpha=0.3)

# --- 子图 4：排放浓度与系统总电耗 ---
x_perf = np.arange(2)
axes[1, 1].bar(x_perf - bar_width/2, [opt_low['C_out_optimized'], opt_low['P_total_optimized']/10],
               bar_width, label=cond_low, color='steelblue')
axes[1, 1].bar(x_perf + bar_width/2, [opt_high['C_out_optimized'], opt_high['P_total_optimized']/10],
               bar_width, label=cond_high, color='darkorange')
axes[1, 1].set_title('排放浓度与总电耗', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x_perf)
axes[1, 1].set_xticklabels(['出口浓度 (mg/Nm$^3$)', '系统电耗/10 (kW)'])
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(axis='y', alpha=0.3)

# ---------------------------------------------------------
# 4. 图表导出与终端输出
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')

print(f">> 成功：可视化图表已生成并保存为 {OUTPUT_IMAGE} (300 DPI)\n")

print(f"{'-'*75}")
print("全工况优先级分析结果汇总")
print(f"{'-'*75}")

# 控制台对齐输出
header = f"{'工况':<14} {'C_in':>6} {'C_out':>6} {'P_opt':>8} {'M_U(%)':>8} {'M_T(%)':>6}  {'Priority':<8}"
print(header)
print("-" * 75)

for _, row in df_priority.iterrows():
    cond_name = row['工况']
    pad_len = 8 - (len(cond_name.encode('gbk')) // 2) 
    formatted_cond = cond_name + '\u3000' * pad_len
    print(f"{formatted_cond} {row['C_in']:>6.2f} {row['C_out']:>6.2f} {row['P_opt']:>8.2f} {row['M_U']:>8.2f} {row['M_T']:>6.2f}  {row['priority']:<8}")

print(f"{'='*75}\n")