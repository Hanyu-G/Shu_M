import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. 环境配置与中文字体设置
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 2. 路径配置与数据加载
# ---------------------------------------------------------
print("执行：振打周期与排放峰值关联性深度分析")

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'Cleaned_Cement_ESP_Data.csv')

if not os.path.exists(data_path):
    print(f"Error: 未找到数据文件: {data_path}")
    exit()

df = pd.read_csv(data_path)

# ---------------------------------------------------------
# 3. 参数设定与振打事件提取
# ---------------------------------------------------------
# 选取电场4作为分析对象（离出口最近，响应特征最显著）
target_ef = 4
delay = 1  # 预设物理滞后延迟

col_t = f'T{target_ef}_s'
# 识别振打脉冲点（差分值小于 -30 识别为计时器复位）
df['T_diff'] = df[col_t].diff()
rap_indices = df[df['T_diff'] < -30].index

cycle_lengths = []
peak_increases = []

print(f"提取电场 {target_ef} 的振打特征数据...")

for idx in rap_indices:
    # 边界检查，确保存在足够的基线和响应窗口
    if idx >= 5 and idx + delay + 2 < len(df):
        # 1. 提取实际振打周期（计时器归零前一时刻的数值）
        cycle_len = df.loc[idx-1, col_t]
        
        # 2. 计算基线浓度（振打前 5 分钟均值）
        baseline = df.loc[idx-5:idx-1, 'C_out_mgNm3'].mean()
        
        # 3. 捕捉响应峰值（在物理延迟窗口内取极大值）
        peak_val = df.loc[idx+delay-1 : idx+delay+1, 'C_out_mgNm3'].max()
        
        # 计算出口浓度瞬时飙升绝对量
        increase = peak_val - baseline
        
        # 过滤非正值异常数据及停机工况
        if increase > 0 and cycle_len > 0:
            cycle_lengths.append(cycle_len)
            peak_increases.append(increase)

# 构建分析数据集
analyze_df = pd.DataFrame({
    '振打周期时长 (s)': cycle_lengths,
    '浓度峰值增幅 (mg/Nm³)': peak_increases
})

# ---------------------------------------------------------
# 4. 回归分析可视化绘制
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.regplot(
    data=analyze_df, 
    x='振打周期时长 (s)', 
    y='浓度峰值增幅 (mg/Nm³)', 
    scatter_kws={'alpha': 0.6, 'color': '#2ca02c', 's': 30},
    line_kws={'color': '#d62728', 'linewidth': 2}
)

plt.title('振打周期与瞬时排放峰值增幅关联性分析 (电场4)', fontsize=16)
plt.xlabel('振打周期设定时长 (s)', fontsize=12)
plt.ylabel('出口粉尘浓度瞬时飙升量 (mg/Nm³)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 保存分析结果图表
plot_path = os.path.join(current_dir, 'Cycle_vs_Peak_Regression.png')
plt.savefig(plot_path, dpi=300)
plt.show()

# ---------------------------------------------------------
# 5. 统计特征输出
# ---------------------------------------------------------
corr_val = analyze_df['振打周期时长 (s)'].corr(analyze_df['浓度峰值增幅 (mg/Nm³)'])
print(f"\n分析完成。")
print(f"图表输出路径: {plot_path}")
print(f"振打周期与峰值增幅的 Pearson 相关系数为: {corr_val:.4f}")