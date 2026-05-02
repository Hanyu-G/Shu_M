import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 配置可视化字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1. 路径配置与数据加载
# ---------------------------------------------------------
print("启动分析：时序延迟计算与振打峰值提取")

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'Cleaned_Cement_ESP_Data.csv')

if not os.path.exists(data_path):
    print(f"Error: 未找到指定数据文件: {data_path}")
    exit()
    
df = pd.read_csv(data_path)

# ---------------------------------------------------------
# 2. 振打时刻识别
# ---------------------------------------------------------
print("正在识别各电场振打时刻脉冲信号...")
for i in range(1, 5):
    col = f'T{i}_s'
    diff_col = f'T{i}_diff'
    df[diff_col] = df[col].diff()
    # 基于时序差分检测计时器复位点，差值小于 -30 识别为振打动作
    df[f'Rap_Signal_{i}'] = np.where(df[diff_col] < -30, 1, 0)

# ---------------------------------------------------------
# 3. 基于互相关的物理延迟计算 (物理约束版)
# ---------------------------------------------------------
print("执行互相关分析计算物理延迟...")
max_lag = 15 
best_lags = {}

plt.figure(figsize=(10, 6))
colors = ['blue', 'orange', 'green', 'red']

for i in range(1, 5):
    correlations = []
    target = df['C_out_mgNm3']
    signal = df[f'Rap_Signal_{i}']
    
    for lag in range(max_lag + 1):
        corr = signal.shift(lag).corr(target)
        correlations.append(corr)
        
    # 基于空间排布引入物理约束：电场 3 延迟强制约束在 [2, 4] 分钟窗口内寻优
    if i == 3:
        search_window = correlations[2:5] 
        best_lag = np.argmax(search_window) + 2 
    else:
        best_lag = np.argmax(correlations)
        
    best_lags[f'电场{i}'] = best_lag
    
    plt.plot(range(max_lag + 1), correlations, marker='o', 
             label=f'电场 {i} (最优延迟: {best_lag}min)', color=colors[i-1])

plt.title('振打脉冲与出口浓度互相关分析 (受限搜索寻优)', fontsize=16)
plt.xlabel('延迟滞后时间 (min)', fontsize=12)
plt.ylabel('互相关系数', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plot_path = os.path.join(current_dir, 'Cross_Correlation_Lags_Physics.png')
plt.savefig(plot_path, dpi=300) 
plt.show()

# ---------------------------------------------------------
# 4. 振打峰值提取与浓度增幅统计
# ---------------------------------------------------------
print("提取振打响应峰值并生成统计报表...")
results = []

for i in range(1, 5):
    signal_col = f'Rap_Signal_{i}'
    delay = best_lags[f'电场{i}']
    rap_indices = df[df[signal_col] == 1].index
    
    peaks = []
    baselines = []
    
    for idx in rap_indices:
        if idx >= 5 and idx + delay + 2 < len(df):
            # 基线浓度定义：振打前 5 分钟均值
            baseline = df.loc[idx-5:idx-1, 'C_out_mgNm3'].mean()
            baselines.append(baseline)
            
            # 峰值捕捉：在预设物理延迟附近取局部极大值
            peak_window = df.loc[idx+delay-1 : idx+delay+1, 'C_out_mgNm3']
            peaks.append(peak_window.max())
            
    baselines = np.array(baselines)
    peaks = np.array(peaks)
    increases = peaks - baselines
    increase_pct = (increases / baselines) * 100
    
    results.append({
        '电场': f'电场 {i}',
        '振打频次': len(rap_indices),
        '物理延迟(min)': delay,
        '平均基准浓度': np.mean(baselines),
        '平均峰值浓度': np.mean(peaks),
        '平均相对增幅(%)': np.mean(increase_pct),
        '最大瞬时峰值': np.max(peaks)
    })

# 构建并导出统计结果
result_df = pd.DataFrame(results).round(2)
print("\n=== 各电场振打响应特征汇总表 ===")
print(result_df.to_string(index=False))

csv_save_path = os.path.join(current_dir, 'Rapping_Peak_Statistics.csv')
result_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')

print(f"\n分析完成。")
print(f"可视化图表已输出至: {plot_path}")
print(f"统计数据已导出至: {csv_save_path}")