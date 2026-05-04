import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# 忽略环境与版本警告
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------
# 1. 全局绘图参数配置
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# ---------------------------------------------------------
# 2. 数据加载
# ---------------------------------------------------------
DATA_PATH = 'Cement_ESP_Data.csv'
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

# ---------------------------------------------------------
# 3. 序列提取与跳变点检测
# ---------------------------------------------------------
VIEW_ROWS = 200
T1_series = df['T1_s'].head(VIEW_ROWS)

# 计算一阶前向差分以定位计时器重置时刻
diff_series = T1_series.diff()

# 设定振打触发判定的负向跳变阈值
DROP_THRESHOLD = -25 

# 获取跳变点索引
rapping_indices = diff_series[diff_series < DROP_THRESHOLD].index

# ---------------------------------------------------------
# 4. 可视化绘制
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

# 绘制基础时序序列
ax.plot(T1_series.index, T1_series.values, 
        color='#1f77b4', linewidth=1.5, marker='.', markersize=6, 
        label='T1_s (振打计时器)')

# 散点标记跳变位置
ax.scatter(rapping_indices, T1_series.loc[rapping_indices], 
           color='red', s=60, zorder=5, edgecolors='black', 
           label=f'振打触发点 ($\Delta T < {DROP_THRESHOLD}$s)')

# 添加标注与指示箭头
for idx in rapping_indices:
    ax.annotate('振打/重置', 
                xy=(idx, T1_series.loc[idx]), 
                xytext=(idx + 2, T1_series.loc[idx] + 15),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
                fontsize=9, color='darkred', rotation=0)

# 设置图表属性
ax.set_title('图 X: 一电场振打周期 ($T_1$) 锯齿波特征及振打时刻检测示意图', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('时间序列 (分钟)', fontsize=12)
ax.set_ylabel('振打计时器数值 ($T_1$/s)', fontsize=12)

ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# ---------------------------------------------------------
# 5. 图像输出
# ---------------------------------------------------------
plt.tight_layout()
OUTPUT_FILENAME = 'Task1_T1_Sawtooth_Analysis.png'
plt.savefig(OUTPUT_FILENAME, bbox_inches='tight')

print(f"分析完成。共检测到振打事件: {len(rapping_indices)} 次。")
print(f"时序特征图已保存至: {OUTPUT_FILENAME}")