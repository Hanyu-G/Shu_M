import os
import pandas as pd
import matplotlib.pyplot as plt

# 配置绘图字体与负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("执行单次振打事件微观特征提取...")

# ---------------------------------------------------------
# 1. 路径配置与数据加载
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'Cleaned_Cement_ESP_Data.csv')

if not os.path.exists(data_path):
    print(f"Error: 未找到预处理数据文件: {data_path}")
    print("请确保已运行数据清洗脚本生成该文件。")
    exit()

df = pd.read_csv(data_path)

# ---------------------------------------------------------
# 2. 振打事件定位
# ---------------------------------------------------------
# 计算电场 4 振打计时器差分，识别复位时刻
df['T4_diff'] = df['T4_s'].diff()
# 筛选触发振打的时间索引 (阈值设为 -30)
rap_indices = df[df['T4_diff'] < -30].index

# 选取特定振打样本进行可视化（此处取第 10 次记录）
sample_idx = 10
if len(rap_indices) <= sample_idx:
    sample_idx = 0
    
target_idx = rap_indices[sample_idx]

# 设定观察窗口：振打时刻前 5 分钟至后 15 分钟
window_data = df.loc[target_idx - 5 : target_idx + 15].copy()

# ---------------------------------------------------------
# 3. 瞬态响应可视化
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(window_data.index, window_data['C_out_mgNm3'], 
         marker='o', linestyle='-', color='#1f77b4', label='出口粉尘浓度')

# 标记振打起始时刻
plt.axvline(x=target_idx, color='red', linestyle='--', linewidth=2, label='电场 4 振打触发点')

# 定位响应区间内的最高浓度点（滞后峰值）
peak_idx = window_data.loc[target_idx : target_idx + 15, 'C_out_mgNm3'].idxmax()
peak_value = window_data.loc[peak_idx, 'C_out_mgNm3']

# 绘制峰值标注点
plt.scatter(peak_idx, peak_value, color='red', s=150, marker='*', zorder=5, 
            label=f'滞后峰值 ({peak_value:.2f} mg/Nm³)')

plt.title('单次振打事件微观瞬态响应特性 (电场 4)', fontsize=16)
plt.xlabel('时间序列索引 (min)', fontsize=12)
plt.ylabel('出口粉尘浓度 (mg/Nm³)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# 保存可视化结果
plot_path = os.path.join(current_dir, 'Single_Rapping_Event_Zoom.png')
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"分析完成。图像已输出至: {plot_path}")