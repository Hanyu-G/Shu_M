import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 路径配置
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, 'Cement_ESP_Data.csv')
output_path = os.path.join(current_dir, 'Cleaned_Cement_ESP_Data.csv')

# ---------------------------------------------------------
# 2. 数据加载与探索
# ---------------------------------------------------------
print("开始读取数据...")
if not os.path.exists(input_path):
    print("Error: 未找到原始数据文件。")
    print(f"请确认文件是否存在于该路径: {input_path}")
    exit()

df = pd.read_csv(input_path)

print("数据总维度 (行, 列):", df.shape)
print("\n缺失值统计:")
print(df.isnull().sum())

# ---------------------------------------------------------
# 3. 缺失值处理
# ---------------------------------------------------------
# 针对具备连续时间属性的 C_out_mgNm3 特征，采用线性插值法补全缺失数据
df['C_out_mgNm3'] = df['C_out_mgNm3'].interpolate(method='linear')

print("\n插值处理后 C_out_mgNm3 的缺失值数量:", df['C_out_mgNm3'].isnull().sum())

# ---------------------------------------------------------
# 4. 数据平滑降噪
# ---------------------------------------------------------
# 对工况特征应用 3 分钟滑动平均处理，滤除高频测量噪声。
# 设定参数（电压、周期）保持原始状态。
cols_to_smooth = ['Temp_C', 'C_in_gNm3', 'Q_Nm3h', 'C_out_mgNm3']

for col in cols_to_smooth:
    df[col + '_smooth'] = df[col].rolling(window=3, min_periods=1).mean()

print("\n数据预处理完成，平滑特征前 5 行预览:")
print(df[['timestamp', 'C_out_mgNm3', 'C_out_mgNm3_smooth']].head())

# ---------------------------------------------------------
# 5. 结果导出
# ---------------------------------------------------------
df.to_csv(output_path, index=False)
print(f"\n预处理数据集已保存至:\n{output_path}")