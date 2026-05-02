import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. 设置数据加载路径
DATA_PATH = os.path.join('..', 'Task2', 'Data_with_Conditions.csv')
OPT_PATH = os.path.join('..', 'Task2', 'Task2_Final_Optimization_Strategies.csv')

df_full = pd.read_csv(DATA_PATH)
df_opt = pd.read_csv(OPT_PATH)

print("=" * 75)
print("Task 3: 提取典型工况最优参数对比表")
print("=" * 75)

# 2. 筛选极端工况 (按入口浓度均值区分)
cond_conc = df_full.groupby('Condition_Name')['C_in_gNm3'].mean().sort_values()
cond_low = cond_conc.index[0]
cond_high = cond_conc.index[-1]

print(f"[工况筛选]\n低负荷典型工况: {cond_low} (均值: {cond_conc[cond_low]:.2f} g/Nm³)")
print(f"高负荷典型工况: {cond_high} (均值: {cond_conc[cond_high]:.2f} g/Nm³)")

# 3. 提取优化参数与基础数据
opt_low = df_opt[df_opt['Condition_Name'] == cond_low].iloc[0]
opt_high = df_opt[df_opt['Condition_Name'] == cond_high].iloc[0]

low_data = df_full[df_full['Condition_Name'] == cond_low]
high_data = df_full[df_full['Condition_Name'] == cond_high]

# 4. 构建对比数据表
comparison_data = {
    '参数': [
        '入口浓度(g/Nm³)', '入口温度(℃)', '流量(Nm³/h)',
        'U1(kV)', 'U2(kV)', 'U3(kV)', 'U4(kV)',
        'T1(s)', 'T2(s)', 'T3(s)', 'T4(s)',
        '出口浓度(mg/Nm³)', '总电耗(kW)'
    ],
    cond_low: [
        f'{cond_conc[cond_low]:.2f}', f'{low_data["Temp_C"].mean():.1f}', f'{low_data["Q_Nm3h"].mean():.0f}',
        f'{opt_low["U1_kV"]:.1f}', f'{opt_low["U2_kV"]:.1f}', f'{opt_low["U3_kV"]:.1f}', f'{opt_low["U4_kV"]:.1f}',
        f'{opt_low["T1_s"]:.0f}', f'{opt_low["T2_s"]:.0f}', f'{opt_low["T3_s"]:.0f}', f'{opt_low["T4_s"]:.0f}',
        f'{opt_low["C_out_optimized"]:.2f}', f'{opt_low["P_total_optimized"]:.2f}'
    ],
    cond_high: [
        f'{cond_conc[cond_high]:.2f}', f'{high_data["Temp_C"].mean():.1f}', f'{high_data["Q_Nm3h"].mean():.0f}',
        f'{opt_high["U1_kV"]:.1f}', f'{opt_high["U2_kV"]:.1f}', f'{opt_high["U3_kV"]:.1f}', f'{opt_high["U4_kV"]:.1f}',
        f'{opt_high["T1_s"]:.0f}', f'{opt_high["T2_s"]:.0f}', f'{opt_high["T3_s"]:.0f}', f'{opt_high["T4_s"]:.0f}',
        f'{opt_high["C_out_optimized"]:.2f}', f'{opt_high["P_total_optimized"]:.2f}'
    ]
}

comparison_df = pd.DataFrame(comparison_data)

# 5. 输出与保存
print(f"\n{'-'*75}")
print(comparison_df.to_string(index=False))
print(f"{'-'*75}")

# 保存为 CSV
csv_save_path = 'table1.csv'
comparison_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
print(f"数据表已保存至: {csv_save_path}")

# 传递工况信息给后续脚本
info_dict = {'low': cond_low, 'high': cond_high}
npy_save_path = 'info.npy'
np.save(npy_save_path, info_dict)
print(f"工况配置已保存至: {npy_save_path}")