import os
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------
# 1. 全局配置与路径设定
# ---------------------------------------------------------
DATA_PATH = 'Data_with_Conditions.csv'
MODEL_DIR = 'Models'
OUTPUT_CSV = 'Task2_Final_Optimization_Strategies.csv'

LAMBDA_E = 10000.0  # 环保超标惩罚系数
LAMBDA_D = 2.0      # 极板积灰惩罚系数
ALPHA = 1.5         # 振打周期安全宽容度
TARGET_CONC = 10.0  # 法定排放限值
K_EMPIRICAL = 1.35  # Deutsch经验常数 (极其关键的机理兜底参数)

print("=" * 75)
print("Task 2: 基于非对称DE算法与双驱模型的多工况协同优化")
print("=" * 75)

try:
    df_full = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"错误: 找不到 {DATA_PATH}")
    exit()

results = []

# ---------------------------------------------------------
# 2. 核心寻优引擎
# ---------------------------------------------------------
for cond_name in df_full['Condition_Name'].unique():
    
    cond_data = df_full[df_full['Condition_Name'] == cond_name].copy()
    
    mean_temp = cond_data['Temp_C'].mean()
    mean_cin = cond_data['C_in_gNm3'].mean()
    mean_q = cond_data['Q_Nm3h'].mean()
    mean_T = [cond_data[f'T{i}_s'].mean() for i in range(1, 5)]
    
    hist_power = cond_data['P_total_kW'].mean()
    
    # 获取历史电压总和极值，作为激活物理公式的阈值
    u_sum_history_max = sum(cond_data[f'U{i}_kV'].max() for i in range(1, 5))
    
    model_power = joblib.load(os.path.join(MODEL_DIR, f'model_power_{cond_name}.pkl'))
    model_conc = joblib.load(os.path.join(MODEL_DIR, f'model_conc_{cond_name}.pkl'))

    # 【创新点前置】非对称寻优边界
    bounds = []
    bounds.append((cond_data['U1_kV'].mean() * 0.7, cond_data['U1_kV'].max() * 1.1))
    bounds.append((cond_data['U2_kV'].mean() * 0.7, cond_data['U2_kV'].max() * 1.2))
    bounds.append((cond_data['U3_kV'].mean() * 0.7, cond_data['U3_kV'].max() * 1.5))
    bounds.append((cond_data['U4_kV'].mean() * 0.7, cond_data['U4_kV'].max() * 1.5))
    
    for i in range(1, 5):
        t_max = cond_data[f'T{i}_s'].max()
        bounds.append((60, max(200, t_max * 1.2)))
    
    # ---------------------------------------------------------
    # 3. 数据与机理双驱目标函数
    # ---------------------------------------------------------
    def objective(x):
        input_p = pd.DataFrame([{
            'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
            'U1_kV': x[0], 'U2_kV': x[1], 'U3_kV': x[2], 'U4_kV': x[3],
            'T1_s': x[4], 'T2_s': x[5], 'T3_s': x[6], 'T4_s': x[7]
        }])
        
        input_c = pd.DataFrame([{
            'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
            'U1_kV': x[0], 'U2_kV': x[1], 'U3_kV': x[2], 'U4_kV': x[3],
            'T1_s_aligned': x[4], 'T2_s_aligned': x[5],
            'T3_s_aligned': x[6], 'T4_s_aligned': x[7]
        }])
        
        pred_power = model_power.predict(input_p)[0]
        base_conc = model_conc.predict(input_c)[0]
        
        # 【修复的核心】机理公式平滑兜底机制
        u_sum_current = sum(x[0:4])
        if u_sum_current > u_sum_history_max:
            q_scaled = mean_q / 10000.0
            final_conc = mean_cin * 1000 * np.exp(-K_EMPIRICAL * u_sum_current / q_scaled)
        else:
            final_conc = base_conc
        
        emission_penalty = LAMBDA_E * max(0, final_conc - TARGET_CONC)**2
        
        dust_penalty = 0
        for i in range(4):
            t_current = x[4+i]
            t_safe_limit = mean_T[i] * ALPHA
            if t_current > t_safe_limit:
                dust_penalty += LAMBDA_D * (t_current - t_safe_limit)**2
        
        return pred_power + emission_penalty + dust_penalty

    print(f"\n[{cond_name}] 正在进行 DE 全局寻优 (目标 < 10mg)...")
    
    result = differential_evolution(
        objective, bounds,
        strategy='best1bin', popsize=20, mutation=(0.5, 1.2),
        recombination=0.8, seed=42, disp=False
    )
    best_x = result.x
    
    # 还原真实预测值
    opt_power = model_power.predict(pd.DataFrame([{
        'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
        'U1_kV': best_x[0], 'U2_kV': best_x[1], 'U3_kV': best_x[2], 'U4_kV': best_x[3],
        'T1_s': best_x[4], 'T2_s': best_x[5], 'T3_s': best_x[6], 'T4_s': best_x[7]
    }]))[0]
    
    # 结果展示同样需要经过机理判断
    u_sum_final = sum(best_x[0:4])
    if u_sum_final > u_sum_history_max:
        q_scaled_final = mean_q / 10000.0
        opt_conc = mean_cin * 1000 * np.exp(-K_EMPIRICAL * u_sum_final / q_scaled_final)
    else:
        opt_conc = model_conc.predict(pd.DataFrame([{
            'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
            'U1_kV': best_x[0], 'U2_kV': best_x[1], 'U3_kV': best_x[2], 'U4_kV': best_x[3],
            'T1_s_aligned': best_x[4], 'T2_s_aligned': best_x[5],
            'T3_s_aligned': best_x[6], 'T4_s_aligned': best_x[7]
        }]))[0]
    
    power_save_pct = (1.0 - opt_power / hist_power) * 100.0 if hist_power > 0 else 0
    
    results.append({
        'Condition_Name': cond_name,
        'U1_kV': round(best_x[0], 2), 'U2_kV': round(best_x[1], 2),
        'U3_kV': round(best_x[2], 2), 'U4_kV': round(best_x[3], 2),
        'T1_s': round(best_x[4], 0), 'T2_s': round(best_x[5], 0),
        'T3_s': round(best_x[6], 0), 'T4_s': round(best_x[7], 0),
        'C_out_optimized': round(opt_conc, 2),
        'P_total_optimized': round(opt_power, 2),
        'Historical_P': round(hist_power, 2),
        'Power_Save_%': round(power_save_pct, 2)
    })
    
    print(f"  √ 优化完成! 出口浓度: {opt_conc:.2f} mg/Nm³ | 优化后电耗: {opt_power:.2f} kW (节能 {power_save_pct:.2f}%)")

df_results = pd.DataFrame(results)

print(f"\n{'='*75}")
header = "工况类型" + "　"*4 + "   出口浓度   优化电耗    历史电耗   综合节能率(%)"
print(header)
print("-" * 75)

for _, row in df_results.iterrows():
    cond_name = row['Condition_Name']
    pad_len = 8 - (len(cond_name.encode('gbk')) // 2)
    formatted_cond = cond_name + '\u3000' * pad_len
    print(f"{formatted_cond} {row['C_out_optimized']:>10.2f} {row['P_total_optimized']:>10.2f} {row['Historical_P']:>11.2f} {row['Power_Save_%']:>13.2f}")

df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n{'='*75}")
print(f"所有工况的优化策略已成功生成并保存至: {OUTPUT_CSV}")
print(f"{'='*75}\n")