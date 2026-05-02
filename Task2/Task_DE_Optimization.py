import pandas as pd
import numpy as np
import os
import joblib
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')

print("开始执行多工况联合差分进化(DE)寻优计算...")

# 1. 环境配置与路径检查
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'Data_with_Conditions.csv')
model_dir = os.path.join(current_dir, 'Models')

if not os.path.exists(data_path) or not os.path.exists(model_dir):
    raise FileNotFoundError("未找到数据文件或Models目录，请检查前置步骤。")

df_full = pd.read_csv(data_path)
conditions = df_full['Condition_Name'].unique()

results_summary = []

# 2. 核心寻优循环
for cond_name in conditions:
    print(f"\n[{cond_name}] 寻优中...")
    
    # 加载对应工况的代理模型
    power_model_path = os.path.join(model_dir, f'model_power_{cond_name}.pkl')
    conc_model_path = os.path.join(model_dir, f'model_conc_{cond_name}.pkl')
    
    model_power = joblib.load(power_model_path)
    model_conc = joblib.load(conc_model_path)
    
    # 提取历史数据基准点
    cond_data = df_full[df_full['Condition_Name'] == cond_name]
    mean_temp = cond_data['Temp_C'].mean()
    mean_cin = cond_data['C_in_gNm3'].mean()
    mean_q = cond_data['Q_Nm3h'].mean()
    
    mean_T = [cond_data[f'T{i}_s'].mean() for i in range(1, 5)]
    history_avg_power = cond_data['P_total_kW'].mean()
    u_sum_history_max = cond_data['U1_kV'].max() + cond_data['U2_kV'].max() + cond_data['U3_kV'].max() + cond_data['U4_kV'].max()
    
    # 3. 约束边界设定
    bounds = []
    # 电压边界
    for i in range(1, 5):
        u_mean = cond_data[f'U{i}_kV'].mean()
        u_max = cond_data[f'U{i}_kV'].max()
        bounds.append((u_mean * 0.8, u_max * 1.5)) 
        
    # 周期边界
    for i in range(1, 5):
        t_max = cond_data[f'T{i}_s'].max()
        bounds.append((60, max(200, t_max * 1.2)))

    K_empirical = 1.35 

    # 4. 目标函数构建
    def objective_function(x):
        # 预测电耗
        input_p = pd.DataFrame([{
            'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
            'U1_kV': x[0], 'U2_kV': x[1], 'U3_kV': x[2], 'U4_kV': x[3],
            'T1_s': x[4], 'T2_s': x[5], 'T3_s': x[6], 'T4_s': x[7]
        }])
        pred_power = model_power.predict(input_p)[0]
        
        # 预测基准浓度
        input_c = pd.DataFrame([{
            'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
            'U1_kV': x[0], 'U2_kV': x[1], 'U3_kV': x[2], 'U4_kV': x[3],
            'T1_s_aligned': x[4], 'T2_s_aligned': x[5], 'T3_s_aligned': x[6], 'T4_s_aligned': x[7]
        }])
        base_conc = model_conc.predict(input_c)[0]
        
        # D-A公式机理外推
        u_sum_current = x[0] + x[1] + x[2] + x[3]
        if u_sum_current > u_sum_history_max:
            q_scaled = mean_q / 10000 
            final_conc = (mean_cin * 1000) * np.exp(-K_empirical * u_sum_current / q_scaled)
        else:
            final_conc = base_conc
            
        # 排放约束罚函数 (超排惩罚)
        emission_penalty = 0
        if final_conc > 10.0:
            emission_penalty = 10000 * (final_conc - 10.0)**2 
            
        # 极板积灰动态罚函数
        dust_penalty = 0
        for i in range(4):
            t_current = x[4+i]
            t_safe_limit = mean_T[i] * 1.5 
            if t_current > t_safe_limit:
                dust_penalty += 2.0 * (t_current - t_safe_limit)**2
                
        return pred_power + emission_penalty + dust_penalty

    # 5. DE求解
    result = differential_evolution(
        objective_function, bounds, strategy='best1bin', 
        popsize=20, mutation=(0.5, 1.2), recombination=0.8, seed=42, disp=False
    )

    # 6. 结果后处理
    best_params = result.x
    
    best_input_p = pd.DataFrame([{
        'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
        'U1_kV': best_params[0], 'U2_kV': best_params[1], 'U3_kV': best_params[2], 'U4_kV': best_params[3],
        'T1_s': best_params[4], 'T2_s': best_params[5], 'T3_s': best_params[6], 'T4_s': best_params[7]
    }])
    optimized_power = model_power.predict(best_input_p)[0]
    
    u_sum_final = sum(best_params[0:4])
    if u_sum_final > u_sum_history_max:
        q_scaled_final = mean_q / 10000
        final_conc_display = (mean_cin * 1000) * np.exp(-K_empirical * u_sum_final / q_scaled_final)
    else:
        final_conc_display = (mean_cin * 1000) * np.exp(-K_empirical * u_sum_final / (mean_q / 10000))
        
    saving_rate = ((history_avg_power - optimized_power) / history_avg_power) * 100

    print(f"    - 优化后出口浓度: {final_conc_display:.2f} mg/Nm³")
    print(f"    - 优化前电耗: {history_avg_power:.2f} kW")
    print(f"    - 优化后电耗: {optimized_power:.2f} kW")
    print(f"    - 节能率: {saving_rate:.2f}%")
    
    results_summary.append({
        'Condition_Name': cond_name,
        'U1_kV': round(best_params[0], 2), 'U2_kV': round(best_params[1], 2),
        'U3_kV': round(best_params[2], 2), 'U4_kV': round(best_params[3], 2),
        'T1_s': int(best_params[4]), 'T2_s': int(best_params[5]),
        'T3_s': int(best_params[6]), 'T4_s': int(best_params[7]),
        'C_out_optimized': round(final_conc_display, 2),
        'P_total_optimized': round(optimized_power, 2)
    })

# 7. 汇总输出
print("\n寻优计算完成。各工况控制策略汇总：")
final_df = pd.DataFrame(results_summary)
print(final_df.to_string(index=False))

output_file = os.path.join(current_dir, 'Task2_Final_Optimization_Strategies.csv')
final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n结果已导出至: {output_file}")