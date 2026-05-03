import os
import numpy as np
import pandas as pd
import joblib
import warnings

# 忽略环境与版本警告
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------
# 1. 路径配置与数据加载
# ---------------------------------------------------------
DATA_PATH = os.path.join('..', 'Task2', 'Data_with_Conditions.csv')
OPT_PATH = os.path.join('..', 'Task2', 'Task2_Final_Optimization_Strategies.csv')
MODEL_DIR = os.path.join('..', 'Task2', 'Models') 

df_full = pd.read_csv(DATA_PATH)
df_opt = pd.read_csv(OPT_PATH)

# ---------------------------------------------------------
# 2. 超参数与惩罚系数设定 (保持全局目标函数一致性)
# ---------------------------------------------------------
LAMBDA_E = 10000.0  # 环保超标惩罚系数
LAMBDA_D = 2.0      # 极板积灰惩罚系数
ALPHA = 1.5         # 振打周期宽容度阈值
K_EMPIRICAL = 1.35  # Deutsch经验常数

print("=" * 75)
print("Task 3: 基于混合驱动模型的参数优先级局部敏感度分析")
print("=" * 75)

results = []

# ---------------------------------------------------------
# 3. 核心计算模块：遍历典型工况进行敏感度计算
# ---------------------------------------------------------
for cond_name in df_opt['Condition_Name'].unique():
    
    # 提取当前工况的历史数据与最优解
    opt = df_opt[df_opt['Condition_Name'] == cond_name].iloc[0]
    data = df_full[df_full['Condition_Name'] == cond_name]
    
    mean_temp = data['Temp_C'].mean()
    mean_cin = data['C_in_gNm3'].mean()
    mean_q = data['Q_Nm3h'].mean()
    
    # 获取历史电压极值，作为双驱模型切换阈值
    u_sum_history_max = sum(data[f'U{i}_kV'].max() for i in range(1, 5))
    
    U_opt = np.array([opt[f'U{i}_kV'] for i in range(1, 5)])
    T_opt = np.array([opt[f'T{i}_s'] for i in range(1, 5)])
    T_mean = np.array([data[f'T{i}_s'].mean() for i in range(1, 5)])
    
    Cout_base = opt['C_out_optimized']
    P_base = opt['P_total_optimized']
    
    # 计算无扰动状态下的基准目标函数值
    E_base = LAMBDA_E * max(0, Cout_base - 10)**2
    D_base = LAMBDA_D * sum(max(0, T_opt[i] - ALPHA * T_mean[i])**2 for i in range(4))
    Obj_base = P_base + E_base + D_base

    # 动态加载对应工况的 LightGBM 代理模型
    try:
        model_power = joblib.load(os.path.join(MODEL_DIR, f'model_power_{cond_name}.pkl'))
        model_conc = joblib.load(os.path.join(MODEL_DIR, f'model_conc_{cond_name}.pkl'))
    except FileNotFoundError:
        print(f"警告: 找不到工况 '{cond_name}' 的模型文件，跳过该工况分析。")
        continue
    
    # ---------------------------------------------------------
    # 3.1 电压控制域敏感度分析 (M_U)
    # ---------------------------------------------------------
    delta_U = []
    for i in range(4):
        # 施加 ±5% 的双向数值扰动
        for d in [-0.05, 0.05]:
            U_new = U_opt.copy()
            U_new[i] *= (1 + d)
            
            # 构造模型预测输入
            input_p = pd.DataFrame([{
                'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
                'U1_kV': U_new[0], 'U2_kV': U_new[1], 'U3_kV': U_new[2], 'U4_kV': U_new[3],
                'T1_s': T_opt[0], 'T2_s': T_opt[1], 'T3_s': T_opt[2], 'T4_s': T_opt[3]
            }])
            P_new = model_power.predict(input_p)[0]
            
            # 执行数据与机理的平滑切换逻辑
            u_sum_current = np.sum(U_new)
            if u_sum_current > u_sum_history_max:
                q_scaled = mean_q / 10000.0
                C_new = mean_cin * 1000 * np.exp(-K_EMPIRICAL * u_sum_current / q_scaled)
            else:
                input_c = pd.DataFrame([{
                    'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
                    'U1_kV': U_new[0], 'U2_kV': U_new[1], 'U3_kV': U_new[2], 'U4_kV': U_new[3],
                    'T1_s_aligned': T_opt[0], 'T2_s_aligned': T_opt[1], 'T3_s_aligned': T_opt[2], 'T4_s_aligned': T_opt[3]
                }])
                C_new = model_conc.predict(input_c)[0]
            
            # 计算扰动后的综合目标函数偏差
            E_new = LAMBDA_E * max(0, C_new - 10)**2
            Obj_new = P_new + E_new + D_base
            delta_U.append(max(0, (Obj_new - Obj_base)) / Obj_base * 100)
            
    M_U = np.mean(delta_U)
    
    # ---------------------------------------------------------
    # 3.2 振打周期控制域敏感度分析 (M_T)
    # ---------------------------------------------------------
    delta_T = []
    for i in range(4):
        # 施加 ±10% 的双向数值扰动
        for d in [-0.10, 0.10]:
            T_new = T_opt.copy()
            T_new[i] = max(10, T_new[i] * (1 + d))
            
            # 独立评估振打周期对极板积灰惩罚的边际影响
            D_new = LAMBDA_D * sum(max(0, T_new[j] - ALPHA * T_mean[j])**2 for j in range(4))
            Obj_new = P_base + E_base + D_new
            delta_T.append(abs(Obj_new - Obj_base) / Obj_base * 100)
            
    M_T = np.mean(delta_T)
    
    # ---------------------------------------------------------
    # 3.3 死区判定与控制优先级输出
    # ---------------------------------------------------------
    diff = M_U - M_T
    # 设定 20% 的优先级死区阈值，防止高频震荡调节
    if abs(diff) < 0.2 * max(M_U, M_T) or (M_U == 0 and M_T == 0):
        priority = "协同调节"
    elif diff > 0:
        priority = "电压优先"
    else:
        priority = "振打优先"
    
    results.append({
        'Condition': cond_name,
        'C_in': round(mean_cin, 2),
        'C_out': round(Cout_base, 2),
        'P_opt': round(P_base, 2),
        'M_U': round(M_U, 2),
        'M_T': round(M_T, 2),
        'Priority': priority
    })

# ---------------------------------------------------------
# 4. 格式化输出与本地数据化持久
# ---------------------------------------------------------
df_results = pd.DataFrame(results)

print(f"\n{'='*75}")
header = f"{'工况':<14} {'C_in':>6} {'C_out':>6} {'P_opt':>8} {'M_U(%)':>8} {'M_T(%)':>6}  {'Priority':<8}"
print(header)
print("-" * 75)

for _, row in df_results.iterrows():
    cond_name = row['Condition']
    # 计算全角空格补齐量以适配终端排版
    pad_len = 8 - (len(cond_name.encode('gbk')) // 2) 
    formatted_cond = cond_name + '\u3000' * pad_len
    
    print(f"{formatted_cond} {row['C_in']:>6.2f} {row['C_out']:>6.2f} {row['P_opt']:>8.2f} {row['M_U']:>8.2f} {row['M_T']:>6.2f}  {row['Priority']:<8}")

df_results.rename(columns={'Condition': '工况', 'Priority': 'priority'}, inplace=True)
df_results.to_csv('priority_results.csv', index=False, encoding='utf-8-sig')
print(f"{'='*75}\n分析结果已导出至: priority_results.csv")