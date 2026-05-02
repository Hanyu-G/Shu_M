import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. 相对路径加载数据
DATA_PATH = os.path.join('..', 'Task2', 'Data_with_Conditions.csv')
OPT_PATH = os.path.join('..', 'Task2', 'Task2_Final_Optimization_Strategies.csv')

df_full = pd.read_csv(DATA_PATH)
df_opt = pd.read_csv(OPT_PATH)

# 2. 超参数设定
LAMBDA_E = 10000.0  # 环保超标惩罚系数
LAMBDA_D = 2.0      # 极板积灰惩罚系数
ALPHA = 1.5         # 振打周期安全宽容度
K_EMPIRICAL = 1.35  # 经验常数

print("=" * 75)
print("Task 3: 基于目标函数边际扰动的参数优先级量化模型")
print("=" * 75)

results = []

# 3. 核心计算模块
for cond_name in df_opt['Condition_Name'].unique():
    
    # 提取当前工况数据
    opt = df_opt[df_opt['Condition_Name'] == cond_name].iloc[0]
    data = df_full[df_full['Condition_Name'] == cond_name]
    
    Cin = data['C_in_gNm3'].mean()
    Q_scaled = data['Q_Nm3h'].mean() / 10000.0
    
    U_opt = np.array([opt[f'U{i}_kV'] for i in range(1, 5)])
    T_opt = np.array([opt[f'T{i}_s'] for i in range(1, 5)])
    T_mean = np.array([data[f'T{i}_s'].mean() for i in range(1, 5)])
    
    Cout_base = opt['C_out_optimized']
    P_base = opt['P_total_optimized']
    
    # 计算基准目标函数 (Obj_base)
    E_base = LAMBDA_E * max(0, Cout_base - 10)**2
    D_base = LAMBDA_D * sum(max(0, T_opt[i] - ALPHA * T_mean[i])**2 for i in range(4))
    Obj_base = P_base + E_base + D_base
    
    # ---------------------------------------------------------
    # 敏感度分析 I: 电压边际扰动 (M_U)
    # ---------------------------------------------------------
    delta_U = []
    for i in range(4):
        for d in [-0.05, 0.05]:
            U_new = U_opt.copy()
            U_new[i] *= (1 + d)
            
            # Deutsch公式计算新浓度
            C_new = Cin * 1000 * np.exp(-K_EMPIRICAL * np.sum(U_new) / Q_scaled)
            # 等效电阻比例计算新电耗
            P_new = np.sum((U_new**2 / U_opt**2) * (P_base * (U_opt**2 / np.sum(U_opt**2))))
            
            E_new = LAMBDA_E * max(0, C_new - 10)**2
            Obj_new = P_new + E_new + D_base
            # 仅评估目标函数恶化的风险
            delta_U.append(max(0, (Obj_new - Obj_base)) / Obj_base * 100)
    M_U = np.mean(delta_U)
    
    # ---------------------------------------------------------
    # 敏感度分析 II: 振打周期边际扰动 (M_T)
    # ---------------------------------------------------------
    delta_T = []
    for i in range(4):
        for d in [-0.10, 0.10]:
            T_new = T_opt.copy()
            T_new[i] = max(10, T_new[i] * (1 + d))
            
            D_new = LAMBDA_D * sum(max(0, T_new[j] - ALPHA * T_mean[j])**2 for j in range(4))
            Obj_new = P_base + E_base + D_new
            # 评估目标函数变动绝对比例
            delta_T.append(abs(Obj_new - Obj_base) / Obj_base * 100)
    M_T = np.mean(delta_T)
    
    # ---------------------------------------------------------
    # 优先级判定与结果存储
    # ---------------------------------------------------------
    diff = M_U - M_T
    if abs(diff) < 0.2 * max(M_U, M_T) or (M_U == 0 and M_T == 0):
        priority = "协同调节"
    elif diff > 0:
        priority = "电压优先"
    else:
        priority = "振打优先"
    
    results.append({
        'Condition': cond_name,
        'C_in': round(Cin, 2),
        'C_out': round(Cout_base, 2),
        'P_opt': round(P_base, 2),
        'M_U': round(M_U, 2),
        'M_T': round(M_T, 2),
        'Priority': priority
    })

# 4. 终端格式化输出与保存
df_results = pd.DataFrame(results)

print(f"\n{'='*75}")
header = f"{'工况':<14} {'C_in':>6} {'C_out':>6} {'P_opt':>8} {'M_U(%)':>8} {'M_T(%)':>6}  {'Priority':<8}"
print(header)
print("-" * 75)

for _, row in df_results.iterrows():
    cond_name = row['Condition']
    pad_len = 8 - (len(cond_name.encode('gbk')) // 2) 
    formatted_cond = cond_name + '\u3000' * pad_len
    
    print(f"{formatted_cond} {row['C_in']:>6.2f} {row['C_out']:>6.2f} {row['P_opt']:>8.2f} {row['M_U']:>8.2f} {row['M_T']:>6.2f}  {row['Priority']:<8}")

# 重命名列以满足中文输出需求
df_results.rename(columns={'Condition': '工况', 'Priority': 'priority'}, inplace=True)
df_results.to_csv('priority_results.csv', index=False, encoding='utf-8-sig')
print(f"{'='*75}\nResult saved to priority_results.csv")