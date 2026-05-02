import os
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution
import warnings

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

# 设定工作目录为当前脚本所在路径，确保相对路径读取正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------
# 1. 全局配置与路径设定
# ---------------------------------------------------------
DATA_PATH = os.path.join('..', 'Task2', 'Data_with_Conditions.csv')
MODEL_DIR = os.path.join('..', 'Task2', 'Models')
OUTPUT_CSV = 'question4_results.csv'

# 超参数设定 (严格承接前问，保持罚函数体系闭环)
LAMBDA_E = 10000.0  # 环保超标惩罚系数
LAMBDA_D = 2.0      # 极板积灰惩罚系数
ALPHA = 1.5         # 振打周期安全宽容度
K_EMPIRICAL = 1.35  # Deutsch经验常数

print("=" * 75)
print("Task 4: 排放标准 10 -> 5 mg/Nm³ 的极限减排成本与协同优化控制")
print("=" * 75)

# 加载包含工况划分的完整数据集
df_full = pd.read_csv(DATA_PATH)
results = []

# ---------------------------------------------------------
# 2. 核心寻优模块 (按典型工况循环遍历)
# ---------------------------------------------------------
for cond_name in df_full['Condition_Name'].unique():
    
    # 提取当前工况的数据子集
    cond_data = df_full[df_full['Condition_Name'] == cond_name].copy()
    
    # 计算当前工况的基础物理特征均值与历史极值
    mean_temp = cond_data['Temp_C'].mean()
    mean_cin = cond_data['C_in_gNm3'].mean()
    mean_q = cond_data['Q_Nm3h'].mean()
    mean_T = [cond_data[f'T{i}_s'].mean() for i in range(1, 5)]
    u_sum_history_max = sum(cond_data[f'U{i}_kV'].max() for i in range(1, 5))
    
    # 加载针对该工况训练的高精度机器学习代理模型
    model_power = joblib.load(os.path.join(MODEL_DIR, f'model_power_{cond_name}.pkl'))
    model_conc = joblib.load(os.path.join(MODEL_DIR, f'model_conc_{cond_name}.pkl'))
    
    # ---------------------------------------------------------
    # 【核心创新】基于颗粒物分级捕集机理的非对称寻优边界重构
    # 策略：前级控流（防止无效电耗），后级升压（拔高微细粉尘荷电率）
    # ---------------------------------------------------------
    bounds = []
    
    # E1, E2: 粗颗粒居多，限制电压盲目上升上限 (1.1x ~ 1.2x)
    bounds.append((cond_data['U1_kV'].mean() * 0.8, cond_data['U1_kV'].max() * 1.1))
    bounds.append((cond_data['U2_kV'].mean() * 0.8, cond_data['U2_kV'].max() * 1.2))
    
    # E3, E4: 空间电荷少，专门拦截极难荷电的微细粉尘，大幅放开电压寻优上限 (1.6x ~ 1.8x)
    bounds.append((cond_data['U3_kV'].mean() * 0.8, cond_data['U3_kV'].max() * 1.6))
    bounds.append((cond_data['U4_kV'].mean() * 0.8, cond_data['U4_kV'].max() * 1.8))
    
    # 振打周期边界 (T1~T4): 保持在物理安全和机械允许的合理范围内
    for i in range(1, 5):
        t_max = cond_data[f'T{i}_s'].max()
        bounds.append((60, max(200, t_max * 1.2)))
    
    # ---------------------------------------------------------
    # 3. 数据与机理双驱混合目标函数定义
    # ---------------------------------------------------------
    def objective(x, C_limit):
        # 构造用于预测电耗的输入 DataFrame
        input_p = pd.DataFrame([{
            'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
            'U1_kV': x[0], 'U2_kV': x[1], 'U3_kV': x[2], 'U4_kV': x[3],
            'T1_s': x[4], 'T2_s': x[5], 'T3_s': x[6], 'T4_s': x[7]
        }])
        
        # 构造用于预测浓度的输入 DataFrame
        input_c = pd.DataFrame([{
            'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
            'U1_kV': x[0], 'U2_kV': x[1], 'U3_kV': x[2], 'U4_kV': x[3],
            'T1_s_aligned': x[4], 'T2_s_aligned': x[5],
            'T3_s_aligned': x[6], 'T4_s_aligned': x[7]
        }])
        
        # 模型推理：预测基础电耗与基础出口浓度
        pred_power = model_power.predict(input_p)[0]
        base_conc = model_conc.predict(input_c)[0]
        
        # 【核心创新】数据与机理的平滑切换外推机制
        u_sum_current = sum(x[0:4])
        if u_sum_current > u_sum_history_max:
            # 当寻优突破历史电压极限时，AI外推可能失效，自适应切换至 Deutsch 物理公式兜底
            q_scaled = mean_q / 10000.0
            final_conc = mean_cin * 1000 * np.exp(-K_EMPIRICAL * u_sum_current / q_scaled)
        else:
            # 在历史数据置信域内，完全信任机器学习模型的高精度预测
            final_conc = base_conc
        
        # 惩罚项 1：环保超标平方惩罚
        emission_penalty = LAMBDA_E * max(0, final_conc - C_limit)**2
        
        # 惩罚项 2：极板过度积灰平方惩罚 (保障设备长期健康运行)
        dust_penalty = 0
        for i in range(4):
            t_current = x[4+i]
            t_safe_limit = mean_T[i] * ALPHA
            if t_current > t_safe_limit:
                dust_penalty += LAMBDA_D * (t_current - t_safe_limit)**2
        
        # 返回综合目标函数值 (成本化模型)
        return pred_power + emission_penalty + dust_penalty

    print(f"\n[{cond_name}] 正在进行 DE 全局寻优计算，请稍候...")
    
    # ---------------------------------------------------------
    # 4. 差分进化 (DE) 算法全局寻优计算
    # ---------------------------------------------------------
    # 场景 A：寻优达到 5 mg/Nm³ 超低排放标准
    result_5 = differential_evolution(
        lambda x: objective(x, 5.0), bounds,
        strategy='best1bin', popsize=20, mutation=(0.5, 1.2),
        recombination=0.8, seed=42, disp=False
    )
    best_5 = result_5.x
    
    # 根据最优参数，通过模型还原达标 5mg 的实际预测电耗
    P_5 = model_power.predict(pd.DataFrame([{
        'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
        'U1_kV': best_5[0], 'U2_kV': best_5[1], 'U3_kV': best_5[2], 'U4_kV': best_5[3],
        'T1_s': best_5[4], 'T2_s': best_5[5], 'T3_s': best_5[6], 'T4_s': best_5[7]
    }]))[0]
    
    # 场景 B：寻优达到 10 mg/Nm³ 常规排放标准
    result_10 = differential_evolution(
        lambda x: objective(x, 10.0), bounds,
        strategy='best1bin', popsize=20, mutation=(0.5, 1.2),
        recombination=0.8, seed=42, disp=False
    )
    best_10 = result_10.x
    
    # 根据最优参数，通过模型还原达标 10mg 的实际预测电耗
    P_10 = model_power.predict(pd.DataFrame([{
        'Temp_C': mean_temp, 'C_in_gNm3': mean_cin, 'Q_Nm3h': mean_q,
        'U1_kV': best_10[0], 'U2_kV': best_10[1], 'U3_kV': best_10[2], 'U4_kV': best_10[3],
        'T1_s': best_10[4], 'T2_s': best_10[5], 'T3_s': best_10[6], 'T4_s': best_10[7]
    }]))[0]
    
    # 计算边际减排带来的电耗相对增幅 (%)
    increase_pct = (P_5 / P_10 - 1.0) * 100.0
    
    # 将结果保存至汇总列表
    results.append({
        '工况': cond_name,
        '入口浓度(g/Nm³)': round(mean_cin, 2),
        '达标10电耗(kW)': round(P_10, 2),
        '达标5电耗(kW)': round(P_5, 2),
        '电耗增幅(%)': round(increase_pct, 2)
    })
    
    # 控制台实时监控输出 (统一保留2位小数以显示微小差异)
    print(f"  √ 达标 10mg 电耗: {P_10:.2f} kW  |  达标 5mg 电耗: {P_5:.2f} kW  |  预期增幅: {increase_pct:.2f}%")

# ---------------------------------------------------------
# 5. 格式化输出与结果保存
# ---------------------------------------------------------
df_results = pd.DataFrame(results)

print(f"\n{'='*75}")
# 使用全角与半角空格混合运算，确保中英文表头在终端下绝对对齐
header = "工况类型" + "　"*4 + "   入口浓度   达标10电耗    达标5电耗   电耗增幅(%)"
print(header)
print("-" * 75)

for _, row in df_results.iterrows():
    cond_name = row['工况']
    # 动态计算所需的中文全角空格数量，保证工况名对齐
    pad_len = 8 - (len(cond_name.encode('gbk')) // 2)
    formatted_cond = cond_name + '\u3000' * pad_len
    
    # 精准控制格式化占位符，使数据与表头对齐
    print(f"{formatted_cond} {row['入口浓度(g/Nm³)']:>10.2f} {row['达标10电耗(kW)']:>12.2f} {row['达标5电耗(kW)']:>12.2f} {row['电耗增幅(%)']:>13.2f}")

# 将结果保存为 CSV 文件，支持 Excel 打开且无中文乱码
df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

# ---------------------------------------------------------
# 6. 自动预警分析报告 (按电耗增幅痛点触发)
# ---------------------------------------------------------
# 自动锁定最高风险工况（即实现超低排放时代价最大的瓶颈工况）
high_risk_cond = df_results.sort_values('电耗增幅(%)', ascending=False).iloc[0]

print(f"\n{'='*75}")
print(">> 高风险工况极限减排预警分析报告 <<")
print(f"【触发工况】: {high_risk_cond['工况']} (电耗增幅达全量极值 {high_risk_cond['电耗增幅(%)']:.2f}%)")
print(f"【能耗代价】: 达标 10mg 电耗为 {high_risk_cond['达标10电耗(kW)']:.2f} kW。若强制要求达标 5mg，")
print(f"             在非对称供电策略优化后，系统最低需 {high_risk_cond['达标5电耗(kW)']:.2f} kW，")
print(f"             能耗边际成本急剧恶化。")
print("【应对建议】: 高温引发的高比电阻与反电晕效应是能耗激增的核心诱因。建议：")
print("             1. 在前端部署烟气调质装置（如喷水降温增湿），从物理根源上降低粉尘比电阻；")
print("             2. 对末级电场进行脉冲电源改造，利用高频脉冲突破亚微米粉尘的荷电瓶颈。")
print(f"{'='*75}\n")