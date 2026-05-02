import os
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

warnings.filterwarnings('ignore')

# 1. 路径配置与数据加载
print("开始执行分工况代理模型批量训练...")

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'Data_with_Conditions.csv')
model_save_dir = os.path.join(current_dir, 'Models')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"未找到数据文件: {data_path}")

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

df_full = pd.read_csv(data_path)
conditions = df_full['Condition_Name'].unique()

print(f"共检测到 {len(conditions)} 种典型工况。\n")

delay_dict = {'T1_s': 9, 'T2_s': 5, 'T3_s': 0, 'T4_s': 1}

features_power = ['Temp_C', 'C_in_gNm3', 'Q_Nm3h', 'U1_kV', 'U2_kV', 'U3_kV', 'U4_kV', 'T1_s', 'T2_s', 'T3_s', 'T4_s']
features_conc = ['Temp_C', 'C_in_gNm3', 'Q_Nm3h', 'U1_kV', 'U2_kV', 'U3_kV', 'U4_kV', 'T1_s_aligned', 'T2_s_aligned', 'T3_s_aligned', 'T4_s_aligned']

# 2. 分工况批量训练
for cond_name in conditions:
    print(f"[{cond_name}] 模型训练中...")
    
    df = df_full[df_full['Condition_Name'] == cond_name].copy()
    
    # 特征时序对齐
    for col, lag in delay_dict.items():
        if lag > 0:
            df[f'{col}_aligned'] = df[col].shift(lag)
        else:
            df[f'{col}_aligned'] = df[col]

    df = df.dropna().reset_index(drop=True)
    
    # 2.1 训练总电耗回归模型
    X_power = df[features_power]
    y_power = df['P_total_kW']

    X_train_p, X_test_p, yp_train, yp_test = train_test_split(X_power, y_power, test_size=0.2, random_state=42)

    model_power = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
    model_power.fit(X_train_p, yp_train)

    power_r2 = r2_score(yp_test, model_power.predict(X_test_p))
    print(f"  - [电耗模型] R²: {power_r2:.4f}")

    # 2.2 训练出口浓度回归模型
    X_conc = df[features_conc]
    y_conc = df['C_out_mgNm3']

    X_train_c, X_test_c, yc_train, yc_test = train_test_split(X_conc, y_conc, test_size=0.2, random_state=42)

    # 模型参数设定，避免在低方差数据上过拟合
    model_conc = lgb.LGBMRegressor(
        n_estimators=100,      
        learning_rate=0.01,    
        objective='mse',       
        max_depth=4,           
        random_state=42,
        verbose=-1
    )

    model_conc.fit(X_train_c, yc_train)
    yc_pred = model_conc.predict(X_test_c)
    
    # 使用 RMSE 和 MAE 评估模型误差
    rmse = np.sqrt(mean_squared_error(yc_test, yc_pred))
    mae = mean_absolute_error(yc_test, yc_pred)

    print(f"  - [浓度模型] RMSE: {rmse:.4f} mg/Nm³")
    print(f"  - [浓度模型] MAE: {mae:.4f} mg/Nm³")
    
    # 2.3 模型持久化保存
    power_model_filename = f"model_power_{cond_name}.pkl"
    conc_model_filename = f"model_conc_{cond_name}.pkl"
    
    joblib.dump(model_power, os.path.join(model_save_dir, power_model_filename))
    joblib.dump(model_conc, os.path.join(model_save_dir, conc_model_filename))
    
print("\n所有工况代理模型训练完毕。")