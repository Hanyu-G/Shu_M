import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# 配置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

# ---------------------------------------------------------
# 1. 路径配置与数据加载
# ---------------------------------------------------------
print("加载数据...")
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'Cleaned_Cement_ESP_Data.csv')

if not os.path.exists(data_path):
    print(f"Error: 未找到数据文件。请确认路径: {data_path}")
    exit()

df = pd.read_csv(data_path)

# ---------------------------------------------------------
# 2. 特征选择 (采用平滑后数据)
# ---------------------------------------------------------
features = ['Temp_C_smooth', 'C_in_gNm3_smooth', 'Q_Nm3h_smooth', 
            'U1_kV', 'U2_kV', 'U3_kV', 'U4_kV', 
            'T1_s', 'T2_s', 'T3_s', 'T4_s']
target = 'C_out_mgNm3_smooth'

# ---------------------------------------------------------
# 3. 相关性分析与热力图绘制 (Pearson & Spearman)
# ---------------------------------------------------------
print("计算相关性矩阵并绘制热力图...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Pearson (线性相关)
corr_pearson = df[features + [target]].corr(method='pearson')
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
axes[0].set_title('Pearson 相关性矩阵 (基于平滑消噪数据)', fontsize=16)

# Spearman (单调非线性相关)
corr_spearman = df[features + [target]].corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
axes[1].set_title('Spearman 秩相关矩阵 (基于平滑消噪数据)', fontsize=16)

plt.tight_layout()

plot_path1 = os.path.join(current_dir, 'Correlation_Heatmaps.png')
plt.savefig(plot_path1, dpi=300) 
plt.show()

# ---------------------------------------------------------
# 4. 随机森林特征重要性评估
# ---------------------------------------------------------
print("训练随机森林模型...")
X = df[features]
y = df[target]

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# 提取特征重要性并排序
importances = rf.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# 绘制特征重要性柱状图
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('随机森林特征重要性排序 (多维耦合下对出口浓度的贡献度)', fontsize=16)
plt.xlabel('重要性权重')
plt.ylabel('操作/工况特征')
plt.tight_layout()

plot_path2 = os.path.join(current_dir, 'Feature_Importance_RF.png')
plt.savefig(plot_path2, dpi=300)
plt.show()

# ---------------------------------------------------------
# 5. 运行结束
# ---------------------------------------------------------
print("运行完成。")
print(f"热力图保存至: {plot_path1}")
print(f"重要性图保存至: {plot_path2}")