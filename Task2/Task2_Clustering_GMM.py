import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# 1. 环境配置与中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("开始执行基于 GMM 的典型工况聚类分析...")

# 2. 路径配置与数据加载
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'Task1', 'Cleaned_Cement_ESP_Data.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"未找到预处理数据文件: {data_path}")

df = pd.read_csv(data_path)

# 3. 特征提取与标准化
condition_features = ['Temp_C', 'C_in_gNm3', 'Q_Nm3h']
X_cond = df[condition_features]

scaler = StandardScaler()
X_cond_scaled = scaler.fit_transform(X_cond)

# 4. 轮廓系数评估
print("计算 Silhouette 轮廓系数以评估最佳聚类数...")
best_k = 4  
for k in range(2, 6):
    temp_gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    temp_labels = temp_gmm.fit_predict(X_cond_scaled)
    score = silhouette_score(X_cond_scaled, temp_labels, sample_size=3000, random_state=42)
    print(f"  -> K={k}, Silhouette Score: {score:.4f}")
    
print(f"设定最终聚类数 n_components = {best_k}\n")

# 5. GMM 聚类
gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42)
df['Condition_Label'] = gmm.fit_predict(X_cond_scaled)

# 6. 聚类中心统计与物理特征解析
cluster_centers = df.groupby('Condition_Label')[condition_features].mean().round(2)
cluster_counts = df['Condition_Label'].value_counts().rename('样本量(分钟)')
cluster_summary = pd.concat([cluster_centers, cluster_counts], axis=1)

global_temp_mean = cluster_centers['Temp_C'].mean()
global_cin_mean = cluster_centers['C_in_gNm3'].mean()

def label_condition_dual(row):
    temp_status = "高温" if row['Temp_C'] > global_temp_mean else "低温"
    cin_status = "高负荷" if row['C_in_gNm3'] > global_cin_mean else "低负荷"
    return f"{temp_status}-{cin_status}工况"

cluster_summary['工况物理含义'] = cluster_summary.apply(label_condition_dual, axis=1)
cluster_summary.index.name = '工况编号'

print("=== 典型工况聚类中心特征统计 ===")
print(cluster_summary.to_string())

df['Condition_Name'] = df['Condition_Label'].map(cluster_summary['工况物理含义'])

output_path = os.path.join(current_dir, 'Data_with_Conditions.csv')
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n聚类数据已导出至: {output_path}")

# 7. 聚类结果可视化
plt.figure(figsize=(12, 7))

sns.scatterplot(x='Temp_C', y='C_in_gNm3', hue='Condition_Name', 
                palette='Set1', data=df, s=25, alpha=0.6)

centers = scaler.inverse_transform(gmm.means_)
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='*', s=300, label='聚类中心', zorder=5)

plt.title('典型工况聚类分布图 (烟气温度 vs 入口浓度)', fontsize=16)
plt.xlabel(r'烟气入口温度 $Temp\_C$ (℃)', fontsize=12)
plt.ylabel(r'入口粉尘浓度 $C_{in}$ (g/Nm$^3$)', fontsize=12)
plt.legend(title='工况类型', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

img_path = os.path.join(current_dir, 'GMM_Clustering_Temp_vs_Cin.png')
plt.savefig(img_path, dpi=300)

print(f"聚类联合分布图已保存至: {img_path}")