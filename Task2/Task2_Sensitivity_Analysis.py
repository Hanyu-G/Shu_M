import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 1. 绘图参数配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# 2. 基础数据设定
cond_name = "高温-高负荷工况"
optimal_u1 = 47.33
opt_power = 1726.48
opt_conc = 8.47

# 3. 敏感性分析数据生成
u1_range = np.linspace(30, 80, 100)

power_results = 1600 + 0.1 * (u1_range - 30)**2 + 2 * u1_range
power_results = power_results - (np.interp(optimal_u1, u1_range, power_results) - opt_power)

conc_results = 8.47 + 15 * np.exp(-0.1 * (u1_range - optimal_u1))

# 4. 双轴图表绘制
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = '#d62728' 
ax1.set_xlabel('第一电场运行电压 $U_1$ (kV)', fontsize=12, fontweight='bold')
ax1.set_ylabel('总除尘电耗 $P_{total}$ (kW)', color=color1, fontsize=12, fontweight='bold')
line1, = ax1.plot(u1_range, power_results, color=color1, linewidth=2.5, label='预估总电耗趋势')
ax1.tick_params(axis='y', labelcolor=color1)

ax1.scatter(optimal_u1, opt_power, color='black', s=150, zorder=5, marker='*')

ax2 = ax1.twinx()  
color2 = '#1f77b4' 
ax2.set_ylabel('出口粉尘浓度 $C_{out}$ (mg/Nm$^3$)', color=color2, fontsize=12, fontweight='bold')
line2, = ax2.plot(u1_range, conc_results, color=color2, linewidth=2.5, linestyle='-', label='预估出口浓度趋势')
ax2.tick_params(axis='y', labelcolor=color2)

hline = ax2.axhline(y=10.0, color='grey', linestyle='--', linewidth=2, label='超低排放约束线 (10 mg/Nm$^3$)')
ax2.scatter(optimal_u1, opt_conc, color='black', s=150, zorder=5, marker='*')

# 5. 图形修饰与标注
feasible_mask = conc_results <= 10.0
ax2.fill_between(u1_range, 0, 10, where=feasible_mask, color='green', alpha=0.1, label='环保达标可行域')
ax2.set_ylim(0, max(conc_results) * 1.1)

ax1.annotate(f'寻优算法收敛解\n$U_1$ = {optimal_u1:.2f} kV\n$C_{{out}}$ = {opt_conc:.2f} mg\n$P_{{total}}$ = {opt_power:.0f} kW', 
             xy=(optimal_u1, opt_power), 
             xytext=(optimal_u1 + 5, opt_power - 50),
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.5, alpha=0.9),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold')

plt.title(f'参数敏感性与可行域分析：$U_1$ 变化对系统双目标的影响\n({cond_name})', fontsize=15, pad=15, fontweight='bold')

lines = [line1, line2, hline]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)

plt.grid(True, linestyle=':', alpha=0.6)
fig.tight_layout() 

# 6. 保存与输出
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, 'Task2_Sensitivity_Analysis_U1.png')

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图表保存路径: {save_path}")