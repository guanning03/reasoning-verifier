import json
import matplotlib.pyplot as plt
import numpy as np

with open("evaluations/outputs_MATH-500_Qwen2.5-Math-1.5B_Qwen_MATH_0shot_0.6_4096.json", "r") as f:
    data = json.load(f)

correct_list = []
for item in data:
    correct_list.append(item['accuracy'].count(True))

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建直方图
plt.figure(figsize=(6, 4))
bins = np.arange(0, 65, 8)  # 创建bin边界：0, 8, 16, ..., 64
plt.hist(correct_list, bins=bins, edgecolor='black', color='pink', label=f'Avg_Pass = 43.7%')

# 添加图例
plt.legend()

# 创建区间标签
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_labels = [f'[{int(bins[i])}-{int(bins[i+1])}]' for i in range(len(bins)-1)]

# 设置x轴刻度和标签
plt.xticks(bin_centers, bin_labels, rotation=45)

# 设置标签和标题
plt.xlabel('Correct Response in 64 Responses', labelpad=10, fontsize=12)
plt.ylabel('Number of Questions', labelpad=10, fontsize=12)
plt.title('Qwen2.5-Math-1.5B / MATH-500 Zero-Shot', fontsize=14)

# 显示网格
plt.grid(True, alpha=0.3)

# 调整布局，确保所有元素都在图内
plt.tight_layout()

# 保存图片，设置较高的DPI值以提高清晰度
plt.savefig('correct_number_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

