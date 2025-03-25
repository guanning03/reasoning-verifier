import matplotlib.pyplot as plt

data = [0.384, 0.551]

# 设置更大的字体大小
plt.rcParams.update({'font.size': 14})  # 增加全局字体大小

plt.bar(['Qwen2-0.5B-Instruct', 'Qwen2-0.5B-Instruct-GRPO'], data)
plt.ylim(0, 0.6)

# 添加百分比标签（更大字号）
for i, v in enumerate(data):
    plt.text(i, v, f'{v:.1%}', ha='center', va='bottom', fontsize=14)

# 添加纵轴标题（更大字号）
plt.ylabel('GSM8k Pass Rate', fontsize=14)

# 设置刻度标签的字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig('plots/plot1.png')