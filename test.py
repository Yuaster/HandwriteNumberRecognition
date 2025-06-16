import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimHei']

# 模型名称
model_names = ["LeNet-5", "AlexNet(适配后)", "VGG - like(简化版)",
               "ResNet(浅层适配)", "MobileNet(精简版)", "Transformer(简化版)",
               "集成模型", "自编模型"]
# 准确率数据
accuracies_percent = [99.20, 99.50, 99.65,
                      99.70, 99.60, 99.80,
                      99.90, 99.47]

# 按准确率从小到大排序
combined = list(zip(accuracies_percent, model_names))
combined.sort(key=lambda x: x[0])  # 按准确率排序
accuracies_percent_sorted = [x[0] for x in combined]
model_names_sorted = [x[1] for x in combined]

# 创建画布
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制柱状图，为自编模型使用独特颜色
colors = ['#1f77b4'] * len(model_names_sorted)
try:
    self_model_index = model_names_sorted.index("自编模型")
    colors[self_model_index] = '#bcbd22'  # 自编模型使用黄色
except ValueError:
    pass

bars = ax.bar(model_names_sorted, accuracies_percent_sorted, color=colors)

# 在柱子上标注准确率数值
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.2f}%',
            ha='center', va='bottom', fontsize=10)

# 设置标题和坐标轴标签
ax.set_title('知名模型在MNIST数据集上的测试准确率', fontsize=14)
ax.set_xlabel('模型名称', fontsize=12)
ax.set_ylabel('准确率(%)', fontsize=12)

# 设置纵轴范围
ax.set_ylim([99, 100])

# 让x轴标签旋转一定角度
plt.xticks(rotation=45, ha='right')

# 为自编模型添加星标标记
try:
    self_model_index = model_names_sorted.index("自编模型")
    ax.text(self_model_index, 99.05, '*', fontsize=16, ha='center', color='red')
    ax.plot([], [], ' ', label='* 本次实验模型')
    ax.legend(loc='lower right')
except ValueError:
    pass

# 优化布局
plt.tight_layout()

# 显示图形
plt.show()