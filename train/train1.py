import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['SimHei']

# 1. 加载并预处理数据
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 数据预处理（网页3、5、7）
# 归一化像素值到0-1范围并调整维度形状
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 2. 构建CNN模型
model = models.Sequential([
    # 第一卷积层：32个3x3卷积核
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # 第二卷积层：64个3x3卷积核
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 展平后连接全连接层
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 3. 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 训练模型
history = model.fit(train_images, train_labels,
                    epochs=5,
                    validation_split=0.1)  # 使用10%数据作为验证集

# 5. 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\n测试准确率: {test_acc:.4f}")


def plot_training_history(history):
    """绘制训练和验证准确率、损失曲线"""
    plt.figure(figsize=(12, 4))

    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.legend()

    plt.tight_layout()
    plt.savefig('basic_training_history.png')
    plt.show()


# 可视化训练过程
plot_training_history(history)

model.save('model.h5')
