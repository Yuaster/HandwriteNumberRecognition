import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model
import matplotlib.pyplot as plt
import numpy as np


# 定义SE注意力模块为自定义层
class SEBlock(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.gap(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.reshape(x, [-1, 1, 1, inputs.shape[-1]])
        return inputs * x

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


# 构建带注意力机制的CNN模型
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        SEBlock(reduction_ratio=16),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        SEBlock(reduction_ratio=16),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 加载并预处理数据
def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # 数据预处理
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    return (train_images, train_labels), (test_images, test_labels)


# 可视化训练历史
def visualize_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.legend()
    plt.title('模型准确率')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.legend()
    plt.title('模型损失')
    plt.tight_layout()
    plt.savefig('training_history.png')  # 保存图像
    plt.show()


# 主函数
def main():
    print("开始加载数据...")
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

    print("构建模型...")
    model = build_model()
    model.summary()

    print("开始训练模型...")
    history = model.fit(train_images, train_labels,
                        epochs=5,
                        validation_split=0.1)

    print("评估模型...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\n测试准确率: {test_acc:.4f}")

    print("保存模型...")
    model.save('mnist_attention_model.h5')

    print("可视化训练历史...")
    visualize_training_history(history)

    print("训练完成!")


if __name__ == "__main__":
    main()