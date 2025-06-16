import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input

plt.rcParams['font.family'] = ['SimHei']

# 1. 加载和预处理数据
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


def preprocess(images):
    """预处理图像数据：扩展维度并归一化"""
    images = np.expand_dims(images, -1).astype("float32") / 255.0
    return images


# 预处理数据
train_images = preprocess(train_images)
test_images = preprocess(test_images)


# 2. 创建增强模型 - 移除 RandomShear，调整其他参数
def create_augmentation_model():
    """创建数据增强模型，包含多种随机变换"""
    model = models.Sequential([
        layers.RandomRotation(factor=0.1, fill_mode='constant', input_shape=(28, 28, 1)),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant'),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode='constant'),
    ], name='augmentation')
    return model


# 3. 创建推理模型 - 增加模型复杂度
def create_inference_model():
    """创建用于推理的主模型架构"""
    model = models.Sequential([
        Input(shape=(28, 28, 1)),

        # 第一组卷积层
        layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # 第二组卷积层
        layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # 第三组卷积层
        layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),

        # 全连接分类器
        layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ], name='inference')
    return model


# 4. 组合训练模型 - 数据增强 + 推理模型
train_model = models.Sequential([
    create_augmentation_model(),
    create_inference_model()
])

# 5. 优化训练配置
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0
)

# 6. 编译模型
train_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. 设置回调函数
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_train_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

# 8. 执行训练 - 增加训练轮次和减小批量大小
history = train_model.fit(
    train_images, train_labels,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# 7. 保存纯推理模型（无数据增强层）
inference_model = create_inference_model()

# 获取推理模型需要的权重数量
num_inference_weights = len(inference_model.weights)
# 从train_model的权重中提取推理部分的权重（忽略增强层权重）
inference_weights = train_model.get_weights()[-num_inference_weights:]
inference_model.set_weights(inference_weights)

# 编译推理模型（使用与训练模型相同的配置）
inference_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

inference_model.save('inference_model.h5')  # 用于部署的干净模型

test_loss, test_acc = inference_model.evaluate(test_images, test_labels)
print(f"测试准确率: {test_acc * 100:.4f}%")


# 11. 可视化训练历史
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
    plt.savefig('enhance_random_training_history.png')
    plt.show()


# 可视化训练过程
plot_training_history(history)