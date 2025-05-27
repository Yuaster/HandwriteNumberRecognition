import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
import numpy as np

# 1. 分离数据增强与模型架构（关键修改）
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


def preprocess(images):
    images = np.expand_dims(images, -1).astype("float32") / 255.0
    return images


# 创建独立的数据增强管道（训练时使用，推理时自动关闭）
def create_augmentation_model():
    model = models.Sequential([
        layers.RandomRotation(0.05, fill_mode='constant', input_shape=(28, 28, 1)),
        layers.RandomZoom(0.05, fill_mode='constant'),
    ], name='augmentation')
    return model


# 预处理数据（保持基础预处理）
train_images = preprocess(train_images)
test_images = preprocess(test_images)


# 2. 构建无随机操作的推理模型（关键修改）
def create_inference_model():
    model = models.Sequential([
        Input(shape=(28, 28, 1)),

        # 去除非确定性卷积参数
        layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ], name='inference')
    return model


# 3. 训练时组合增强模型（解决 ImageProjectiveTransformV3 警告）
train_model = models.Sequential([
    create_augmentation_model(),
    create_inference_model()
])

# 4. 优化训练配置（添加梯度裁剪和EMA）
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0
)

train_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 修改回调配置（添加TensorBoard）
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # 更保守的学习率衰减
        patience=3
    ),
    tf.keras.callbacks.ModelCheckpoint(
        '../self_model/best_train_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

# 6. 执行训练
history = train_model.fit(
    train_images, train_labels,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# 7. 保存纯推理模型（无数据增强层）
inference_model = create_inference_model()
inference_model.set_weights(train_model.get_weights())  # 跳过增强层权重
inference_model.save('inference_model.h5')  # 用于部署的干净模型
