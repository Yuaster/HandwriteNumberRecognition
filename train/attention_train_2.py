import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
import numpy as np
import os


# 1. 数据处理模块
def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    def preprocess(images):
        images = np.expand_dims(images, -1).astype("float32") / 255.0
        return images

    train_images = preprocess(train_images)
    test_images = preprocess(test_images)

    return (train_images, train_labels), (test_images, test_labels)


# 2. 注意力机制模块
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


# 3. 模型构建模块
def create_augmentation_model():
    model = models.Sequential([
        layers.RandomRotation(0.05, fill_mode='constant', input_shape=(28, 28, 1)),
        layers.RandomZoom(0.05, fill_mode='constant'),
    ], name='augmentation')
    return model


def create_inference_model():
    model = models.Sequential([
        Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        SEBlock(reduction_ratio=16),

        layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        SEBlock(reduction_ratio=16),

        layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        SEBlock(reduction_ratio=16),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ], name='inference')
    return model


def create_training_model():
    return models.Sequential([
        create_augmentation_model(),
        create_inference_model()
    ])


# 4. 训练配置模块
def get_optimizer():
    return tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )


def get_callbacks(model_save_path='../self_model/best_train_attention_model.h5', log_dir='logs'):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ]


# 5. 模型训练模块
def train_model_1(train_images, train_labels,
                epochs=20, batch_size=256, validation_split=0.1):
    train_model = create_training_model()
    train_model.compile(
        optimizer=get_optimizer(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = train_model.fit(
        train_images, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=get_callbacks(),
        shuffle=True,
        verbose=1
    )

    return train_model, history


# 6. 模型保存模块（改进版本）
def save_inference_model(train_model, save_path='inference_model.h5'):
    """通过层索引复制权重，避免整体权重列表不匹配问题"""
    # 创建推理模型
    inference_model = create_inference_model()

    # 获取训练模型和推理模型的层
    train_layers = train_model.layers[1].layers  # 跳过增强层，获取推理模型层
    inference_layers = inference_model.layers

    # 确保层数匹配
    assert len(train_layers) == len(inference_layers), f"层数不匹配: {len(train_layers)} != {len(inference_layers)}"

    # 逐层复制权重
    for i in range(len(train_layers)):
        if len(train_layers[i].weights) > 0:
            inference_layers[i].set_weights(train_layers[i].get_weights())
            print(f"已复制层 {i}: {inference_layers[i].name}")

    # 保存模型
    inference_model.save(save_path)
    print(f"推理模型已保存至: {save_path}")
    return inference_model


# 7. 主函数
def main():
    print("加载数据...")
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

    print("构建模型...")
    train_model, history = train_model_1(train_images, train_labels)

    print("评估模型...")
    test_loss, test_acc = train_model.evaluate(test_images, test_labels, verbose=2)
    print(f"测试准确率: {test_acc:.4f}")

    print("保存推理模型...")
    save_inference_model(train_model)

    print("训练完成!")


if __name__ == "__main__":
    main()