import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Model
import numpy as np


def preprocess(images):
    images = np.expand_dims(images, -1).astype("float32") / 255.0
    return images


def create_augmentation_model():
    model = models.Sequential([
        layers.RandomRotation(0.05, fill_mode='constant', input_shape=(28, 28, 1)),
        layers.RandomZoom(0.05, fill_mode='constant'),
    ], name='augmentation')
    return model


class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]

        self.shared_layer_one = layers.Dense(channel // self.ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.shared_layer_two = layers.Dense(channel,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')

        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.global_max_pool = layers.GlobalMaxPooling2D()
        self.reshape = layers.Reshape((1, 1, channel))
        self.add = layers.Add()
        self.activation = layers.Activation('sigmoid')
        self.multiply = layers.Multiply()

    def call(self, inputs):
        avg_pool = self.global_avg_pool(inputs)
        avg_pool = self.reshape(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = self.global_max_pool(inputs)
        max_pool = self.reshape(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        cbam_feature = self.add([avg_pool, max_pool])
        cbam_feature = self.activation(cbam_feature)

        return self.multiply([inputs, cbam_feature])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({"ratio": self.ratio})
        return config


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.avg_pool = layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))
        self.max_pool = layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))
        self.concat = layers.Concatenate(axis=3)
        self.conv = layers.Conv2D(filters=1,
                                  kernel_size=self.kernel_size,
                                  strides=1,
                                  padding='same',
                                  activation='sigmoid',
                                  kernel_initializer='he_normal',
                                  use_bias=False)
        self.multiply = layers.Multiply()

    def call(self, inputs):
        if tf.keras.backend.image_data_format() == 'channels_first':
            cbam_feature = tf.keras.backend.permute_dimensions(inputs, (0, 2, 3, 1))
        else:
            cbam_feature = inputs

        avg_pool = self.avg_pool(cbam_feature)
        max_pool = self.max_pool(cbam_feature)
        concat = self.concat([avg_pool, max_pool])
        cbam_feature = self.conv(concat)

        if tf.keras.backend.image_data_format() == 'channels_first':
            cbam_feature = tf.keras.backend.permute_dimensions(cbam_feature, (0, 3, 1, 2))

        return self.multiply([inputs, cbam_feature])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class CBAMBlock(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAMBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channel_att = ChannelAttention(self.ratio)
        self.spatial_att = SpatialAttention()

    def call(self, inputs):
        x = self.channel_att(inputs)
        x = self.spatial_att(x)
        return x

    def get_config(self):
        config = super(CBAMBlock, self).get_config()
        config.update({"ratio": self.ratio})
        return config


def create_inference_model():
    model = models.Sequential([
        Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        CBAMBlock(ratio=8),

        layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        CBAMBlock(ratio=8),

        layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ], name='inference_with_attention')
    return model


def main():
    # 加载数据
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # 预处理数据
    train_images = preprocess(train_images)
    test_images = preprocess(test_images)

    # 创建训练模型
    train_model = models.Sequential([
        create_augmentation_model(),
        create_inference_model()
    ])

    # 优化训练配置
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )

    train_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 修改回调配置
    callbacks = [
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
            '../self_model/best_train_model_with_attention.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.TensorBoard(log_dir='logs_with_attention')
    ]

    # 执行训练
    print("开始训练模型...")
    history = train_model.fit(
        train_images, train_labels,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    # 保存纯推理模型
    print("训练完成，正在保存推理模型...")
    inference_model = create_inference_model()
    inference_model.set_weights(train_model.get_weights())
    inference_model.save('inference_model_with_attention.h5')
    print("模型已成功保存为 inference_model_with_attention.h5")


if __name__ == "__main__":
    main()