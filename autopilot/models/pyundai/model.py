from abc import abstractmethod, ABC
import os
from decimal import Decimal

import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout, Activation

import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.optimizer_v2.adam import Adam


IM_WIDTH = 320
IM_HEIGHT = 240
MOB_NET_SHAPE = (224,224, 3)
MOB_NET_INPUT = tf.keras.Input(shape=MOB_NET_SHAPE)
IM_SHAPE = (IM_HEIGHT, IM_WIDTH)




def postprocess_speed(speed: float) -> int:
    speed_factor = 35
    speed_threshold = 0.2

    if speed > speed_threshold:
        if speed > 1:
            speed = 1
        return int(speed * speed_factor)
    else:
        return 0


def postprocess_angle(angle: float) -> int:
    angle_threshold = np.clip(angle, 0, 1)
    return int(80 * angle_threshold + 50)


def null_preprocess_fn(input_img):
    return cv2.resize(input_img, MOB_NET_SHAPE[:2])


class AbstractModel(ABC):
    """Base class of a model that other models we work with will extend"""

    def __init__(self):
        self.model = self.create_model()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def compile(self):
        """Compiles model. Should take into account Losses, Loss Weights, and Metrics for both branches"""
        self.model.compile()

class Model(AbstractModel):
    def __init__(self):
        model_path = os.path.join(os.getcwd(),"autopilot\models\pyundai\Pyundai_live_testing_model.h5")
        super().__init__()
        self.compile()
        self.model.load_weights(model_path)

    def compile(self):
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss={
                "angle": "mse",
                "v": "binary_crossentropy",
            },
            loss_weights={
                "angle": 1.,
                "v": 1.,
            }
        )

    def create_model(self):
        mob_net_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=MOB_NET_SHAPE,
            include_top=False,
            weights='imagenet'
        )

        # Finetuning....
        mob_net_model.trainable = True
        num_unfreeze = -1 * int(len(mob_net_model.layers) * 0.3) # 30% seems best, 50% got val loss of 0.03357, 30% got 0.03208
        for layer in mob_net_model.layers[:num_unfreeze]:  # un freeze last 10% of model
            layer.trainable = False
        print(f"{num_unfreeze} layers unfrozen.")

        mob_net_out = mob_net_model(MOB_NET_INPUT, training=False)

        # Angle Branch
        y = Flatten()(mob_net_out)
        y = Dropout(rate=0.3)(y)
        y = Dense(128, activation="elu")(y)
        y = Dropout(rate=0.15)(y)
        y = Dense(64, activation="elu")(y)
        y = Dense(32, activation="elu")(y)
        y = Dense(16, activation="elu")(y)
        y = Dense(1)(y)
        angle_out = Activation("linear", name="angle")(y)

        # Velocity Branch
        z = Flatten()(mob_net_out)
        z = Dropout(rate=0.3)(z)
        z = Dense(128, activation="elu")(z)
        z = Dropout(rate=0.15)(z)
        z = Dense(64, activation="elu")(z)
        z = Dense(32, activation="elu")(z)
        z = Dense(16, activation="elu")(z)
        z = Dense(1)(z)
        v_out = Activation("sigmoid", name="v")(z)

        return keras.Model(inputs=MOB_NET_INPUT, outputs=[angle_out, v_out])

    def preprocess_image(self, image):
        return null_preprocess_fn(image)

    def predict(self, image):
        image = cv2.resize(image, (224,224))
        image = np.expand_dims(image, axis = 0)
        angle, speed = self.model.predict(image)

        output_speed = postprocess_speed(speed)
        output_angle = postprocess_angle(angle)

        return output_angle, output_speed

