import tensorflow as tf
from config.config import INPUT_SHAPE, CLASS_NAMES
from utils.logger import get_logger

logger = get_logger("ModelArchitectures")

DEFAULT_NUM_CLASSES = len(CLASS_NAMES)

def build_efficientnet_b0(num_classes=DEFAULT_NUM_CLASSES, freeze_base=True):
    logger.info("Building EfficientNet-B0 transfer learning model...")
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    if freeze_base:
        base_model.trainable = False
        
    x = base_model(x, training=False if freeze_base else True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name="ArecaVision_EfficientNetB0")
    return model

def build_mobilenet_v3(num_classes=DEFAULT_NUM_CLASSES, freeze_base=True):
    logger.info("Building MobileNetV3 transfer learning model...")
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    
    base_model = tf.keras.applications.MobileNetV3Large(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    if freeze_base:
        base_model.trainable = False
        
    x = base_model(x, training=False if freeze_base else True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name="ArecaVision_MobileNetV3")
    return model

def build_resnet50(num_classes=DEFAULT_NUM_CLASSES, freeze_base=True):
    logger.info("Building ResNet50 transfer learning model...")
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    if freeze_base:
        base_model.trainable = False
        
    x = base_model(x, training=False if freeze_base else True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name="ArecaVision_ResNet50")
    return model

def unfreeze_model_top_layers(model, unfreeze_layers=40):
    """
    Unfreezes top layers of the backbone model inside the wrapper architecture for fine-tuning.
    """
    logger.info(f"Unfreezing top {unfreeze_layers} layers of backbone for fine-tuning...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True
            if len(layer.layers) > unfreeze_layers:
                for sublayer in layer.layers[:-unfreeze_layers]:
                    sublayer.trainable = False
                for sublayer in layer.layers[-unfreeze_layers:]:
                    if isinstance(sublayer, tf.keras.layers.BatchNormalization):
                        sublayer.trainable = False
                    else:
                        sublayer.trainable = True

