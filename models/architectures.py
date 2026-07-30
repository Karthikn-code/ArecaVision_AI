import tensorflow as tf
from config.config import INPUT_SHAPE, CLASS_NAMES
from utils.logger import get_logger

logger = get_logger("ModelArchitectures")

DEFAULT_NUM_CLASSES = len(CLASS_NAMES)

# L2 weight decay applied across dense layers
L2_REG = tf.keras.regularizers.l2(1e-4)


def _build_classification_head(x, num_classes):
    """
    Shared, stronger classification head used across all three backbones:
      GlobalAveragePooling2D -> BN -> Dropout(0.5) ->
      Dense(512, SiLU) -> BN -> Dropout(0.4) ->
      Dense(256, SiLU) -> BN -> Dropout(0.3) ->
      Dense(num_classes, softmax)

    Using SiLU (Swish) instead of ReLU for smoother gradient flow and
    L2 regularization to reduce overfitting on the 14-class task.
    """
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.50)(x)

    x = tf.keras.layers.Dense(512, activation="swish", kernel_regularizer=L2_REG)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    x = tf.keras.layers.Dense(256, activation="swish", kernel_regularizer=L2_REG)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", kernel_regularizer=L2_REG
    )(x)
    return outputs


def build_efficientnet_b0(num_classes=DEFAULT_NUM_CLASSES, freeze_base=True):
    logger.info("Building EfficientNet-B0 transfer learning model (strong head)...")
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
    outputs = _build_classification_head(x, num_classes)

    model = tf.keras.Model(inputs, outputs, name="ArecaVision_EfficientNetB0")
    return model


def build_mobilenet_v3(num_classes=DEFAULT_NUM_CLASSES, freeze_base=True):
    logger.info("Building MobileNetV3 transfer learning model (strong head)...")
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
    outputs = _build_classification_head(x, num_classes)

    model = tf.keras.Model(inputs, outputs, name="ArecaVision_MobileNetV3")
    return model


def build_resnet50(num_classes=DEFAULT_NUM_CLASSES, freeze_base=True):
    logger.info("Building ResNet50 transfer learning model (strong head)...")
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
    outputs = _build_classification_head(x, num_classes)

    model = tf.keras.Model(inputs, outputs, name="ArecaVision_ResNet50")
    return model


def unfreeze_model_top_layers(model, unfreeze_layers=80):
    """
    Unfreezes the top `unfreeze_layers` backbone layers for fine-tuning.
    BatchNorm layers are kept frozen (inference mode) to preserve stable
    ImageNet statistics and prevent destabilization during fine-tuning.

    Default changed from 40 -> 80 for deeper backbone fine-tuning.
    """
    logger.info(f"Unfreezing top {unfreeze_layers} layers of backbone for fine-tuning...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True
            num_backbone_layers = len(layer.layers)
            freeze_until = max(0, num_backbone_layers - unfreeze_layers)

            for sublayer in layer.layers[:freeze_until]:
                sublayer.trainable = False
            for sublayer in layer.layers[freeze_until:]:
                # Keep all BatchNorm frozen during fine-tuning (critical for stability)
                if isinstance(sublayer, tf.keras.layers.BatchNormalization):
                    sublayer.trainable = False
                else:
                    sublayer.trainable = True

    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    logger.info(f"Trainable parameters after unfreeze: {trainable_params:,}")
