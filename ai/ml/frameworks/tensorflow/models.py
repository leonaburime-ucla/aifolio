from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np
import tensorflow as tf

try:
    from ...ml_util import TaskType
except ImportError:  # pragma: no cover
    from ml_util import TaskType  # type: ignore

from ...core.types import ModelBundle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(
    input_dim: int,
    output_dim: int,
    task: TaskType,
    training_mode: str,
    hidden_dim: int,
    num_hidden_layers: int,
    dropout: float,
    learning_rate: float,
) -> tf.keras.Model:
    quantile_tau = 0.8

    def pinball_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile_tau * err, (quantile_tau - 1.0) * err))

    inputs = tf.keras.layers.Input(shape=(input_dim,))

    if training_mode == "linear_glm_baseline":
        logits = tf.keras.layers.Dense(output_dim, activation=None, name="linear_head")(inputs)
    elif training_mode == "entity_embeddings":
        embed_dim = max(16, min(hidden_dim, 128))
        embedded = tf.keras.layers.Dense(embed_dim, activation=None, name="entity_embed_proj")(inputs)
        embedded = tf.keras.layers.BatchNormalization(name="entity_embed_bn")(embedded)
        embedded = tf.keras.layers.ReLU(name="entity_embed_relu")(embedded)
        deep = embedded
        for idx in range(max(1, num_hidden_layers)):
            deep = tf.keras.layers.Dense(hidden_dim, activation=None, name=f"entity_dense_{idx}")(deep)
            deep = tf.keras.layers.BatchNormalization(name=f"entity_bn_{idx}")(deep)
            deep = tf.keras.layers.ReLU(name=f"entity_relu_{idx}")(deep)
            if dropout > 0:
                deep = tf.keras.layers.Dropout(dropout, name=f"entity_dropout_{idx}")(deep)
        logits = tf.keras.layers.Dense(output_dim, activation=None, name="entity_head")(deep)
    elif training_mode == "wide_and_deep":
        wide_logits = tf.keras.layers.Dense(output_dim, activation=None, name="wide_head")(inputs)
        deep = inputs
        for idx in range(max(1, num_hidden_layers)):
            deep = tf.keras.layers.Dense(hidden_dim, activation=None, name=f"deep_dense_{idx}")(deep)
            deep = tf.keras.layers.BatchNormalization(name=f"deep_bn_{idx}")(deep)
            deep = tf.keras.layers.ReLU(name=f"deep_relu_{idx}")(deep)
            if dropout > 0:
                deep = tf.keras.layers.Dropout(dropout, name=f"deep_dropout_{idx}")(deep)
        deep_logits = tf.keras.layers.Dense(output_dim, activation=None, name="deep_head")(deep)
        logits = tf.keras.layers.Add(name="wide_deep_add")([wide_logits, deep_logits])
    elif training_mode == "time_aware_tabular":
        gate = tf.keras.layers.Dense(input_dim, activation="sigmoid", name="temporal_gate")(inputs)
        gated = tf.keras.layers.Multiply(name="temporal_multiply")([inputs, gate])
        merged = tf.keras.layers.Concatenate(name="temporal_concat")([inputs, gated])
        deep = merged
        for idx in range(max(1, num_hidden_layers)):
            deep = tf.keras.layers.Dense(hidden_dim, activation=None, name=f"time_dense_{idx}")(deep)
            deep = tf.keras.layers.BatchNormalization(name=f"time_bn_{idx}")(deep)
            deep = tf.keras.layers.ReLU(name=f"time_relu_{idx}")(deep)
            if dropout > 0:
                deep = tf.keras.layers.Dropout(dropout, name=f"time_dropout_{idx}")(deep)
        logits = tf.keras.layers.Dense(output_dim, activation=None, name="time_head")(deep)
    else:
        deep = inputs
        for idx in range(max(1, num_hidden_layers)):
            deep = tf.keras.layers.Dense(hidden_dim, activation=None, name=f"dense_{idx}")(deep)
            deep = tf.keras.layers.BatchNormalization(name=f"bn_{idx}")(deep)
            deep = tf.keras.layers.ReLU(name=f"relu_{idx}")(deep)
            if dropout > 0:
                deep = tf.keras.layers.Dropout(dropout, name=f"dropout_{idx}")(deep)
        if training_mode == "autoencoder_head":
            bottleneck_dim = max(8, hidden_dim // 2)
            bottleneck = tf.keras.layers.Dense(bottleneck_dim, activation="relu", name="autoencoder_bottleneck")(deep)
            recon = tf.keras.layers.Dense(input_dim, activation="linear", name="reconstruction_output")(bottleneck)
            pred_logits = tf.keras.layers.Dense(output_dim, activation=None, name="main_logits")(bottleneck)
        elif training_mode == "multi_task_learning":
            shared = tf.keras.layers.Dense(hidden_dim, activation="relu", name="shared_trunk")(deep)
            main_logits = tf.keras.layers.Dense(output_dim, activation=None, name="main_logits")(shared)
            aux_logits = tf.keras.layers.Dense(output_dim, activation=None, name="aux_logits")(shared)
        else:
            logits = tf.keras.layers.Dense(output_dim, activation=None, name="mlp_head")(deep)

    if task == "classification":
        loss_fn: str | tf.keras.losses.Loss = "sparse_categorical_crossentropy"
        if training_mode == "calibrated_classifier":
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05)
        if training_mode == "autoencoder_head":
            main_probs = tf.keras.layers.Activation("softmax", name="main_output")(pred_logits)
            model = tf.keras.Model(inputs=inputs, outputs=[main_probs, recon], name=f"tf_{training_mode}")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={"main_output": loss_fn, "reconstruction_output": "mse"},
                loss_weights={"main_output": 1.0, "reconstruction_output": 0.2},
                metrics={"main_output": ["accuracy"]},
            )
        elif training_mode == "multi_task_learning":
            main_probs = tf.keras.layers.Activation("softmax", name="main_output")(main_logits)
            aux_probs = tf.keras.layers.Activation("softmax", name="aux_output")(aux_logits)
            model = tf.keras.Model(inputs=inputs, outputs=[main_probs, aux_probs], name=f"tf_{training_mode}")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={"main_output": loss_fn, "aux_output": loss_fn},
                loss_weights={"main_output": 1.0, "aux_output": 0.3},
                metrics={"main_output": ["accuracy"]},
            )
        else:
            outputs = tf.keras.layers.Activation("softmax", name="softmax")(logits)
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"tf_{training_mode}")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_fn,
                metrics=["accuracy"],
            )
    else:
        loss_fn: str | Callable[..., tf.Tensor] = pinball_loss if training_mode == "quantile_regression" else "mse"
        if training_mode == "autoencoder_head":
            main_out = tf.keras.layers.Activation("linear", name="main_output")(pred_logits)
            model = tf.keras.Model(inputs=inputs, outputs=[main_out, recon], name=f"tf_{training_mode}")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={"main_output": loss_fn, "reconstruction_output": "mse"},
                loss_weights={"main_output": 1.0, "reconstruction_output": 0.2},
            )
        elif training_mode == "multi_task_learning":
            main_out = tf.keras.layers.Activation("linear", name="main_output")(main_logits)
            aux_out = tf.keras.layers.Activation("linear", name="aux_output")(aux_logits)
            model = tf.keras.Model(inputs=inputs, outputs=[main_out, aux_out], name=f"tf_{training_mode}")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={"main_output": loss_fn, "aux_output": "mse"},
                loss_weights={"main_output": 1.0, "aux_output": 0.3},
            )
        else:
            outputs = tf.keras.layers.Activation("linear", name="regression_output")(logits)
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"tf_{training_mode}")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_fn,
            )

    return model


def model_hidden_dim(bundle: ModelBundle) -> int:
    return int((bundle.model_config or {}).get("hidden_dim", 128))


def model_num_hidden_layers(bundle: ModelBundle) -> int:
    return int((bundle.model_config or {}).get("num_hidden_layers", 2))


def model_dropout(bundle: ModelBundle) -> float:
    return float((bundle.model_config or {}).get("dropout", 0.1))
