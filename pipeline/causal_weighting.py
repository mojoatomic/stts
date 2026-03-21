"""Stage 2: Causal weighting W — per paper Section 4.3.

Amplifies features causally upstream of failure, suppresses downstream.
For turbofan: speed/mechanical sensors > pressure/efficiency > temperature.
"""

from __future__ import annotations

import numpy as np

from pipeline.config import (
    ACTIVE_SENSORS, SENSOR_CAUSAL_GROUPS, CAUSAL_WEIGHT_VALUES,
    FEATURE_CLASS_MULTIPLIERS,
)
from pipeline.feature_extraction import (
    get_feature_class_indices, get_feature_sensor_mapping,
)


def build_weight_vector(sensor_names: list[str] | None = None) -> np.ndarray:
    """Build the diagonal weight vector W.

    Each feature gets: sensor_causal_weight * feature_class_multiplier.
    Covariance features (no single sensor) get the medium causal weight.

    Args:
        sensor_names: list of active sensor names. Defaults to ACTIVE_SENSORS.
    Returns:
        1-D weight vector, same length as the feature vector.
    """
    if sensor_names is None:
        sensor_names = ACTIVE_SENSORS

    n_sensors = len(sensor_names)
    feature_sensor_map = get_feature_sensor_mapping(n_sensors, sensor_names)
    feature_class_idx = get_feature_class_indices(n_sensors)

    # Build sensor -> causal weight lookup
    sensor_weight_lookup = {}
    for group, sensors in SENSOR_CAUSAL_GROUPS.items():
        for s in sensors:
            sensor_weight_lookup[s] = CAUSAL_WEIGHT_VALUES[group]

    total_features = len(feature_sensor_map)
    weights = np.ones(total_features)

    # Apply sensor-level causal weights
    for i, sensor in enumerate(feature_sensor_map):
        if sensor is not None and sensor in sensor_weight_lookup:
            weights[i] = sensor_weight_lookup[sensor]
        elif sensor is None:
            # Covariance features — use a moderate weight
            weights[i] = CAUSAL_WEIGHT_VALUES["medium_high"]

    # Apply feature-class multipliers
    for cls_name, indices in feature_class_idx.items():
        multiplier = FEATURE_CLASS_MULTIPLIERS.get(cls_name, 1.0)
        for idx in indices:
            if idx < total_features:
                weights[idx] *= multiplier

    return weights


def apply_weights(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Element-wise multiply features by causal weight vector.

    Args:
        features: (n_samples, feature_dim) or (feature_dim,)
        weights: (feature_dim,)
    Returns:
        weighted features, same shape as input
    """
    return features * weights
