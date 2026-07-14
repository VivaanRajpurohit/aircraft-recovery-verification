import numpy as np

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data.preprocessing import ObservationPreprocessor


def test_preprocessing_preserves_exact_42_features_and_masks() -> None:
    values = np.arange(84, dtype=np.float32).reshape(2, 42)
    nav_index = OBSERVATION_FEATURES.index("navigation_data_valid")
    airport_index = OBSERVATION_FEATURES.index("airport_available")
    values[:, nav_index] = [0.0, 1.0]
    values[:, airport_index] = [1.0, 0.0]
    preprocessor = ObservationPreprocessor()
    preprocessor.fit(values)
    transformed = preprocessor.transform(values)
    assert transformed.shape == (2, 42)
    assert np.array_equal(transformed[:, nav_index], values[:, nav_index])
    assert np.array_equal(transformed[:, airport_index], values[:, airport_index])


def test_missing_values_are_replaced_deterministically() -> None:
    training = np.ones((3, 42), dtype=np.float32)
    preprocessor = ObservationPreprocessor()
    preprocessor.fit(training)
    invalid = np.ones(42, dtype=np.float32)
    invalid[0], invalid[1], invalid[2] = np.nan, np.inf, -np.inf
    first = preprocessor.transform(invalid)
    second = preprocessor.transform(invalid)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_validation_values_do_not_influence_training_statistics() -> None:
    training = np.zeros((4, 42), dtype=np.float32)
    validation = np.full((2, 42), 100000.0, dtype=np.float32)
    preprocessor = ObservationPreprocessor()
    statistics = preprocessor.fit(training)
    preprocessor.transform(validation)
    assert statistics.means[0] == 0.0
    assert statistics.fitted_transition_count == 4

