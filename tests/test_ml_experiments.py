from examples.ML_EXPERIMENTS.ml_experiment_simple import main as \
    ml_experiment_simple_main
from examples.ML_EXPERIMENTS.ml_experiment_multi import main as \
    ml_experiment_multi_main
from examples.ML_EXPERIMENTS.ml_experiment_complex import main as \
    ml_experiment_complex_main


def test_simple_ml_experiment():
    ml_experiment_simple_main().shutdown()


def test_multi_ml_experiment():
    ml_experiment_multi_main().shutdown()


def test_complex_ml_experiment():
    ml_experiment_complex_main().shutdown()
