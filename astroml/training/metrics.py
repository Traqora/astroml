"""Prometheus metrics for training services."""
from prometheus_client import Counter, Gauge, Histogram, Summary

# Training metrics
TRAINING_EPOCHS_TOTAL = Counter(
    "astroml_training_epochs_total",
    "Total number of training epochs completed",
    ["model_type", "dataset"]
)

TRAINING_LOSS = Gauge(
    "astroml_training_loss",
    "Current training loss value",
    ["model_type", "dataset", "phase"]  # phase: train, val, test
)

TRAINING_ACCURACY = Gauge(
    "astroml_training_accuracy",
    "Current accuracy value",
    ["model_type", "dataset", "phase"]
)

TRAINING_DURATION = Histogram(
    "astroml_training_duration_seconds",
    "Time spent training per epoch",
    ["model_type", "dataset"]
)

MODEL_PARAMETERS = Gauge(
    "astroml_model_parameters",
    "Number of model parameters",
    ["model_type"]
)

LEARNING_RATE = Gauge(
    "astroml_learning_rate",
    "Current learning rate",
    ["model_type"]
)

GRADIENT_NORM = Histogram(
    "astroml_gradient_norm",
    "Gradient norm during training",
    ["model_type"]
)

INFERENCE_REQUESTS_TOTAL = Counter(
    "astroml_inference_requests_total",
    "Total number of inference requests",
    ["model_type"]
)

INFERENCE_LATENCY = Histogram(
    "astroml_inference_latency_seconds",
    "Time spent per inference request",
    ["model_type"]
)

INFERENCE_ERRORS_TOTAL = Counter(
    "astroml_inference_errors_total",
    "Total number of inference errors",
    ["model_type", "error_type"]
)
