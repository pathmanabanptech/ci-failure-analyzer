"""Model errors, importable without loading a model SDK. Messages are user-facing."""


class ModelError(Exception):
    """Base class; the message is meant for the user as-is."""


class ModelConfigError(ModelError):
    """Missing or invalid configuration."""


class ModelUnavailableError(ModelError):
    """Provider unreachable, model not found, credentials rejected, or the call failed."""
