"""Landsat Collection 2 Level-2 preprocessing pipeline (v1)."""

__all__ = ["run_preprocessing_pipeline"]


def __getattr__(name: str):
    if name == "run_preprocessing_pipeline":
        from preprocessing.pipeline import run_preprocessing_pipeline

        return run_preprocessing_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
