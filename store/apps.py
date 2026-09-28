"""
store/apps.py — AppConfig for the store app.

ML model singletons are loaded here (once, at server startup) via ready(),
never inside views or serializers.  The loaders themselves live in
store/ml/loader.py so this file stays thin.
"""

from django.apps import AppConfig


class StoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "store"

    def ready(self) -> None:
        """
        Called once when Django finishes startup.

        We deliberately do NOT load ML models here unconditionally —
        loading heavy GPU models during `migrate`, `shell`, or management
        commands that don't need them wastes time and VRAM.

        Instead, each management command that needs an ML model calls
        store.ml.loader.get_sentiment_predictor() or
        store.ml.loader.get_translation_pipeline() on first use.
        Those functions cache the result in a module-level singleton
        so the model is loaded at most once per process.
        """
        pass
