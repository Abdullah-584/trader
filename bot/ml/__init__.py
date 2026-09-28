"""Local ML layer: feature building and walk-forward training (scikit-learn).

Everything trains on your machine on historical candles. No lookahead:
labels are future returns, features are bar-close values, and the
walk-forward split with an embargo gap guarantees the model never sees
the future it predicts.
"""

from bot.ml.train import TrainingResult, train_classifier

__all__ = ["TrainingResult", "train_classifier"]
