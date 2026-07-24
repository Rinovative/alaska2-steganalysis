"""Binary model evaluation and visualization interfaces.

Provides:
- metrics: ALASKA2 weighted AUC and classification summaries
- plots: training, confusion-matrix, ROC, and score figures
- runner: one-pass binary model evaluation
"""

from __future__ import annotations

from . import evaluation_metrics as metrics
from . import evaluation_plots as plots
from . import evaluation_runner as runner

__all__ = ["metrics", "plots", "runner"]
