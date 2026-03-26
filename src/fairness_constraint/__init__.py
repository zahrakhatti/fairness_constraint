"""fairness_constraint - Fair supervised learning with smooth nonconvex constraints.

Provides models, training utilities, fairness metrics, and plotting tools
for constrained fair classification experiments.
"""

from .config import get_args, update_config, get_config_value
from .models import Feedforward

__all__ = [
    "get_args",
    "update_config",
    "get_config_value",
    "Feedforward",
]
