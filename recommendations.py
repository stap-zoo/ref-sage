import warnings

class ParamRecommendationWarning(UserWarning):
    """A parameter deviates from the recommended ("official") settings."""

class ModeRecommendationWarning(UserWarning):
    """A mode deviates from the recommended ("official") settings."""

def recommend(msg: str, toy: bool, stacklevel: int = 3) -> None:
    """Enforce a recommendation-level check. Toy instances explicitly disclaim any
    target security level, so the shortfall is only surfaced as a warning; everything
    else must satisfy the recommendation, or construction fails outright."""
    if toy:
        warnings.warn(msg, ParamRecommendationWarning, stacklevel=stacklevel)
    else:
        raise ParamRecommendationWarning(msg)
