"""Shared activity groups for temporal reasoning."""
from ..models import State

IN_BED = {State.LYING_IN_BED, State.SITTING_ON_BED}
OUT_OF_BED = {State.STANDING, State.WALKING, State.SITTING_OUTSIDE_BED, State.OUT_OF_BED}
