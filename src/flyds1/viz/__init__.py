"""Live instrumentation: which populations exist, and what they are doing.

:mod:`flyds1.viz.groups` resolves the neuron table into index arrays once;
:mod:`flyds1.viz.state` reduces a rate vector to the numbers a browser draws.
Neither renders anything, and neither touches the environment.
"""

from flyds1.viz.groups import DEFAULT_TYPE_GROUPS, PopulationIndex, TypeGroup
from flyds1.viz.state import LiveStateBuilder

__all__ = ["DEFAULT_TYPE_GROUPS", "LiveStateBuilder", "PopulationIndex", "TypeGroup"]
