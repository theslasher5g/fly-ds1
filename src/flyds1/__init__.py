"""flyds1 -- a connectome-constrained fly brain wired to a video game.

The package follows the six steps of the roadmap, one sub-package each:

1. ``flyds1.connectome`` -- fetch/parse FlyWire (or synthesise) neuron and
   synapse tables.
2. ``flyds1.net``        -- turn the signed, weighted graph into a simulable
   recurrent network (numpy reference + differentiable PyTorch module).
3. ``flyds1.vision``     -- screen -> ommatidia sampling and Reichardt motion
   pre-processing.
4. ``flyds1.envs``       -- Gymnasium environments: a synthetic arena for
   offline work, plus the screen-capture/key-injection game bridge.
5. ``flyds1.motor``      -- descending-neuron activity -> game actions.
6. ``flyds1.agent``      -- Stable-Baselines3 glue for PPO training.
"""

__version__ = "0.1.0"
