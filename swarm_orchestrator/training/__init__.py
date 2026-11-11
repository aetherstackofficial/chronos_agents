"""Chronos v2 model training (see the Model Retraining Guide).

TensorFlow is installed on this machine but nothing here uses it. Stable-
Baselines3 imports tensorboard, which imports TensorFlow when it can find it,
and that doubles every process's memory (measured 548 MB -> 262 MB without it).
With 8 training workers plus eval workers that ran the machine out of RAM, so
the import is blocked for every process that imports this package (the spawned
SubprocVecEnv workers re-import it too). TensorBoard logging still works
through its built-in TensorFlow stub.
"""
import sys

sys.modules.setdefault("tensorflow", None)
