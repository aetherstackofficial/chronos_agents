"""runner/sandbox.py: a strategy record rebuilt from meta.json has only a pid; Windows reuses pids, so the pid
counts as the strategy's only when its command line names the strategy."""
import os

import psutil

from runner.sandbox import SandboxManager


def test_a_pid_is_the_strategy_only_when_its_command_line_names_it():
    me = os.getpid()
    word = psutil.Process(me).cmdline()[0]               # a piece of this process's own command line
    assert SandboxManager._is_strategy_process(me, word)
    assert not SandboxManager._is_strategy_process(me, "deadbeef-not-a-strategy")
    assert not SandboxManager._is_strategy_process(None, "x")
    assert not SandboxManager._is_strategy_process(2 ** 22 + 12345, "x")   # no such process


def test_kill_tree_ignores_a_missing_process():
    SandboxManager._kill_tree(2 ** 22 + 12345)                              # no error
