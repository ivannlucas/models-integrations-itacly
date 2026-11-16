"""Why a job's child process (training, async prediction) ended without reporting its result.

The child writes its own terminal state for any Python exception. When it ends without one it
died outside Python: killed by a signal (SIGKILL is what the kernel OOM killer sends inside a
container: the cgroup kills its largest process, the child, and the pod keeps serving), a native
crash (SIGSEGV), or an interpreter exit before reaching the job code (e.g. an import error).
"""
from __future__ import annotations

import signal


def describe_child_exit(exitcode: int | None) -> str:
    """Spanish description of a child's ``multiprocessing`` exit code, for the job's error."""
    if exitcode is None:
        return "el proceso terminó sin código de salida"
    if exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:
            name = f"señal {-exitcode}"
        if -exitcode == signal.SIGKILL:
            return (
                f"el sistema terminó el proceso ({name}); en Kubernetes casi siempre es falta de "
                "memoria (OOM): el entrenamiento necesita más memoria de la que el pod tiene libre"
            )
        return f"el proceso terminó por la señal {name}"
    return f"el proceso terminó con código de salida {exitcode} sin guardar el resultado"
