"""Run the portable analysis worker inside a Windows CPU-rate-limited job."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import runpy
import sys


CPU_LIMITS = (20, 40, 60)
DEFAULT_CPU_LIMIT = 40
LIMIT_FAILED = "CPU_LIMIT_FAILED"


class CpuRateControl(ctypes.Structure):
    _fields_ = (("flags", wintypes.DWORD), ("rate", wintypes.DWORD))


def limit_current_process(percent):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    )
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        job_cpu_rate_control = 15
        enable_hard_cap = 0x1 | 0x4
        control = CpuRateControl(enable_hard_cap, percent * 100)
        if not kernel.SetInformationJobObject(job, job_cpu_rate_control, ctypes.byref(control), ctypes.sizeof(control)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        # The assigned process keeps the job in effect until it exits.
        kernel.CloseHandle(job)


def main():
    _, setting, script, *arguments = sys.argv
    percent = int(setting)
    if percent not in CPU_LIMITS:
        raise ValueError("Unknown CPU limit")
    try:
        if os.name != "nt":
            raise OSError("CPU rate control requires Windows.")
        limit_current_process(percent)
    except OSError as error:
        print(f"{LIMIT_FAILED}: {error}", file=sys.stderr, flush=True)
        return 1
    worker = Path(script).resolve()
    sys.path.insert(0, str(worker.parent))
    sys.argv = [str(worker), *arguments]
    print(f"CPU_LIMIT_ACTIVE {percent}%", flush=True)
    runpy.run_path(str(worker), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
