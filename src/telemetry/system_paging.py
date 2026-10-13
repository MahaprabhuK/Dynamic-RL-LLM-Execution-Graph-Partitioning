r"""
System-wide hard paging counter (Windows).

Reads the raw value of the PDH counter "\Memory\Pages Input/sec": a
cumulative count of pages read from disk to resolve hard page faults, for the
whole machine. Differences between two reads give the pages paged in during
that interval. Unlike the per-process num_page_faults (which counts soft faults
too and does not rise when Windows trims other processes instead of ours), this
rises whenever anything on the machine is paging from disk.

On other platforms, or if PDH is unavailable, read() returns None.
"""

import ctypes
import sys
from ctypes import wintypes

COUNTER_PATH = r"\Memory\Pages Input/sec"


class _PdhRawCounter(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("TimeStamp", wintypes.FILETIME),
                ("FirstValue", ctypes.c_longlong), ("SecondValue", ctypes.c_longlong),
                ("MultiCount", wintypes.DWORD)]


class SystemPagesInput:
    def __init__(self):
        self._query = self._counter = None
        if sys.platform != "win32":
            return
        try:
            pdh = ctypes.WinDLL("pdh")
            query, counter = ctypes.c_void_p(), ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, None, ctypes.byref(query)) != 0:
                return
            if pdh.PdhAddEnglishCounterW(query, COUNTER_PATH, None, ctypes.byref(counter)) != 0:
                pdh.PdhCloseQuery(query)
                return
            self._pdh, self._query, self._counter = pdh, query, counter
        except (OSError, AttributeError):
            self._query = None

    @property
    def available(self):
        return self._query is not None

    def read(self):
        """Cumulative system pages input since boot, or None if unavailable."""
        if self._query is None:
            return None
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return None
        raw, ctype = _PdhRawCounter(), wintypes.DWORD()
        if self._pdh.PdhGetRawCounterValue(self._counter, ctypes.byref(ctype), ctypes.byref(raw)) != 0:
            return None
        return int(raw.FirstValue)

    def close(self):
        if self._query is not None:
            self._pdh.PdhCloseQuery(self._query)
            self._query = None
