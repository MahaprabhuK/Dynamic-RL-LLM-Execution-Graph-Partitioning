import os
import sys

_torch_lib = r"C:\Users\Pavithran\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
if os.path.exists(_torch_lib):
    try:
        os.add_dll_directory(_torch_lib)
    except Exception:
        pass
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
try:
    import torch
except Exception:
    pass
