import ctypes
import logging


def _dlclose(dll: ctypes.CDLL):
    f_dlclose = None

    syms = ctypes.CDLL(None)
    if not hasattr(syms, "dlclose"):
        # Alpine Linux
        syms = ctypes.CDLL("libc.so")

    if hasattr(syms, "dlclose"):
        f_dlclose = syms.dlclose

    if f_dlclose is not None:
        f_dlclose.argtypes = [ctypes.c_void_p]
        f_dlclose(dll._handle)
    else:
        logging.warning(
            "dll unloading function was not found, library may not be unloaded properly!"
        )


class MemLoader:
    def __init__(self, lib_path: str):
        self.lib_path = lib_path
        # Load the dynamic library into memory
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.is_open = True  # Track if the library is currently loaded

    def close(self):
        """Safely close the loaded library if it's open."""
        if self.is_open:
            _dlclose(self.lib)  # Assumes _dlclose is defined to handle closing the library
            self.is_open = False

    def __getattr__(self, name):
        """Dynamically handle getting attributes (functions) from the loaded library."""
        if not self.is_open:
            raise RuntimeError(f"Cannot use closed library: {self.lib_path}")

        # Retrieve the specified function from the library
        method = getattr(self.lib, name)

        def _wrapped_func(*args):
            """Wrap the library function to handle errors."""
            err = method(*args)
            if err:
                raise RuntimeError(f"Error in function {method.__name__}")

        return _wrapped_func
