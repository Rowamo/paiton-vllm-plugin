import ctypes
import json
import logging
from pathlib import Path

from paiton_vllm_plugin.artifact_manifest import (
    ARTIFACT_BINARY_ABI_VERSION,
    ArtifactCompatibilityError,
    detect_runtime_gpu_arch,
    load_and_validate_artifact_manifest,
    manifest_path_for,
)


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
        self._validate_manifest_before_load()
        # Load the dynamic library into memory
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.is_open = True  # Track if the library is currently loaded
        try:
            self._validate_binary_descriptor()
        except Exception:
            self.close()
            raise

    def _validate_manifest_before_load(self):
        manifest_path = manifest_path_for(self.lib_path)
        if not manifest_path.is_file():
            try:
                selected_arch = detect_runtime_gpu_arch()
            except ArtifactCompatibilityError:
                return
            if selected_arch.startswith("gfx12"):
                raise ArtifactCompatibilityError(
                    f"RDNA4 refuses legacy artifact without manifest: {self.lib_path}"
                )
            return
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            tp_size = int(manifest["parallelism"]["tp_size"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ArtifactCompatibilityError(
                f"invalid artifact manifest before load: {manifest_path}"
            ) from error
        load_and_validate_artifact_manifest(
            self.lib_path,
            expected_arch=detect_runtime_gpu_arch(),
            expected_tp_size=tp_size,
        )

    def _validate_binary_descriptor(self):
        manifest_path = manifest_path_for(self.lib_path)
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            abi_function = self.lib.PaitonArtifactGetBinaryAbiVersion
            arch_function = self.lib.PaitonArtifactGetTargetArch
            wave_function = self.lib.PaitonArtifactGetWaveSize
        except (OSError, json.JSONDecodeError, AttributeError) as error:
            raise ArtifactCompatibilityError(
                "versioned Paiton artifact does not expose its binary ABI descriptor"
            ) from error

        abi_function.restype = ctypes.c_uint32
        arch_function.restype = ctypes.c_char_p
        wave_function.restype = ctypes.c_uint32
        binary_abi = int(abi_function())
        binary_arch_raw = arch_function()
        binary_arch = binary_arch_raw.decode("ascii") if binary_arch_raw else ""
        binary_wave = int(wave_function())
        manifest_target = manifest.get("target", {})
        expected = (
            ARTIFACT_BINARY_ABI_VERSION,
            manifest_target.get("arch"),
            manifest_target.get("wave_size"),
        )
        actual = (binary_abi, binary_arch, binary_wave)
        if actual != expected:
            raise ArtifactCompatibilityError(
                f"artifact binary descriptor {actual} does not match manifest {expected}"
            )

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
