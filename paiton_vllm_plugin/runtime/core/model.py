"""
Python bindings to the Paiton runtime.
"""

import ctypes
import enum
import math
import struct
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union, NamedTuple, TypeVar
from .utils.lib_wrapper import MemLoader
from .utils.torch_utils import write_tensor_binary

import numpy as np

from .dtype import dtype_str_to_enum

# Controls how many runtimes will be used in ModelContainer by default.
# See the runtime README.md for more information on the Model/ModelContainer
# system and the num_runtimes parameter.
# This value is used as the default for the num_runtimes argument
# in both Model.__init__ and compile_model. Changing it will have no
# effect since Python default arguments only get evaluated once.
NUM_RUNTIMES = 1

TorchTensor = TypeVar("TorchTensor")

# Define a mapping from PyTorch dtype objects to string representations.
#
# Paiton's public dtype strings are legacy names inherited from the original
# runtime and compiler. On gfx950, generated code interprets the FP8 payloads
# as standard OCP FP8 even though the PData dtype string still says
# `*fnuz`. Because of that, the dtype string alone must not be used as a cue
# to reinterpret torch FP8 tensors between fn and fnuz bit layouts.
dtype_mapping = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float8_e4m3fn: "float8_e4m3fnuz",
    torch.float8_e4m3fnuz: "float8_e4m3fnuz",
    torch.float8_e5m2: "float8_e5m2fnuz",
    torch.float8_e5m2fnuz: "float8_e5m2fnuz",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int16: "int16",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def runtime_uses_fnuz_fp8() -> bool:
    """Return True when the active ROCm device uses FNUZ FP8 natively.

    MI300-class gfx94x devices expose FNUZ as the native FP8 format, while
    MI350/MI355 gfx950 devices expose standard OCP FP8 (`fn` / `e5m2`)
    intrinsics. The Paiton runtime still uses legacy `*fnuz` dtype strings in
    both cases, so callers should use this helper when deciding whether an
    actual tensor payload conversion is needed.
    """
    try:
        if not torch.cuda.is_available():
            return False
        gcn_arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    except Exception:
        return False
    return "gfx94" in gcn_arch


def torch_dtype_to_string(dtype):
    # Use the dictionary to directly return the string representation
    if dtype in dtype_mapping:
        return dtype_mapping[dtype]
    else:
        raise ValueError(
            f"Unsupported input dtype {dtype}! Supported dtypes are: {list(dtype_mapping.values())}"
        )


# This could be interesting for optimizations later on
class PaitonMemcpyKind(enum.Enum):
    HostToDevice = 0
    DeviceToHost = 1
    DeviceToDevice = 2


class PaitonAllocatorKind(enum.Enum):
    DEFAULT = 0
    TRACKING = 1


class PaitonModelCapability(enum.IntFlag):
    NONE = 0
    LOGITS_REPLICATED_ON_ALL_TP_RANKS = 1 << 0


def _query_model_capabilities(memloader, handle) -> PaitonModelCapability:
    """Read artifact capabilities, defaulting old ABI artifacts to none."""
    if not hasattr(memloader.lib, "PaitonModelContainerGetCapabilities"):
        return PaitonModelCapability.NONE
    capabilities = ctypes.c_uint64()
    memloader.PaitonModelContainerGetCapabilities(
        handle, ctypes.byref(capabilities)
    )
    return PaitonModelCapability(capabilities.value)


class PData(NamedTuple):
    """
    Input or output tensor for Model.run. We require the extra data for safety
    checks inside the runtime.
    """

    data_ptr: int
    shape: List[int]
    dtype: str


class _PaitonShape(ctypes.Structure):
    _fields_ = [
        ("shape_data", ctypes.POINTER(ctypes.c_longlong)),
        ("size", ctypes.c_size_t),
    ]


class _CFormatPData(ctypes.Structure):
    _fields_ = [
        ("pointer", ctypes.c_void_p),
        ("shape", _PaitonShape),
        ("dtype", ctypes.c_int),
    ]


def _check_tensors(
    tensor_list: Union[Dict[str, TorchTensor], List[TorchTensor]],
    is_error_fn: Callable[[TorchTensor], bool],
    list_name: str,
    condition_description: str,
):
    """
    Helper for various input/output sanity checks.
    """
    if isinstance(tensor_list, dict):
        tensor_list = tensor_list.values()

    for i, tensor in enumerate(tensor_list):
        if is_error_fn(tensor):
            raise ValueError(f"{list_name}[{i}] failed check: {condition_description}")


def _check_tensors_contiguous_and_on_gpu(
    tensors: Union[Dict[str, TorchTensor], List[TorchTensor]], name: str
):
    def is_bad_tensor(tensor: TorchTensor) -> bool:
        return not tensor.is_contiguous() or not tensor.is_cuda

    _check_tensors(tensors, is_bad_tensor, name, "contiguous and on GPU")


def _check_tensors_contiguous_and_on_host(
    tensors: Union[Dict[str, TorchTensor], List[TorchTensor]], name: str
):
    def is_bad_tensor(tensor: TorchTensor) -> bool:
        return not tensor.is_contiguous() or tensor.is_cuda

    _check_tensors(tensors, is_bad_tensor, name, "contiguous and on host")


def torch_to_paiton_data(tensor: TorchTensor) -> PData:
    """
    Convert a torch Tensor to a PData.
    """
    return PData(
        tensor.data_ptr(), list(tensor.size()), torch_dtype_to_string(tensor.dtype)
    )


def _convert_tensor_args(params: Union[List[TorchTensor], Dict[str, TorchTensor]]):
    """
    Helper function for the WithTensors APIs.
    """
    if isinstance(params, dict):
        result = {name: torch_to_paiton_data(x) for name, x in params.items()}
    else:
        result = [torch_to_paiton_data(x) for x in params]
    return result


def _reshape_tensor(tensor: TorchTensor, shape: List[int]) -> TorchTensor:
    """
    Reinterpret a blob of contiguous memory as some shape. Used to convert
    outputs in RunWithTensors.
    """
    assert (
        tensor.ndim == len(shape)
    ), f"Expected output tensor's ndim to match the length of Run()'s return value: {tensor.ndim=} != {len(shape)=}"
    numel = math.prod(shape)
    new_tensor = tensor.flatten()[:numel]
    return new_tensor.reshape(shape)


class Model:
    def __init__(
        self,
        lib_path: str,
        num_runtimes: int = NUM_RUNTIMES,
        allocator_kind: Optional[PaitonAllocatorKind] = None,
    ):
        """
        Instantiates a wrapper around the C++ model_interface.

        Parameters
        ----------
        lib_path : str
            The path to the compiled .so
        num_runtimes : int, optional
            How many runtimes should be stored in the internal pool. This
            determines how many inferences can happen concurrently. By
            default, set to 1. Must be positive.
        allocator_kind : PaitonAllocatorKind, optional
            What type of allocator to use when allocating GPU memory.
        """
        # Set of pointers allocated with numpy_to_paiton_data.
        # If the user forgets to free their data, we use this to
        # avoid leaking memory.
        self._allocated_paiton_data = set()

        if num_runtimes <= 0:
            raise ValueError(f"num_runtimes must be positive, but got {num_runtimes}")

        self.memloader = MemLoader(lib_path)
        self.lib_path = lib_path
        self.handle = ctypes.c_void_p()
        self.allocator_handle = ctypes.c_void_p()
        if allocator_kind is not None:
            self.memloader.PaitonAllocatorCreate(
                ctypes.byref(self.allocator_handle),
                ctypes.c_int(allocator_kind.value),
            )

        self.memloader.PaitonModelContainerCreate(
            ctypes.pointer(self.handle),
            ctypes.c_size_t(num_runtimes),
            self.allocator_handle,
        )

        # Capabilities are authoritative properties of the compiled graph.
        # Artifacts built before this ABI was added have no symbol and safely
        # default to the legacy behavior.
        self.capabilities = _query_model_capabilities(self.memloader, self.handle)

        # We use this list to add reference counts of Torch tensors
        # to avoid lifetime issues caused by user misuse.
        self.torch_constant_tensors = {}

        # The corresponding sorted_graph. Optional. For debugging purpose.
        self.debug_sorted_graph = None

        self._output_name_to_index = self._construct_output_name_to_index_map()
        self._input_name_to_index = self._construct_input_name_to_index_map()
        self._output_ndims = [
            len(self.get_output_maximum_shape(i))
            for i in range(len(self._output_name_to_index))
        ]

    def __enter__(self):
        return self

    def has_capability(self, capability: PaitonModelCapability) -> bool:
        return bool(self.capabilities & capability)

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        # Copy to avoid set size changed during iteration
        for ptr in list(self._allocated_paiton_data):
            self.free_gpu_memory(ptr, sync=True)

        # Check that it exists since we may have thrown
        # an exception before initializing it.
        if hasattr(self, "memloader"):
            if self.handle:
                self.memloader.PaitonModelContainerDelete(self.handle)
                self.handle = ctypes.c_void_p()

            if self.allocator_handle:
                self.memloader.PaitonAllocatorDelete(self.allocator_handle)
                self.allocator_handle = ctypes.c_void_p()

            self.memloader.close()

    def __getstate__(self):
        return {"lib_path": self.memloader.lib_path}

    def __setstate__(self, d):
        if "lib_path" not in d:
            raise RuntimeError(f"Didn't find 'lib_path' property in {d}")
        self.__init__(d["lib_path"])

    def _convert_single_param_to_c_format(self, param: PData) -> _CFormatPData:
        pointer, shape, dtype = param
        c_pointer = ctypes.c_void_p(pointer)
        c_shape_data = (ctypes.c_longlong * len(shape))()
        for j, dim in enumerate(shape):
            c_shape_data[j] = ctypes.c_longlong(dim)
        c_shape = _PaitonShape(c_shape_data, ctypes.c_size_t(len(shape)))
        c_dtype = dtype_str_to_enum(dtype)
        return _CFormatPData(c_pointer, c_shape, c_dtype)

    def _convert_params_to_c_format(self, params: List[PData]):
        c_params = (_CFormatPData * len(params))()
        for i, param in enumerate(params):
            c_params[i] = self._convert_single_param_to_c_format(param)
        return c_params

    def _prepare_run(
        self,
        inputs,
        outputs,
        stream_ptr,
    ):
        c_inputs = self._convert_params_to_c_format(inputs)
        c_outputs = self._convert_params_to_c_format(outputs)
        c_stream = (
            ctypes.c_void_p() if stream_ptr is None else ctypes.c_void_p(stream_ptr)
        )

        num_outputs = len(self._output_ndims)
        c_output_shapes_out = (ctypes.POINTER(ctypes.c_int64) * num_outputs)()
        for i in range(num_outputs):
            c_output_shapes_out[i] = ctypes.cast(
                (ctypes.c_int64 * self._output_ndims[i])(),
                ctypes.POINTER(ctypes.c_int64),
            )

        return (
            c_inputs,
            c_outputs,
            c_stream,
            c_output_shapes_out,
        )

    def _dict_to_ordered_list(self, params, is_inputs):
        if is_inputs:
            index_map = self._input_name_to_index
        else:
            index_map = self._output_name_to_index
        if len(params) != len(index_map):
            raise ValueError(
                f"Did not get correct number of {'inputs' if is_inputs else 'outputs'} expected {len(index_map)}, got {len(params)}"
            )

        result = [None] * len(index_map)
        for name, tensor in params.items():
            if name not in index_map:
                raise ValueError(
                    f"Got unexpected {'input' if is_inputs else 'output'}: {name}"
                )

            result[index_map[name]] = tensor

        return result

    def _write_tensors_for_standalone_testcase(
        self,
        tensor_dict: Dict[str, TorchTensor],
        file_handle,
        is_inputs: bool = True,
    ) -> None:
        if is_inputs:
            index_map = self._input_name_to_index
        else:
            index_map = self._output_name_to_index
        result = [None] * len(index_map)
        for name, tensor in tensor_dict.items():
            if name not in index_map:
                raise ValueError(
                    f"Got unexpected {'input' if is_inputs else 'output'}: {name}"
                )
            idx = index_map[name]
            result[idx] = tensor
        for tensor in result:
            write_tensor_binary(tensor, file_handle)

    def write_standalone_testcase_data(
        self,
        filename,
        inputs: Dict[str, TorchTensor],
        expected_outputs: List[TorchTensor],
        atol=1e-2,
        rtol=1e-2,
    ):
        with open(filename, "wb") as file_handle:
            file_handle.write(struct.pack("ff", atol, rtol))
            self._write_tensors_for_standalone_testcase(
                tensor_dict=inputs, file_handle=file_handle
            )
            for out in expected_outputs:
                write_tensor_binary(out, file_handle)

    def _make_paiton_outputs(
        self, outputs: List[PData], c_output_shapes
    ) -> Dict[str, PData]:
        output_shapes = []
        for i, c_shape in enumerate(c_output_shapes):
            shape = []
            for j in range(self._output_ndims[i]):
                shape.append(c_shape[j])
            output_shapes.append(shape)

        return {
            name: PData(outputs[idx].data_ptr, output_shapes[idx], outputs[idx].dtype)
            for name, idx in self._output_name_to_index.items()
        }

    def _run_impl(
        self,
        inputs: Union[Dict[str, PData], List[PData]],
        outputs: Union[Dict[str, PData], List[PData]],
        stream_ptr: Optional[int] = None,
        sync: bool = True,
        graph_mode: bool = False,
        outputs_on_host: bool = False,
    ) -> Dict[str, PData]:
        if isinstance(inputs, dict):
            inputs = self._dict_to_ordered_list(inputs, is_inputs=True)
        if isinstance(outputs, dict):
            outputs = self._dict_to_ordered_list(outputs, is_inputs=False)
        (
            c_inputs,
            c_outputs,
            c_stream,
            c_output_shapes_out,
        ) = self._prepare_run(
            inputs,
            outputs,
            stream_ptr,
        )

        if not outputs_on_host:
            self.memloader.PaitonModelContainerRun(
                self.handle,
                c_inputs,
                ctypes.c_size_t(len(inputs)),
                c_outputs,
                ctypes.c_size_t(len(outputs)),
                c_stream,
                ctypes.c_bool(sync),
                ctypes.c_bool(graph_mode),
                c_output_shapes_out,
            )
        else:
            self.memloader.PaitonModelContainerRunWithOutputsOnHost(
                self.handle,
                c_inputs,
                ctypes.c_size_t(len(inputs)),
                c_outputs,
                ctypes.c_size_t(len(outputs)),
                c_stream,
                ctypes.c_bool(graph_mode),
                c_output_shapes_out,
            )

        return self._make_paiton_outputs(outputs, c_output_shapes_out)

    def run(
        self,
        inputs: Union[Dict[str, PData], List[PData]],
        outputs: Union[Dict[str, PData], List[PData]],
        stream_ptr: Optional[int] = None,
        sync: bool = True,
        graph_mode: bool = False,
    ) -> Dict[str, PData]:
        """
        Run the model.

        Parameters
        ----------
        inputs: Union[Dict[str, PData], List[PData]]
            The inputs to use. PData is a named tuple containing
            the tensor's data_ptr, size, and dtype. If inputs is a list,
            it must be ordered correctly (as specified by GetInputNameToIndexMap).
            This parameter can also be a dictionary (name -> PData).
        outputs: Union[Dict[str, PData], List[PData]]
            The outputs to use. Similar to inputs, can either be a list of ordered
            outputs, or a dictionary (output name -> PData).
            These should be allocated with enough memory to store their maximum
            size (which can be queried with GetOutputMaximumSize).
        stream_ptr: int
            A pointer to HIP stream to run on. If None, use the legacy stream.
        sync: bool:
            If True, synchronize the stream at the end of the run
        graph_mode: bool
            If True, use a CUDA graph kernel (experimental)

        Returns
        -------
        PDatas with output shapes that are computed by shape inference. This may not be
        the maximum shape. The output memory blobs that are passed in to Run()
        should be interpreted and possibly truncated according to these sizes.
        """
        return self._run_impl(
            inputs, outputs, stream_ptr, sync, graph_mode, outputs_on_host=False
        )

    def bind_inputs(
        self,
        inputs: Union[Dict[str, PData], List[PData]],
    ) -> None:
        """Persistently bind inputs at the C level.

        Deep-copies the PData array (ptrs + shapes + dtypes) into the C
        ModelContainer so the caller can free its own copy. Subsequent steps
        only need ``update_input_pointers`` + ``run_bound``, avoiding the
        per-step ctypes struct rebuild for hundreds of inputs.
        """
        if isinstance(inputs, dict):
            inputs = self._dict_to_ordered_list(inputs, is_inputs=True)
        c_inputs = self._convert_params_to_c_format(inputs)
        self.memloader.PaitonModelContainerBindInputs(
            self.handle,
            c_inputs,
            ctypes.c_size_t(len(inputs)),
        )
        # Rebinding may replace the C-side input array, so do not reuse the
        # pointer-array allocation from a previous binding.
        self._bound_update_ptr_array = None

    def update_input_pointers(
        self,
        ptrs: Union[List[int], "torch.Tensor"],
    ) -> None:
        """Update only the data pointers of bound inputs (per-step fast path).

        ``ptrs`` must be in the same input-index order as ``bind_inputs``.
        This is a single flat ``void*[]`` copy vs rebuilding hundreds of
        _CFormatPData structs + shape arrays per step.
        """
        n = len(ptrs)
        cached = getattr(self, "_bound_update_ptr_array", None)
        if cached is None or cached[0] != n:
            c_ptrs = (ctypes.c_void_p * n)()
            self._bound_update_ptr_array = (n, c_ptrs)
        else:
            c_ptrs = cached[1]
        for i, p in enumerate(ptrs):
            c_ptrs[i] = ctypes.c_void_p(int(p))
        self.memloader.PaitonModelContainerUpdateInputPointers(
            self.handle,
            c_ptrs,
            ctypes.c_size_t(n),
        )

    def run_bound(
        self,
        outputs: Union[Dict[str, PData], List[PData]],
        stream_ptr: Optional[int] = None,
        sync: bool = True,
        graph_mode: bool = False,
    ) -> Dict[str, PData]:
        """Run inference using the bound inputs (no inputs array needed)."""
        if isinstance(outputs, dict):
            outputs = self._dict_to_ordered_list(outputs, is_inputs=False)
        c_outputs = self._convert_params_to_c_format(outputs)
        c_stream = (
            ctypes.c_void_p() if stream_ptr is None else ctypes.c_void_p(stream_ptr)
        )
        num_outputs = len(self._output_ndims)
        c_output_shapes_out = (ctypes.POINTER(ctypes.c_int64) * num_outputs)()
        for i in range(num_outputs):
            c_output_shapes_out[i] = ctypes.cast(
                (ctypes.c_int64 * self._output_ndims[i])(),
                ctypes.POINTER(ctypes.c_int64),
            )
        self.memloader.PaitonModelContainerRunBound(
            self.handle,
            c_outputs,
            ctypes.c_size_t(len(outputs)),
            c_stream,
            ctypes.c_bool(sync),
            ctypes.c_bool(graph_mode),
            c_output_shapes_out,
        )
        return self._make_paiton_outputs(outputs, c_output_shapes_out)

    def profile(
        self,
        inputs: Union[Dict[str, PData], List[PData]],
        outputs: Union[Dict[str, PData], List[PData]],
        num_iters: int,
        filename: str,
        stream_ptr: Optional[int] = None,
    ) -> None:
        if isinstance(inputs, dict):
            inputs = self._dict_to_ordered_list(inputs, is_inputs=True)
        if isinstance(outputs, dict):
            outputs = self._dict_to_ordered_list(outputs, is_inputs=False)
        (
            c_inputs,
            c_outputs,
            c_stream,
            c_output_shapes_out,
        ) = self._prepare_run(
            inputs,
            outputs,
            stream_ptr,
        )
        self.memloader.PaitonModelContainerProfile(
            self.handle,
            c_inputs,
            ctypes.c_size_t(len(inputs)),
            c_outputs,
            ctypes.c_size_t(len(outputs)),
            c_stream,
            ctypes.c_size_t(num_iters),
            ctypes.c_char_p(filename.encode("utf-8")),
        )

    def profile_with_tensors(
        self,
        inputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        outputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        num_iters: int,
        filename: str,
        stream_ptr: Optional[int] = None,
    ) -> None:
        _check_tensors_contiguous_and_on_gpu(
            inputs,
            name="inputs",
        )
        _check_tensors_contiguous_and_on_gpu(
            outputs,
            name="outputs",
        )
        self.profile(
            _convert_tensor_args(inputs),
            _convert_tensor_args(outputs),
            num_iters,
            filename,
            stream_ptr,
        )

    def _interpret_tensors_as_shapes(
        self,
        outputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        outputs_paiton: Dict[str, PData],
    ) -> Dict[str, TorchTensor]:
        if isinstance(outputs, dict):
            return {
                name: _reshape_tensor(tensor, outputs_paiton[name].shape)
                for name, tensor in outputs.items()
            }
        else:
            return {
                name: _reshape_tensor(outputs[idx], outputs_paiton[name].shape)
                for name, idx in self._output_name_to_index.items()
            }

    def run_with_tensors(
        self,
        inputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        outputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        stream_ptr: Optional[int] = None,
        sync: bool = True,
        graph_mode: bool = False,
    ) -> Dict[str, TorchTensor]:
        """
        Run the model with torch.Tensors. See Run() for information about the
        arguments.

        Inputs may either be a dictionary (name -> torch.Tensor), or a list
        of torch.Tensors ordered according to GetInputNameToIndexMap. Outputs
        can also be a dictionary, or a list ordered according to GetOutputNameToIndexMap.
        """

        _check_tensors_contiguous_and_on_gpu(
            inputs,
            name="inputs",
        )
        _check_tensors_contiguous_and_on_gpu(
            outputs,
            name="outputs",
        )
        outputs_paiton = self.run(
            _convert_tensor_args(inputs),
            _convert_tensor_args(outputs),
            stream_ptr=stream_ptr,
            sync=sync,
            graph_mode=graph_mode,
        )

        return self._interpret_tensors_as_shapes(outputs, outputs_paiton)

    def _run_with_outputs_on_host(
        self,
        inputs: Union[Dict[str, PData], List[PData]],
        outputs: Union[Dict[str, PData], List[PData]],
        stream_ptr: Optional[int] = None,
        graph_mode: bool = False,
    ) -> Dict[str, PData]:
        """
        Like Run(), but takes host memory outputs. Note that there is no sync parameter;
        the stream will always be synchronized after copying the outputs to the host.

        Warning: don't use this! It's not optimal with respect to performance.
        It's here for use if you need it for debugging purpose.
        """
        return self._run_impl(
            inputs, outputs, stream_ptr, graph_mode=graph_mode, outputs_on_host=True
        )

    def _run_with_tensors_outputs_on_host(
        self,
        inputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        outputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        stream_ptr: Optional[int] = None,
        graph_mode: bool = False,
    ) -> Dict[str, TorchTensor]:
        """
        Like RunWithTensors(), but takes host memory tensors

        Warning: don't use this! It's not optimal with respect to performance.
        It's here for use if you need it for debugging.
        """
        _check_tensors_contiguous_and_on_gpu(
            inputs,
            name="inputs",
        )
        _check_tensors_contiguous_and_on_host(
            outputs,
            name="outputs",
        )
        output_shapes = self._run_with_outputs_on_host(
            _convert_tensor_args(inputs),
            _convert_tensor_args(outputs),
            stream_ptr=stream_ptr,
            graph_mode=graph_mode,
        )
        return self._interpret_tensors_as_shapes(outputs, output_shapes)

    def benchmark(
        self,
        inputs: Union[Dict[str, PData], List[PData]],
        outputs: Union[Dict[str, PData], List[PData]],
        stream_ptr: Optional[int] = None,
        graph_mode: bool = False,
        count: int = 10,
        repeat: int = 1,
        num_threads: int = 1,
        use_unique_stream_per_thread: bool = False,
    ) -> Tuple[float, float, Dict[str, PData]]:
        """
        Benchmark the model. See run() for information on most parameters.
        """
        if isinstance(inputs, dict):
            inputs = self._dict_to_ordered_list(inputs, is_inputs=True)
        if isinstance(outputs, dict):
            outputs = self._dict_to_ordered_list(outputs, is_inputs=False)
        (
            c_inputs,
            c_outputs,
            c_stream,
            c_output_shapes_out,
        ) = self._prepare_run(
            inputs,
            outputs,
            stream_ptr,
        )
        time_ms = []
        runtime_ms = ctypes.c_float()
        for _ in range(repeat):
            self.memloader.PaitonModelContainerBenchmark(
                self.handle,
                c_inputs,
                ctypes.c_size_t(len(inputs)),
                c_outputs,
                ctypes.c_size_t(len(outputs)),
                c_stream,
                ctypes.c_bool(graph_mode),
                ctypes.c_size_t(count),
                ctypes.c_size_t(num_threads),
                ctypes.c_bool(use_unique_stream_per_thread),
                ctypes.byref(runtime_ms),
                c_output_shapes_out,
            )
            time_ms.append(runtime_ms.value)
        mean = np.mean(time_ms)
        std = np.std(time_ms)

        return (mean, std, self._make_paiton_outputs(outputs, c_output_shapes_out))

    def benchmark_with_tensors(
        self,
        inputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        outputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        stream_ptr: Optional[int] = None,
        graph_mode: bool = False,
        count: int = 10,
        repeat: int = 1,
        num_threads: int = 1,
        use_unique_stream_per_thread: bool = False,
    ) -> Tuple[float, float, Dict[str, TorchTensor]]:
        """
        Benchmark the model. See run_with_tensors() for information on most parameters.
        """

        _check_tensors_contiguous_and_on_gpu(
            inputs,
            name="inputs",
        )
        _check_tensors_contiguous_and_on_gpu(
            outputs,
            name="outputs",
        )

        mean, std, paiton_outputs = self.benchmark(
            _convert_tensor_args(inputs),
            _convert_tensor_args(outputs),
            stream_ptr,
            graph_mode,
            count,
            repeat,
            num_threads,
            use_unique_stream_per_thread,
        )
        return (mean, std, self._interpret_tensors_as_shapes(outputs, paiton_outputs))

    def _get_map_helper(self, n: int, get_name_func) -> Dict[str, int]:
        result = {}
        for i in range(n):
            c_name = ctypes.c_char_p()
            c_idx = ctypes.c_size_t(i)
            get_name_func(c_idx, ctypes.byref(c_name))
            name = c_name.value.decode("utf-8")
            result[name] = i
        return result

    def _construct_input_name_to_index_map(self) -> Dict[str, int]:
        num_inputs = ctypes.c_size_t()
        self.memloader.PaitonModelContainerGetNumInputs(
            self.handle, ctypes.byref(num_inputs)
        )
        get_input_name = (
            lambda idx, name: self.memloader.PaitonModelContainerGetInputName(
                self.handle, idx, name
            )
        )
        return self._get_map_helper(num_inputs.value, get_input_name)

    def get_input_name_to_index_map(self) -> Dict[str, int]:
        """
        Get the name to index mapping. Note that the ordering of inputs
        is not guaranteed to be deterministic.

        If using run()'s list interface, this ordering must be used!
        """
        # Copy so people can't modify our version of the map
        return self._input_name_to_index.copy()

    def get_input_maximum_shape(self, input_idx_or_name: Union[int, str]) -> List[int]:
        """
        Get the maximum input shape. The input here can either be an input name
        or an index. The index is the runtime's internal index.
        """
        if isinstance(input_idx_or_name, int):
            input_idx = input_idx_or_name
        elif isinstance(input_idx_or_name, str):
            if input_idx_or_name not in self._input_name_to_index:
                raise ValueError(
                    f"Name {input_idx_or_name} not in InputNameToIndexMap! Available names: {list(self._input_name_to_index.keys())}"
                )
            input_idx = self._input_name_to_index[input_idx_or_name]
        else:
            raise TypeError(
                f"input_idx_or_name must be str or int, but got {type(input_idx_or_name)}"
            )

        class Shape(ctypes.Structure):
            _fields_ = [
                ("shape_data", ctypes.POINTER(ctypes.c_longlong)),
                ("size", ctypes.c_size_t),
            ]

        raw_shape = Shape()
        self.memloader.PaitonModelContainerGetMaximumInputShape(
            self.handle, input_idx, ctypes.byref(raw_shape)
        )
        return [raw_shape.shape_data[idx] for idx in range(raw_shape.size)]

    def _construct_output_name_to_index_map(self) -> Dict[str, int]:
        num_outputs = ctypes.c_size_t()
        self.memloader.PaitonModelContainerGetNumOutputs(
            self.handle, ctypes.byref(num_outputs)
        )
        get_output_name = (
            lambda idx, name: self.memloader.PaitonModelContainerGetOutputName(
                self.handle, idx, name
            )
        )
        return self._get_map_helper(num_outputs.value, get_output_name)

    def get_output_name_to_index_map(self) -> Dict[str, int]:
        """
        Get the name to index mapping. Unlike inputs, outputs
        have a guaranteed ordering; the order that outputs were
        provided to `compile_model` is always used as the internal
        name to index mapping.

        If using run()'s list interface, this ordering must be used!
        """
        # Copy so people can't modify our version of the map
        return self._output_name_to_index.copy()

    def set_constant(self, name: str, tensor: PData):
        """
        Set a constant. All constants must have values before calling run().

        Note that the pointer inside tensor must be valid for the entire
        duration of run().
        """
        b_name = name.encode("utf-8")
        c_name = ctypes.c_char_p(b_name)
        c_tensor = self._convert_single_param_to_c_format(tensor)
        self.memloader.PaitonModelContainerSetConstant(
            self.handle, c_name, ctypes.byref(c_tensor)
        )

    def set_many_constants(self, tensors: Dict[str, PData]):
        """
        Bulk set many constants at once. More efficient than set_constant()
        since it only has to acquire the lock once.
        """
        c_names = (ctypes.c_char_p * len(tensors))()
        c_tensors = (_CFormatPData * len(tensors))()
        paiton_tensors = dict()
        for name, tensor in tensors.items():
            try:
                paiton_tensors[name.encode("utf-8")] = self._convert_single_param_to_c_format(tensor)
            except Exception as e:
                raise ValueError(f"Failed to create paiton tensor '{name}' from '{tensor}': {e}")

        for i, (name_bytes, tensor) in enumerate(paiton_tensors.items()):
            c_names[i] = ctypes.c_char_p(name_bytes)
            c_tensors[i] = tensor

        num_tensors = ctypes.c_size_t(len(tensors))
        self.memloader.PaitonModelContainerSetManyConstants(
            self.handle, c_names, c_tensors, num_tensors
        )

    def set_double_buffer_constant(
        self, name: str, tensor: PData, stream_ptr: Optional[int] = None
    ):
        """
        Set a constant. All constants must have values before calling run().

        Note that the pointer inside tensor must be valid for the entire
        duration of run().
        """
        b_name = name.encode("utf-8")
        c_name = ctypes.c_char_p(b_name)
        c_tensor = self._convert_single_param_to_c_format(tensor)
        self.memloader.PaitonModelContainerSetDoubleBufferConstant(
            self.handle, ctypes.c_void_p(stream_ptr), c_name, ctypes.byref(c_tensor)
        )

    def set_many_double_buffer_constants(
        self, tensors: Dict[str, PData], stream_ptr: Optional[int] = None
    ):
        """
        Bulk set many constants at once. More efficient than set_constant()
        since it only has to acquire the lock once.
        """
        c_names = (ctypes.c_char_p * len(tensors))()
        c_tensors = (_CFormatPData * len(tensors))()
        paiton_tensors = {
            name.encode("utf-8"): self._convert_single_param_to_c_format(tensor)
            for name, tensor in tensors.items()
        }
        for i, (name_bytes, tensor) in enumerate(paiton_tensors.items()):
            c_names[i] = ctypes.c_char_p(name_bytes)
            c_tensors[i] = tensor

        num_tensors = ctypes.c_size_t(len(tensors))
        self.memloader.PaitonModelContainerSetManyDoubleBufferConstants(
            self.handle, ctypes.c_void_p(stream_ptr), c_names, c_tensors, num_tensors
        )

    def set_many_constants_with_tensors(self, tensors: Dict[str, TorchTensor]):
        paiton_tensors = {}
        for name, tensor in tensors.items():
            if not tensor.is_contiguous() or not tensor.is_cuda:
                raise ValueError(f"Constant {name} must be contiguous and on the GPU.")
            self.torch_constant_tensors[name] = tensor
            paiton_tensors[name] = torch_to_paiton_data(tensor)
        self.set_many_constants(paiton_tensors)

    def set_double_buffer_constant_with_tensor(
        self, name: str, tensor: TorchTensor, stream_ptr: Optional[int] = None
    ):
        """
        Set a constant with a PyTorch tensor.
        Model will store a reference to the given tensor in
        torch_constant_tensors until it is explicitly deleted or replaced.
        """
        if not tensor.is_contiguous() or not tensor.is_cuda:
            raise ValueError(f"Constant {name} must be contiguous and on the GPU.")
        self.torch_constant_tensors[name] = tensor
        self.set_double_buffer_constant(name, torch_to_paiton_data(tensor), stream_ptr)

    def set_many_double_buffer_constants_with_tensors(
        self, tensors: Dict[str, TorchTensor], stream_ptr: Optional[int] = None
    ):
        paiton_tensors = {}
        for name, tensor in tensors.items():
            if not tensor.is_contiguous() or not tensor.is_cuda:
                raise ValueError(f"Constant {name} must be contiguous and on the GPU.")
            self.torch_constant_tensors[name] = tensor
            paiton_tensors[name] = torch_to_paiton_data(tensor)
        self.set_many_double_buffer_constants(paiton_tensors, stream_ptr)

    def set_constant_with_tensor(self, name: str, tensor: TorchTensor):
        """
        Set a constant with a PyTorch tensor.
        Model will store a reference to the given tensor in
        torch_constant_tensors until it is explicitly deleted or replaced.
        """
        if not tensor.is_contiguous() or not tensor.is_cuda:
            raise ValueError(f"Constant {name} must be contiguous and on the GPU.")
        self.torch_constant_tensors[name] = tensor
        self.set_constant(name, torch_to_paiton_data(tensor))

    def get_output_maximum_shape(
        self, output_idx_or_name: Union[int, str]
    ) -> List[int]:
        """
        Get the maximum output shape. The input here can either be an output name
        or an index. The index is the runtime's internal index (as specified by
        GetOutputNameToIndexMap)
        """
        if isinstance(output_idx_or_name, int):
            output_idx = output_idx_or_name
        elif isinstance(output_idx_or_name, str):
            if output_idx_or_name not in self._output_name_to_index:
                raise ValueError(
                    f"Name {output_idx_or_name} not in OutputNameToIndexMap! Available names: {list(self._output_name_to_index.keys())}"
                )
            output_idx = self._output_name_to_index[output_idx_or_name]
        else:
            raise TypeError(
                f"output_idx_or_name must be str or int, but got {type(output_idx_or_name)}"
            )

        class Shape(ctypes.Structure):
            _fields_ = [
                ("shape_data", ctypes.POINTER(ctypes.c_longlong)),
                ("size", ctypes.c_size_t),
            ]

        raw_shape = Shape()
        self.memloader.PaitonModelContainerGetMaximumOutputShape(
            self.handle, output_idx, ctypes.byref(raw_shape)
        )
        return [raw_shape.shape_data[idx] for idx in range(raw_shape.size)]

    def get_output_dtype(self, index):
        """
        Get the expected dtype of an output.
        """
        output = ctypes.c_int()
        self.memloader.PaitonModelContainerGetOutputDtype(
            self.handle, index, ctypes.byref(output)
        )
        return output.value

    def allocate_gpu_memory(
        self, nbytes: int, stream_ptr: Optional[int] = None, sync: bool = True
    ) -> int:
        """
        Helper function for allocating memory on the GPU. Can be useful if
        third-party libraries like PyTorch or pycuda are not available.

        The pointer returned by this function must be freed by free_gpu_memory
        to avoid memory leaks.
        """
        ptr = ctypes.c_void_p()
        self.memloader.PaitonDeviceMalloc(
            ctypes.byref(ptr),
            ctypes.c_size_t(nbytes),
            ctypes.c_void_p(stream_ptr),
            ctypes.c_bool(sync),
        )
        return ptr.value

    def free_gpu_memory(
        self, ptr: int, stream_ptr: Optional[int] = None, sync: bool = True
    ) -> None:
        """
        Helper function for freeing memory on the GPU. Can be useful if
        third-party libraries like PyTorch or pycuda are not available.
        """
        if ptr in self._allocated_paiton_data:
            self._allocated_paiton_data.remove(ptr)

        self.memloader.PaitonDeviceFree(
            ctypes.c_void_p(ptr), ctypes.c_void_p(stream_ptr), ctypes.c_bool(sync)
        )

    def memcpy(
        self,
        dst: int,
        src: int,
        count: int,
        kind: PaitonMemcpyKind,
        stream_ptr: Optional[int] = None,
        sync: bool = True,
    ) -> None:
        """
        Helper function for copying memory on the GPU. Can be useful if
        third-party libraries like PyTorch or pycuda are not available.

        Supports D2H, H2D, and D2D copies. The copy direction can be
        specified by the PaitonMemcpyKind enum.
        """
        self.memloader.PaitonMemcpy(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            ctypes.c_size_t(count),
            ctypes.c_int(kind.value),
            ctypes.c_void_p(stream_ptr),
            ctypes.c_bool(sync),
        )

    def get_num_runtimes(self) -> int:
        """
        Get the number of runtimes this model container stores.
        """
        out = ctypes.c_size_t()
        self.memloader.PaitonModelContainerGetNumRuntimes(
            self.handle, ctypes.byref(out)
        )
        return out.value

    def numpy_to_paiton_data(
        self, arr: np.ndarray, stream_ptr: Optional[int] = None, sync: bool = True
    ) -> PData:
        """
        Convert a numpy array to Paiton-usable data. Mallocs and copies
        on the given stream.

        The allocated buffer should be manually freed with free_gpu_memory.
        But, in case of misuse, Model will keep track of pointers allocated with
        this method and free them all at the end.
        """
        dtype = str(arr.dtype)
        shape = list(arr.shape)
        gpu_mem = self.allocate_gpu_memory(arr.nbytes, stream_ptr=stream_ptr, sync=sync)
        self._allocated_paiton_data.add(gpu_mem)
        self.memcpy(
            gpu_mem,
            arr.ctypes._data.value,
            arr.nbytes,
            PaitonMemcpyKind.HostToDevice,
            sync=sync,
            stream_ptr=stream_ptr,
        )
        return PData(gpu_mem, shape, dtype)

    def paiton_data_to_numpy(
        self,
        paiton_data: PData,
        stream_ptr: Optional[int] = None,
        sync: bool = True,
    ) -> np.ndarray:
        """
        Create numpy array from an PData.
        Copies on the given stream.
        """
        arr = np.empty(paiton_data.shape, dtype=paiton_data.dtype)
        self.memcpy(
            arr.ctypes._data.value,
            paiton_data.data_ptr,
            arr.nbytes,
            PaitonMemcpyKind.DeviceToHost,
            sync=sync,
            stream_ptr=stream_ptr,
        )
        return arr

    def fold_constants(
        self,
        stream_ptr: Optional[int] = None,
        sync: bool = True,
        double_buffer: bool = False,
    ):
        if double_buffer:
            self.memloader.PaitonModelContainerFoldConstantsInDoubleBuffer(
                self.handle,
                ctypes.c_void_p(stream_ptr),
                ctypes.c_bool(sync),
            )
        else:
            self.memloader.PaitonModelContainerFoldConstants(
                self.handle,
                ctypes.c_void_p(stream_ptr),
                ctypes.c_bool(sync),
            )

    def swap_constants(self):
        self.memloader.PaitonModelContainerSwapConstants(self.handle)

    def _get_constant_names_impl(
        self, unbound_constants_only: bool, constant_folding_only: bool
    ) -> List[str]:
        num_constants = ctypes.c_size_t()
        constant_folding_inputs_only = ctypes.c_bool(constant_folding_only)
        unbound_constants_only_ = ctypes.c_bool(unbound_constants_only)
        self.memloader.PaitonModelContainerGetNumConstants(
            self.handle,
            unbound_constants_only_,
            constant_folding_inputs_only,
            ctypes.byref(num_constants),
        )
        names = (ctypes.c_char_p * num_constants.value)()
        self.memloader.PaitonModelContainerGetConstantNames(
            self.handle, unbound_constants_only_, constant_folding_inputs_only, names
        )
        return [name.decode("utf-8") for name in names]

    def get_constant_names(
        self, unbound_constants_only: bool = True, constant_folding_only: bool = False
    ) -> List[str]:
        return self._get_constant_names_impl(
            unbound_constants_only, constant_folding_only
        )

    def get_constant_folding_input_names(
        self, unbound_constants_only: bool = True
    ) -> List[str]:
        return self._get_constant_names_impl(unbound_constants_only, True)
