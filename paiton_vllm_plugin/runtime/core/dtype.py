"""
dtype definitions and utility functions of Paiton
"""


_DTYPE2BYTE = {
    "bool": 1,
    "uint8": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2fnuz": 1,
    "float16": 2,
    "float32": 4,
    "float": 4,
    "int": 4,
    "int32": 4,
    "int64": 8,
    "bfloat16": 2,
}


# Maps dtype strings to PaitonDtype enum in model_interface.h.
# Must be kept in sync!
# We can consider defining an PaitonDtype enum to use on the Python
# side at some point, but stick to strings for now to keep things consistent
# with other Python APIs.
_DTYPE_TO_ENUM = {
    "float16": 1,
    "float32": 2,
    "float": 2,
    "int": 3,
    "int32": 3,
    "int64": 4,
    "bool": 5,
    "bfloat16": 6,
    "float8_e4m3fnuz": 7,  # kFloat8_e4m3
    "float8_e5m2fnuz": 8,  # kFloat8_e5m2
    "uint8": 9,  # kUInt8
}


def get_dtype_size(dtype: str) -> int:
    """Returns size (in bytes) of the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    int
        Size (in bytes) of this dtype.
    """

    if dtype not in _DTYPE2BYTE:
        raise KeyError(f"Unknown dtype: {dtype}. Expected one of {_DTYPE2BYTE.keys()}")
    return _DTYPE2BYTE[dtype]


def normalize_dtype(dtype: str) -> str:
    """Returns a normalized dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    str
        normalized dtype str.
    """
    if dtype == "int":
        return "int32"
    if dtype == "float":
        return "float32"
    return dtype


def dtype_str_to_enum(dtype: str) -> int:
    """Returns the PaitonDtype enum value (defined in model_interface.h) of
    the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    int
        the PaitonDtype enum value.
    """
    if dtype not in _DTYPE_TO_ENUM:
        raise ValueError(
            f"Got unsupported input dtype {dtype}! Supported dtypes are: {list(_DTYPE_TO_ENUM.keys())}"
        )
    return _DTYPE_TO_ENUM[dtype]


def dtype_to_enumerator(dtype: str) -> str:
    """Returns the string representation of the PaitonDtype enum
    (defined in model_interface.h) for the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    str
        the PaitonDtype enum string representation.
    """
    dtype_map = {
        "float16": "kHalf",
        "float32": "kFloat",
        "float": "kFloat",
        "int32": "kInt",
        "int": "kInt",
        "int64": "kLong",
        "bool": "kBool",
        "bfloat16": "kBFloat16",
        "float8_e4m3fnuz": "kFloat8_e4m3fnuz",
        "float8_e5m2fnuz": "kFloat8_e5m2fnuz",
        "uint8": "kUInt8",
    }

    return f"PaitonDtype::{dtype_map[dtype]}"


def is_same_dtype(dtype1: str, dtype2: str) -> bool:
    """Returns True if dtype1 and dtype2 are the same dtype and False otherwise.

    Parameters
    ----------
    dtype1: str
        A data type string.
    dtype2: str
        A data type string.

    Returns
    ----------
    bool
        whether dtype1 and dtype2 are the same dtype
    """
    return normalize_dtype(dtype1) == normalize_dtype(dtype2)
