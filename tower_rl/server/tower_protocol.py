import hashlib

import msgpack
import numpy as np

MAX_BYTES = 64 * 1024 * 1024


class PolicyError(Exception):
    @property
    def category(self):
        return "infrastructure" if str(self) == "endpoint_track_version_mismatch" else "participant"


def _array(value):
    if isinstance(value, np.ndarray):
        if value.dtype not in (np.dtype("uint8"), np.dtype("<f4"), np.dtype("<f8")):
            raise PolicyError("unsupported_array_dtype")
        a = np.ascontiguousarray(value)
        return msgpack.ExtType(
            42, msgpack.packb([a.dtype.str, list(a.shape), a.tobytes()], use_bin_type=True)
        )
    raise TypeError("Unsupported wire type")


def _extension(code, raw):
    if code != 42:
        raise PolicyError("unknown_array_extension")
    dtype, shape, data = msgpack.unpackb(raw, raw=False, max_bin_len=MAX_BYTES)
    if (
        dtype not in ("|u1", "<f4", "<f8")
        or not isinstance(shape, list)
        or not 1 <= len(shape) <= 4
    ):
        raise PolicyError("invalid_array")
    count = 1
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise PolicyError("invalid_array_shape")
        count *= dim
        if count > MAX_BYTES:
            raise PolicyError("array_too_large")
    dt = np.dtype(dtype)
    if not isinstance(data, bytes) or len(data) != count * dt.itemsize or len(data) > MAX_BYTES:
        raise PolicyError("invalid_array_bytes")
    return np.frombuffer(data, dtype=dt).reshape(shape).copy()


def pack(value):
    data = msgpack.packb(value, default=_array, use_bin_type=True)
    if len(data) > MAX_BYTES:
        raise PolicyError("message_too_large")
    return data


def unpack(data):
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise PolicyError("invalid_message")
    try:
        result = msgpack.unpackb(
            data,
            raw=False,
            ext_hook=_extension,
            strict_map_key=True,
            max_bin_len=MAX_BYTES,
            max_array_len=100000,
            max_map_len=1000,
            max_str_len=1024 * 1024,
            max_ext_len=MAX_BYTES,
        )
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise PolicyError("invalid_encoding") from exc
    if not isinstance(result, dict):
        raise PolicyError("message_not_object")
    return result


def sample_seed(base, index):
    return int.from_bytes(
        hashlib.sha256(
            b"tower-seed-stream-v1\0" + base.to_bytes(4, "big") + index.to_bytes(8, "big")
        ).digest()[:8],
        "big",
    )


def observation_spec(track):
    return dict(
        schema=track["observation_schema"],
        arm_order=["left", "back"],
        joint_positions={"shape": [2, 7], "dtype": "<f4", "unit": "rad"},
        gripper_widths_m={"shape": [2], "dtype": "<f4", "unit": "m"},
        cameras={
            k: {"shape": [*v, 3], "dtype": "|u1", "color": "RGB"}
            for k, v in track["cameras"].items()
        },
    )


def action_spec(track):
    return dict(
        schema=track["action_schema"],
        shape=["H", 16],
        dtype="<f4",
        dt_s=track["action_dt_s"],
        arm_order=["left", "back"],
        representation="absolute_joint16",
        gripper_unit="m",
        joint_unit="rad",
    )
