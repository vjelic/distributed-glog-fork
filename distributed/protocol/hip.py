from __future__ import annotations

import dask
from dask.utils import typename

from distributed.protocol import pickle
from distributed.protocol.serialize import (
    ObjectDictSerializer,
    register_serialization_family,
)

hip_serialize = dask.utils.Dispatch("hip_serialize")
hip_deserialize = dask.utils.Dispatch("hip_deserialize")


def hip_dumps(x):
    type_name = typename(type(x))
    try:
        dumps = hip_serialize.dispatch(type(x))
    except TypeError:
        raise NotImplementedError(type_name)

    sub_header, frames = dumps(x)
    header = {
        "sub-header": sub_header,
        "type-serialized": pickle.dumps(type(x)),
        "serializer": "hip",
        "compression": (False,) * len(frames),  # no compression for gpu data
    }
    return header, frames


def hip_loads(header, frames):
    typ = pickle.loads(header["type-serialized"])
    loads = hip_deserialize.dispatch(typ)
    return loads(header["sub-header"], frames)


register_serialization_family("hip", hip_dumps, hip_loads)


hip_object_with_dict_serializer = ObjectDictSerializer("hip")

hip_deserialize.register(dict)(hip_object_with_dict_serializer.deserialize)
