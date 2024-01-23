from __future__ import annotations

import os
from enum import IntEnum, auto
from typing import NamedTuple

import dask
try:
    from pyamdsmi import rocml as _rocml
    from hip import hip as hip_binding
except ImportError:
    _rocml = None


class ROCMLState(IntEnum):
    UNINITIALIZED = auto()
    """No attempt yet made to initialize pyamdsmi"""
    INITIALIZED = auto()
    """pyamdsmi was successfully initialized"""
    DISABLED_PYROCML_NOT_AVAILABLE = auto()
    """pyamdsmi not installed"""
    DISABLED_CONFIG = auto()
    """pyamdsmi diagnostics disabled by ``distributed.diagnostics.rocml`` config setting"""
    DISABLED_LIBRARY_NOT_FOUND = auto()
    """pyamdsmi available, but ROCML not installed"""
    # DISABLED_WSL_INSUFFICIENT_DRIVER = auto()
    # """pyamdsmi and ROCML available, but on WSL and the driver version is insufficient"""


class ROCMLError_NotSupported(Exception):
    pass


class ROCMLError_LibraryNotFound(Exception):
    pass


class ROCMLError_DriverNotLoaded(Exception):
    pass


class ROCMLError_Unknown(Exception):
    pass


class ROCMLError_Uninitialized(Exception):
    pass


class RocmDeviceInfo(NamedTuple):
    uuid: int | None = None
    device_index: int | None = None


class RocmContext(NamedTuple):
    has_context: bool
    device_info: RocmDeviceInfo | None = None


# Initialisation must occur per-process, so an initialised state is a
# (state, pid) pair
ROCML_STATE = (
    ROCMLState.DISABLED_PYROCML_NOT_AVAILABLE
    if _rocml is None
    else ROCMLState.UNINITIALIZED
)
"""Current initialization state"""

ROCML_OWNER_PID = None
"""PID of process that successfully called pyamdsmi.rocml.asmi_initialize"""


def is_initialized():
    """Is pyamdsmi initialized on this process?"""
    return ROCML_STATE == ROCMLState.INITIALIZED and ROCML_OWNER_PID == os.getpid()


def init_once():
    """Idempotent (per-process) initialization of PyROCML

    Notes
    -----

    Modifies global variables ROCML_STATE and ROCML_OWNER_PID"""
    global ROCML_STATE, ROCML_OWNER_PID

    if ROCML_STATE in {
        ROCMLState.DISABLED_PYROCML_NOT_AVAILABLE,
        ROCMLState.DISABLED_CONFIG,
        ROCMLState.DISABLED_LIBRARY_NOT_FOUND,
    }:
        return
    elif ROCML_STATE == ROCMLState.INITIALIZED and ROCML_OWNER_PID == os.getpid():
        return
    elif ROCML_STATE == ROCMLState.UNINITIALIZED and not dask.config.get(
        "distributed.diagnostics.rocml"
    ):
        ROCML_STATE = ROCMLState.DISABLED_CONFIG
        return
    elif (
        ROCML_STATE == ROCMLState.INITIALIZED and ROCML_OWNER_PID != os.getpid()
    ) or ROCML_STATE == ROCMLState.UNINITIALIZED:
        try:
            _rocml.asmi_initialize()
        except (
            ROCMLError_LibraryNotFound,
            ROCMLError_DriverNotLoaded,
            ROCMLError_Unknown,
        ):
            ROCML_STATE = ROCMLState.DISABLED_LIBRARY_NOT_FOUND
            return

        if False:
            pass
        else:
            from distributed.worker import add_gpu_metrics

            # initialization was successful
            ROCML_STATE = ROCMLState.INITIALIZED
            ROCML_OWNER_PID = os.getpid()
            add_gpu_metrics()
    else:
        raise RuntimeError(
            f'Unhandled initialisation state ({ROCML_STATE=}, {ROCML_OWNER_PID=})'
        )


def rocml_shut_down():
    _rocml.asmi_shutdown()


def device_get_count():
    init_once()
    if not is_initialized():
        return 0
    else:
        return _rocml.asmi_get_device_count()


def device_get_handle_by_index(idx):
    # for now, just return index, not device handle struct as in NVML
    return idx


def device_get_index_by_handle(handle):
    return handle


def _rocml_handles():
    count = device_get_count()
    if ROCML_STATE == ROCMLState.DISABLED_PYROCML_NOT_AVAILABLE:
        raise RuntimeError('ROCML monitoring requires PyROCML and rocm-smi to be installed')
    elif ROCML_STATE == ROCMLState.DISABLED_LIBRARY_NOT_FOUND:
        raise RuntimeError('PyROCML is installed, but rocm-smi is not')
    elif ROCML_STATE == ROCMLState.DISABLED_CONFIG:
        raise RuntimeError(
            'PyROCML monitoring disabled by \'distributed.diagnostics.rocml\' config setting'
        )
    elif count == 0:
        raise RuntimeError('No GPUs available')
    else:
        try:
            gpu_idx = next(
                map(int, os.environ.get('ROCM_VISIBLE_DEVICES', '').split(','))
            )
        except ValueError:
            # ROCM_VISIBLE_DEVICES is not set, take first device
            gpu_idx = 0
        return device_get_handle_by_index(gpu_idx)


def _running_process_matches(handle):
    init_once()
    comp_procs = _rocml.asmi_get_device_compute_process(handle)

    return any(os.getpid() == pid for pid in comp_procs)


def has_rocm_context():
    """Check whether the current process already has a ROCm context created.

    Returns
    -------
    out : ROCmContext
        Object containing information as to whether the current process has a ROCm
        context created, and in the positive case containing also information about
        the device the context belongs to.
    """
    init_once()
    if is_initialized():
        for index in range(device_get_count()):
            # handle = _rocml.asmi_get_handle_by_index(index)
            handle = index
            if _running_process_matches(handle):
                return get_device_index_and_uuid(handle)

    return RocmContext(has_context=False)


B1 = '%02x'
B2 = B1 * 2
B4 = B1 * 4
B6 = B1 * 6

def get_device_index_and_uuid(device):
    """Get both device index and unique id (long) from device index or uid
       As for now, ROCm smi does not provide uuid, but unique id as long integer.

    Parameters
    ----------
    device: int
        A device index as int, or unique id as long (same as int in Python 3)

    Returns
    -------
    out : RocmDeviceInfo
        A dictionary of 'device-index' and 'uid'
    """
    init_once()
    uuid_str = ''
    device_index = int(device)

    try:
        uuid = hip_binding.hipUUID_t()
        err, uuid = hip_binding.hipDeviceGetUuid(device_index)
        fmt = f'GPU-{B4}-{B2}-{B2}-{B2}-{B6}'
        uuid_str = fmt % tuple(uuid.bytes[:16])
    except ValueError:
        pass

    return RocmDeviceInfo(uuid=uuid_str, device_index=device_index)


def _get_utilization(h):
    try:
        return _rocml.asmi_get_device_utilization(h)
    except ROCMLError_NotSupported:
        return None


def _get_memory_used(h):
    try:
        return _rocml.asmi_get_device_memory_used(h)
    except ROCMLError_NotSupported:
        return None


def _get_memory_total(h):
    try:
        return _rocml.asmi_get_device_memory_total(h)
    except ROCMLError_NotSupported:
        return None


def _get_name(h):
    try:
        return _rocml.asmi_get_device_name(h)
    except ROCMLError_NotSupported:
        return None


def real_time():
    h = _rocml_handles()
    return {
        "utilization": _get_utilization(h),
        "memory-used": _get_memory_used(h),
    }


def one_time():
    h = _rocml_handles()
    return {
        "memory-total": _get_memory_total(h),
        "name": _get_name(h),
    }


if __name__ == '__main__':
    print('ready to initialize...')
    _rocml.asmi_initialize()
    print(f'ngpus = {_rocml.asmi_get_device_count()}')
    _rocml.asmi_shutdown()
