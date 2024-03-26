
from pyrsmi import rocml as _rocml
from hip import hip as hip_binding


class NVMLError_NotSupported(Exception):
    pass


class NVMLError_LibraryNotFound(Exception):
    pass


class NVMLError_DriverNotLoaded(Exception):
    pass


class NVMLError_Unknown(Exception):
    pass


NVML_DEVICE_MIG_DISABLE = 1


def nvmlInit():
    pass


def nvmlSystemGetDriverVersion():
    pass


def nvmlDeviceGetCount():
    pass


def nvmlDeviceGetHandleByIndex(idx: int):
    """return device unique id as handle"""
    pass


def nvmlDeviceGetComputeRunningProcesses(handle):
    pass


def nvmlDeviceGetComputeRunningProcesses_v2(handle):
    pass


def nvmlDeviceGetMigMode(handle):
    throw NVMLError_NotSupported()


def nvmlDeviceGetMaxMigDeviceCount(handle):
    return -1


def nvmlDeviceGetMigDeviceHandleByIndex(handle):
    return -1


def nvmlDeviceGetUUID(handle):
    pass


def nvmlDeviceGetHandleByUUID(uuid):
    pass


def nvmlDeviceGetIndex(handle):
    pass


@dataclass(init=True)
class DeviceUtilization:
    gpu: float = 0.0
    memory: float = 0.0


@dataclass(init=True)
class DeviceMemoryInfo:
    total: int = 0
    used: int = 0


def nvmlDeviceGetUtilizationRates(handle):
    """return device utilization for gpu and memory"""
    gpu = get_gpu_util(handle)
    memory = get_memory_used(handle)

    return DeviceUtilization(gput=gpu, memory=memory)



def nvmlDeviceGetMemoryInfo(handle):
    """return device memory info: total and used"""
    total = get_gpu_mem_total(handle)
    used = get_gpu_mem_used(handle)

    return DeviceMemoryInfo(total=total, used=used)
