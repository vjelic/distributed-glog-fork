from ctypes import *

import os
import logging
import subprocess
import uuid


# Use ROCm installation path if running from standard installation
# With File Reorg rsmiBindings.py will be installed in  /opt/rocm/libexec/rocm_smi.
# relative path changed accordingly
path_librocm = os.path.dirname(os.path.realpath(__file__)) + '/../../lib/librocm_smi64.so'
if not os.path.isfile(path_librocm):
    print('Unable to find %s . Trying /opt/rocm*' % path_librocm)
    for root, dirs, files in os.walk('/opt', followlinks=True):
        if 'librocm_smi64.so' in files:
            path_librocm = os.path.join(os.path.realpath(root), 'librocm_smi64.so')
    if os.path.isfile(path_librocm):
        print('Using lib from %s' % path_librocm)
    else:
        print('Unable to find librocm_smi64.so')

try:
    cdll.LoadLibrary(path_librocm)
    rocm = CDLL(path_librocm)
except OSError:
    print('Unable to load the rocm_smi library.\n'\
          'Set LD_LIBRARY_PATH to the folder containing librocm_smi64.\n'\
          '{0}Please refer to https://github.com/'\
          'RadeonOpenCompute/rocm_smi_lib for the installation guide.{1}'\
          .format('\33[33m', '\033[0m'))
    exit()


ASMI_MAX_BUFFER_LENGTH = 256

class rsmi_status_t(c_int):
    RSMI_STATUS_SUCCESS = 0x0
    RSMI_STATUS_INVALID_ARGS = 0x1
    RSMI_STATUS_NOT_SUPPORTED = 0x2
    RSMI_STATUS_FILE_ERROR = 0x3
    RSMI_STATUS_PERMISSION = 0x4
    RSMI_STATUS_OUT_OF_RESOURCES = 0x5
    RSMI_STATUS_INTERNAL_EXCEPTION = 0x6
    RSMI_STATUS_INPUT_OUT_OF_BOUNDS = 0x7
    RSMI_STATUS_INIT_ERROR = 0x8
    RSMI_INITIALIZATION_ERROR = RSMI_STATUS_INIT_ERROR
    RSMI_STATUS_NOT_YET_IMPLEMENTED = 0x9
    RSMI_STATUS_NOT_FOUND = 0xA
    RSMI_STATUS_INSUFFICIENT_SIZE = 0xB
    RSMI_STATUS_INTERRUPT = 0xC
    RSMI_STATUS_UNEXPECTED_SIZE = 0xD
    RSMI_STATUS_NO_DATA = 0xE
    RSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF


#Dictionary of rsmi ret codes and it's verbose output
rsmi_status_verbose_err_out = {
    rsmi_status_t.RSMI_STATUS_SUCCESS: 'Operation was successful',
    rsmi_status_t.RSMI_STATUS_INVALID_ARGS: 'Invalid arguments provided',
    rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED: 'Not supported on the given system',
    rsmi_status_t.RSMI_STATUS_FILE_ERROR: 'Problem accessing a file',
    rsmi_status_t.RSMI_STATUS_PERMISSION: 'Permission denied',
    rsmi_status_t.RSMI_STATUS_OUT_OF_RESOURCES: 'Unable to acquire memory or other resource',
    rsmi_status_t.RSMI_STATUS_INTERNAL_EXCEPTION: 'An internal exception was caught',
    rsmi_status_t.RSMI_STATUS_INPUT_OUT_OF_BOUNDS: 'Provided input is out of allowable or safe range',
    rsmi_status_t.RSMI_INITIALIZATION_ERROR: 'Error occured during rsmi initialization',
    rsmi_status_t.RSMI_STATUS_NOT_YET_IMPLEMENTED: 'Requested function is not implemented on this setup',
    rsmi_status_t.RSMI_STATUS_NOT_FOUND: 'Item searched for but not found',
    rsmi_status_t.RSMI_STATUS_INSUFFICIENT_SIZE: 'Insufficient resources available',
    rsmi_status_t.RSMI_STATUS_INTERRUPT: 'Interrupt occured during execution',
    rsmi_status_t.RSMI_STATUS_UNEXPECTED_SIZE: 'Unexpected amount of data read',
    rsmi_status_t.RSMI_STATUS_NO_DATA: 'No data found for the given input',
    rsmi_status_t.RSMI_STATUS_UNKNOWN_ERROR: 'Unknown error occured'
}


class rsmi_init_flags_t(c_int):
    RSMI_INIT_FLAG_ALL_GPUS = 0x1


class rsmi_process_info_t(Structure):
    _fields_ = [('process_id', c_uint32),
                ('pasid', c_uint32),
                ('vram_usage', c_uint64),
                ('sdma_usage', c_uint64),
                ('cu_occupancy', c_uint32)]


class rsmi_memory_type_t(c_int):
    RSMI_MEM_TYPE_FIRST = 0
    RSMI_MEM_TYPE_VRAM = RSMI_MEM_TYPE_FIRST
    RSMI_MEM_TYPE_VIS_VRAM = 1
    RSMI_MEM_TYPE_GTT = 2
    RSMI_MEM_TYPE_LAST = RSMI_MEM_TYPE_GTT


# memory_type_l includes names for with rsmi_memory_type_t
# Usage example to get corresponding names:
# memory_type_l[rsmi_memory_type_t.RSMI_MEM_TYPE_VRAM] will return string 'vram'
memory_type_l = ['VRAM', 'VIS_VRAM', 'GTT']


class rsmi_utilization_counter_type(c_int):
    RSMI_UTILIZATION_COUNTER_FIRST = 0
    RSMI_COARSE_GRAIN_GFX_ACTIVITY  = RSMI_UTILIZATION_COUNTER_FIRST
    RSMI_COARSE_GRAIN_MEM_ACTIVITY = 1
    RSMI_UTILIZATION_COUNTER_LAST = RSMI_COARSE_GRAIN_MEM_ACTIVITY

utilization_counter_name = ['GFX Activity', 'Memory Activity']


def _driver_initialized():
    """ Returns true if amdgpu is found in the list of initialized modules
    """
    driverInitialized = ''
    try:
        driverInitialized = str(subprocess.check_output("cat /sys/module/amdgpu/initstate |grep live", shell=True))
    except subprocess.CalledProcessError:
        pass
    return len(driverInitialized) > 0


def asmi_initialize():
    """Initialize ROCm binding of SMI"""
    if _driver_initialized():
        ret_init = rocm.rsmi_init(0)
        if ret_init != 0:
            logging.error(f'ROCm SMI init returned value {ret_init}')
            raise RuntimeError('ROCm SMI initialization failed')
    else:
        raise RuntimeError('ROCm driver initilization failed')


def asmi_get_device_count():
    num_device = c_uint32(0)
    ret = rocm.rsmi_num_monitor_devices(byref(num_device))
    return num_device.value if rsmi_ret_ok(ret) else -1


def asmi_get_handle_by_index(idx):
    # for now, just return index, not device handle struct as in NVML
    return idx


def asmi_get_index_by_handle(handle):
    # handle is same as index
    return handle


def rsmi_ret_ok(my_ret):
    """ Returns true if RSMI call status is 0 (success)

    @param device: DRM device identifier
    @param my_ret: Return of RSMI call (rocm_smi_lib API)
    @param metric: Parameter of GPU currently being analyzed
    """
    if my_ret != rsmi_status_t.RSMI_STATUS_SUCCESS:
        err_str = c_char_p()
        rocm.rsmi_status_string(my_ret, byref(err_str))
        logging.error(err_str.value.decode())
        return False
    return True


# def asmi_get_device_uuid(handle):
#     return uuid.uuid4()


def asmi_get_device_uid(handle):
    idx = asmi_get_index_by_handle(handle)
    uid = c_uint64()
    ret = rocm.rsmi_dev_unique_id_get(idx, byref(uid))
    return uid.value if rsmi_ret_ok(ret) else -1


def asmi_get_device_compute_processes(handle):
    num_procs = c_uint32()
    ret = rocm.rsmi_compute_process_info_get(None, byref(num_procs))
    if rsmi_ret_ok(ret):
        buff_sz = num_procs.value + 10
        proc_info = (rsmi_process_info_t * buff_sz)()
        ret2 = rocm.rsmi_compute_process_info_get(byref(proc_info), byref(num_procs))
        return [proc_info[p].process_id for p in range(num_procs.value)] if rsmi_ret_ok(ret2) else []


def asmi_get_device_name(handle):
    idx = asmi_get_index_by_handle(handle)
    series = create_string_buffer(ASMI_MAX_BUFFER_LENGTH)
    ret = rocm.rsmi_dev_name_get(idx, series, ASMI_MAX_BUFFER_LENGTH)
    return series.value.decode() if rsmi_ret_ok(ret) else ''


def asmi_get_device_memory_total(handle, type='VRAM'):
    dev_idx = asmi_get_index_by_handle(handle)
    type_idx = memory_type_l.index(type)
    total = c_uint64()
    ret = rocm.rsmi_dev_memory_total_get(dev_idx, type_idx, byref(total))
    return total.value if rsmi_ret_ok(ret) else -1


def asmi_get_device_memory_used(handle, type='VRAM'):
    dev_idx = asmi_get_index_by_handle(handle)
    type_idx = memory_type_l.index(type)
    used = c_uint64()
    ret = rocm.rsmi_dev_memory_usage_get(dev_idx, type_idx, byref(used))
    return used.value if rsmi_ret_ok(ret) else -1


def asmi_get_device_utilization(handle):
    dev_idx = asmi_get_index_by_handle(handle)
    timestamp = c_uint64(0)
    length = 1
    util_counters = (rsmi_utilization_counter_type * length)()
    util_counters[0].type = c_int(0)  # this is to set the type = 'GFX Activity'
    ret = rocm.rsmi_utilization_count_get(dev_idx, length, byref(timestamp))
    return (util_counters[0].val, timestamp) if rsmi_ret_ok(ret) else None


def asmi_shut_down():
    rsmi_ret_ok(rocm.rsmi_shut_down())
    return
