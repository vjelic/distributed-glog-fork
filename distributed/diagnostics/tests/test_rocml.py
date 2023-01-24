from __future__ import annotations

import multiprocessing as mp
import os

import pytest

pytestmark = pytest.mark.gpu

pyamdsmi = pytest.importorskip('pyamdsmi')
from pyamdsmi import rocml as amdml
import dask

from distributed.diagnostics import rocml
from distributed.utils_test import gen_cluster


@pytest.fixture(autouse=True)
def reset_rocml_state():
    # try:
    #     amdml.asmi_shutdown()
    # except rocml.ROCMLError_Uninitialized:
    #     pass
    rocml.ROCML_STATE = rocml.ROCML_STATE.UNINITIALIZED
    rocml.ROCML_OWNER_PID = None


def test_one_time():
    if rocml.device_get_count() < 1:
        pytest.skip('No GPUs available')

    output = rocml.one_time()
    assert 'memory-total' in output
    assert 'name' in output

    assert len(output['name']) > 0


def test_enable_disable_rocml():
    with dask.config.set({'distributed.diagnostics.rocml': False}):
        rocml.init_once()
        assert not rocml.is_initialized()
        assert rocml.ROCML_STATE == rocml.ROCMLState.DISABLED_CONFIG
    
    # Idempotent (once we've decided not to turn things on with
    # configuration, it's set in stone)
    rocml.init_once()
    assert not rocml.is_initialized()
    assert rocml.ROCML_STATE == rocml.ROCMLState.DISABLED_CONFIG


def run_has_rocm_context(queue):
    try:
        assert not rocml.has_rocm_context().has_context

        ctx = rocml.has_rocm_context()
        assert (
            ctx.has_context
            and ctx.device_info.device_index == 0
            and isinstance(ctx.device_info.uid, int)
        )

        queue.put(None)

    except Exception as e:
        queue.put(e)


@pytest.mark.xfail(reason='If running on Docker, requires --pid=host')
def test_has_rocm_context():
    if rocml.device_get_count() < 1:
        pytest.skip('No GPUs available')

    # This test should be run in a new process so that it definitely doesn't have a 
    # ROCm context and uses a queue to pass exceptions back
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=run_has_rocm_context, args=(queue,))
    p.start()
    p.join()  # this blocks until the process terminates
    e = queue.get()
    if e is not None:
        raise e
    

def test_1_visible_devices():
    if rocml.device_get_count() < 1:
        pytest.skip('No GPUs available')

    os.environ['ROCM_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # in certain cases ROCm uses CUDA-like name which should be temporary
    output = rocml.one_time()
    h = rocml._rocml_handle()
    assert output['memory-total'] == amdml.asmi_get_device_memory_total(h)


@pytest.mark.parametrize('RVD', ['1,0', '0,1'])
def test_2_visible_devices(RVD):
    if rocml.device_get_count() < 2:
        pytest.skip('Less than two GPUs available')

    os.environ['ROCM_VISIBLE_DEVICES'] = RVD
    os.environ['CUDA_VISIBLE_DEVICES'] = RVD
    idx = int(RVD.split(',')[0])

    h = rocml._rocml_handle()
    h2 = rocml.device_get_handle_by_index(idx)

    uid = amdml.asmi_get_device_unique_id(h)
    uid2 = amdml.asmi_get_device_unique_id(h2)

    assert uid == uid2


@gen_cluster()
async def test_gpu_metrics(s, a, b):
    if rocml.device_get_count() < 1:
        pytest.skip('No GPUs available')
    
    h = rocml._rocml_handle()

    assert 'gpu' in a.metrics
    assert (
        s.workers[a.address].metrics['gpu']['memory-used']
        == amdml.asmi_get_device_memory_used(h)
    )
    assert 'gpu' in a.startup_information
    assert (
        s.workers[a.address].extra['gpu']['name']
        == amdml.asmi_get_device_name(h)
    )


@pytest.mark.flaky(reruns=5, reruns_delay=2)
@gen_cluster()
async def test_gpu_monitoring_recent(s, a, b):
    if rocml.device_get_count() < 1:
        pytest.skip('No GPUs available')
    
    h = rocml._rocml_handle()
    res = await s.get_worker_monitor_info(recent=True)

    # ?? equiv of pynvml.nvmlDeviceGetUtilizationRates(h).gpu,  is this correct?
    assert (
        res[a.address]['range_query']['gpu_utilization']
        == amdml.asmi_get_device_utilization(h)
    )
    assert (
        res[a.address]['range_query']['gpu_memory_used']
        == amdml.asmi_get_device_memory_used(h)
    )
    assert res[a.address]['gpu_name'] == amdml.asmi_get_device_name(h)
    assert res[a.address]['gpu_memory_total'] == amdml.asmi_get_device_memory_total(h)


@gen_cluster()
async def test_gpu_monitoring_range_query(s, a, b):
    if rocml.device_get_count() < 1:
        pytest.skip('No GPUs available')
    
    res = await s.get_worker_monitor_info()
    ms = ['gpu_utilization', 'gpu_memory_used']
    for w in (a, b):
        assert all(res[w.address]['range_query'][m] is not None for m in ms)
        assert res[w.address]['count'] is not None
        assert res[w.address]['last_time'] is not None
