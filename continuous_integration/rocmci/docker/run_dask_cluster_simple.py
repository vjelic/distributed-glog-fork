import dask.array as da
import cupy

from distributed import Client
from dask_hip import LocalHIPCluster
from time import time


def run_multi_gpu(device_array):
    t0 = time()
    (device_array + 1)[::2, ::2].sum().compute()
    t1 = time()
    print(f'{t1 - t0:.4f}')


def main():
    # prep random Dask array in **device** memory
    rs_d = da.random.RandomState(RandomState=cupy.random.RandomState)

    # actual allocation of Dask array on device memory
    x_d = rs_d.normal(10,
                      1,
                      size=(50_000, 50_000),
                      chunks=(1_000, 1_000))

    print('\nProcessing smaller cupy array...')
    with LocalHIPCluster(HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7") as cluster:
        with Client(cluster) as client:
            run_multi_gpu(x_d)

    # bigger array
    x_d2 = rs_d.normal(10,
                       1,
                       size=(400_000, 400_000),    # array size 64X
                       chunks=(40_000, 40_000))    # total chunks: 10 x 10 = 100

    print('\nProcessing larger cupy array...')
    with LocalHIPCluster(HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7") as cluster:
        with Client(cluster) as client:
            run_multi_gpu(x_d2)

    print('\nCompleted!')


if __name__ == '__main__':
    main()
