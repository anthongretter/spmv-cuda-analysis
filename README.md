# SpMV evaluation
Anthon Porath Gretter - 259030

## Building
Before building, make sure you have **CUDA/11.8.0** installed and loaded.
To build the desired implementation you can simply type
```shell
make spmv_<implementation name>
```
Where <_implementation name_> can be `cpu_csr`, `gpu_mem`, `gpu_unrl` or `gpu_dyn`.
Or even, to build all CPU or/and all GPU implementations¹ just run:
```shell
make cpu
make gpu
```
You can always add additional jobs to make, like `-j8`, to enhance compilation time.

## Usage
To run, provide a valid `.mtx` file alongside the desired implementation call.
There are some `.mtx` included in the **resources** directory². Below sits an example:
```shell
./spmv_gpu_mem ./resources/rim.mtx
```
To run GPU implementations on the DISI cluster, you can enter interactively by:
```shell
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --partition=edu-short --pty /bin/bash
```
And then run the first command on this section. Or the script provided with `./run_job` 

> [1] _the implementations are compiled separately
> due to the usage of Compile-time Conditional Inclusion. 
> So if make does not do that automatically,
> please make sure there are no object files from previous compilations._\
> [2] _All available matrix market files were gathered from https://sparse.tamu.edu/_