#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define BLOCK_SIZE 256

// Assume blockDim.x <= 1024
__global__ void sum_helper(int * arr, int n, int* sums){
    __shared__ int warp_sums[32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_sum = 0;

    for(int i = idx; i < n; i += stride){
        local_sum += arr[i];
    }

    int lane = threadIdx.x %32;
    int warp_id = threadIdx.x / 32;

    for(int offset = 16; offset > 0; offset >>= 1){
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if(lane == 0){
        warp_sums[warp_id] = local_sum;
    }

    __syncthreads();

    // Now we coalesce the warps to their per block sums

    int block_sum = 0;
    int warps_per_block = blockDim.x / warpSize;

    if (warp_id == 0){
        if(lane < warps_per_block){
            block_sum = warp_sums[lane];
        }else{
            block_sum = 0;
        }

        for(int offset = 16; offset > 0; offset >>= 1 ){
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }

        if(lane == 0){
            sums[blockIdx.x] = block_sum;
        }
    }
}

__global__ void sum_helper2(int * block_sums, int n){
    __shared__ int warp_sums[32];

    int tid = threadIdx.x;
    int local_sum = 0;

    if(tid < n)
        local_sum = block_sums[tid];

    int lane = tid %32;
    int warp_id = tid / 32;

    for(int offset = 16; offset > 0; offset >>= 1){
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if(lane == 0){
        warp_sums[warp_id] = local_sum;
    }

    __syncthreads();

    if(warp_id == 0){
        local_sum = (lane < (n + 31)/ 32) ? warp_sums[lane] : 0;

        for(int offset = 16; offset > 0; offset >>=1){
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset); 
        }

        if(lane == 0){
            block_sums[0] = local_sum;
        }
    }

}

int sum(int *h_arr, int arrSize){
    int numThreads = 256;
    int size = arrSize * sizeof(int);
    int blocks = (arrSize + numThreads - 1)/numThreads;
    int blockSize = blocks*sizeof(int);

    int *h_blockSum = (int*)malloc(blocks * sizeof(int));

    int *d_arr, *d_blockSum;

    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_blockSum, blockSize);

    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    sum_helper<<<blocks, numThreads>>>(d_arr, arrSize, d_blockSum);

    cudaDeviceSynchronize();

    sum_helper2<<<1, numThreads>>>(d_blockSum, blocks);

    cudaMemcpy(h_blockSum, d_blockSum, blockSize, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_blockSum);

    return h_blockSum[0];
}


int main(){

}
