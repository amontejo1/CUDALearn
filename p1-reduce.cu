#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>  
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

int cuda_sum(int *h_arr, int arrSize){
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
    free(h_blockSum);

    return h_blockSum[0];
}

__global__ void max_helper(int * arr, int n, int * maxes){
    __shared__ int warp_vals[32];
    int local_val = INT_MIN;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < n; i += stride){
        local_val = max(local_val, arr[i]);
    }

    int lane = threadIdx.x %32;
    int warp_id = threadIdx.x / 32;

    for(int offset = 16; offset > 0; offset >>= 1){
        local_val = max(local_val, __shfl_down_sync(0xffffffff, local_val, offset));
    }

    if(lane == 0){
        warp_vals[warp_id] = local_val;
    }

    __syncthreads();

    int block_sum = 0;
    int warps_per_block = blockDim.x / warpSize;

    if (warp_id == 0){
        if(lane < warps_per_block){
            block_sum = warp_vals[lane];
        }else{
            block_sum = INT_MIN;
        }

        for(int offset = 16; offset > 0; offset >>= 1 ){
            block_sum = max(block_sum,
                __shfl_down_sync(0xffffffff, block_sum, offset));
        }

        if(lane == 0){
            maxes[blockIdx.x] = block_sum;
        }
    }
}

__global__ void max_helper2(int * block_maxes, int n){
    __shared__ int warp_sums[32];

    int tid = threadIdx.x;
    int local_val = INT_MIN;

    if(tid < n)
        local_val = block_maxes[tid];

    int lane = tid %32;
    int warp_id = tid / 32;

    for(int offset = 16; offset > 0; offset >>= 1){
        local_val = max(local_val, __shfl_down_sync(0xffffffff, local_val, offset));
    }

    if(lane == 0){
        warp_sums[warp_id] = local_val;
    }

    __syncthreads();

    if(warp_id == 0){
        local_val = (lane < (n + 31)/ 32) ? warp_sums[lane] : INT_MIN;

        for(int offset = 16; offset > 0; offset >>=1){
            local_val = max(local_val, __shfl_down_sync(0xffffffff, local_val, offset));
        }

        if(lane == 0){
            block_maxes[0] = local_val;
        }
    }

}

int cuda_max(int * arr, int arrSize){
    int min = INT_MIN;
    int numThreads = 256;
    int size = arrSize * sizeof(int);
    int blocks = (arrSize + numThreads -1) / numThreads;
    int blockSize = blocks * sizeof(int);

    int *h_blockMaxes = (int*)malloc(blockSize);
    int *d_arr, *d_blockMaxes;

    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_blockMaxes, blockSize);

    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
    
    max_helper<<<blocks, numThreads>>>(d_arr, arrSize, d_blockMaxes);

    cudaDeviceSynchronize();

    max_helper2<<<1, numThreads>>>(d_blockMaxes, blocks);

    cudaMemcpy(h_blockMaxes, d_blockMaxes, blockSize, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_blockMaxes);
    free(h_blockMaxes);

    return h_blockMaxes[0];

}

struct valIdx{
    int val;
    int idx;
};

__global__ void argmax_helper(int * arr, int n, struct valIdx * maxes){
    __shared__ valIdx pairs[32];

    int local_max = INT_MIN;
    int local_max_idx = -1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < n; i += stride){
        if (arr[i] > local_max){
            local_max = arr[i];
            local_max_idx = i;
        }
    }

    int lane = threadIdx.x %32;
    int warp_id = threadIdx.x / 32;

    for(int offset = 16; offset > 0; offset >>= 1){
        int other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_max_idx, offset);

        if(other_val > local_max){
            local_max = other_val;
            local_max_idx = other_idx;
        }else if(other_val == local_max){
            local_max_idx = min(local_max_idx, other_idx);
        }

    }

    if(lane == 0){
        pairs[warp_id].val = local_max;
        pairs[warp_id].idx = local_max_idx;
    }

    __syncthreads();

    int block_max = INT_MIN;
    int block_max_idx = -1;
    int warps_per_block = (blockDim.x + warpSize -1) / warpSize;

    if(warp_id == 0){
        if(lane < warps_per_block){
            block_max = pairs[lane].val;
            block_max_idx = pairs[lane].idx;
        }else{
            block_max = INT_MIN;
            block_max_idx = -1;
        }

        for(int offset = 16; offset >0; offset >>= 1){
            int other_block_max = __shfl_down_sync(0xffffffff, block_max, offset);
            int other_block_max_idx = __shfl_down_sync(0xffffffff, block_max_idx, offset);

            if (other_block_max > block_max){
                block_max = other_block_max;
                block_max_idx = other_block_max_idx;
            }else if(other_block_max == block_max){
                block_max_idx = min(block_max_idx, other_block_max_idx);
            }
        }

        if(lane == 0){
            maxes[blockIdx.x].val = block_max;
            maxes[blockIdx.x].idx = block_max_idx;
        }
    }

}

__global__ void argmax_helper2(struct valIdx * maxes, int n){
    __shared__ valIdx warp_maxes[32];

    int tid = threadIdx.x;
    int local_max = INT_MIN;
    int local_max_idx = -1;

    if(tid < n){
        local_max = maxes[tid].val;
        local_max_idx = maxes[tid].idx;
    }

    int lane = tid % 32;
    int warp_id = tid / 32;

    for(int offset = 16; offset > 0; offset >>= 1){
        int other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_max_idx, offset);

        if(other_val > local_max){
            local_max = other_val;
            local_max_idx = other_idx;
        }else if(other_val == local_max){
            local_max_idx = min(local_max_idx, other_idx);
        }

    }

    if (lane == 0){
        warp_maxes[warp_id].val = local_max;
        warp_maxes[warp_id].idx = local_max_idx;
    }

    __syncthreads();

    int overallMax = INT_MIN;
    int overallMaxIdx = -1;
    int warps_per_block = (blockDim.x + warpSize -1) / warpSize;

    if(warp_id == 0){
        if(lane < warps_per_block){
            overallMax = warp_maxes[lane].val;
            overallMaxIdx = warp_maxes[lane].idx;
        }else{
            overallMax = INT_MIN;
            overallMaxIdx = -1;
        }
        for(int offset = 16; offset > 0; offset >>= 1){
            int other_val = __shfl_down_sync(0xffffffff, overallMax, offset);
            int other_idx = __shfl_down_sync(0xffffffff, overallMaxIdx, offset);

            if(other_val > overallMax){
                overallMax = other_val;
                overallMaxIdx = other_idx;
            }else if(other_val == overallMax){
                overallMaxIdx = min(overallMaxIdx, other_idx);
            }

        }
        if (lane == 0){
            maxes[0].val = overallMax;
            maxes[0].idx = overallMaxIdx;
        }


    }

}


int main(){

}
