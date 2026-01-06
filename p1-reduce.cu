#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>  
#include <span>
#include <vector>
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

    cudaDeviceSynchronize();

    cudaMemcpy(h_blockMaxes, d_blockMaxes, blockSize, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_blockMaxes);
    int res = h_blockMaxes[0];
    free(h_blockMaxes);

    return res;

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

__global__ void argmax_reduce(const valIdx* in, int n, valIdx* out) {
    __shared__ valIdx warp_maxes[32];

    int tid   = threadIdx.x;
    int lane  = tid & 31;            // tid % 32
    int warp  = tid / 5;            // tid / 32

    // Gridstride over the input valIdx array
    int start  = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    int local_max     = INT_MIN;
    int local_max_idx = -1;

    for (int i = start; i < n; i += stride) {
        int v = in[i].val;
        int idx = in[i].idx;

        if (v > local_max) {
            local_max = v;
            local_max_idx = idx;
        } else if (v == local_max) {
            local_max_idx = min(local_max_idx, idx);
        }
    }

    // Warp reduce (val, idx) with tie break on smallest idx
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_max_idx, offset);

        if (other_val > local_max) {
            local_max = other_val;
            local_max_idx = other_idx;
        } else if (other_val == local_max) {
            local_max_idx = min(local_max_idx, other_idx);
        }
    }

    if (lane == 0) {
        warp_maxes[warp].val = local_max;
        warp_maxes[warp].idx = local_max_idx;
    }

    __syncthreads();

    int warps_per_block = (blockDim.x + 31) / 32;

    int block_max = INT_MIN;
    int block_max_idx = -1;

    if (warp == 0) {
        if (lane < warps_per_block) {
            block_max     = warp_maxes[lane].val;
            block_max_idx = warp_maxes[lane].idx;
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            int other_val = __shfl_down_sync(0xffffffff, block_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, block_max_idx, offset);

            if (other_val > block_max) {
                block_max = other_val;
                block_max_idx = other_idx;
            } else if (other_val == block_max) {
                block_max_idx = min(block_max_idx, other_idx);
            }
        }

        if (lane == 0) {
            out[blockIdx.x].val = block_max;
            out[blockIdx.x].idx = block_max_idx;
        }
    }
}

int cuda_argmax(int * arr, int arrSize){
    int numThreads = 256;
    int size = arrSize * sizeof(int);
    int blocks = (arrSize + numThreads -1) / numThreads;
    int blockSize = blocks * sizeof(valIdx);

    valIdx h_out;
    int *d_arr;
    valIdx *d_blockMaxes;

    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_blockMaxes, blockSize);

    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
    
    // reduces each warp and writes one max val and idx per block
    argmax_helper<<<blocks, numThreads>>>(d_arr, arrSize, d_blockMaxes);

    cudaDeviceSynchronize();

    // iteratively reduce each block into global max value and its corresponding idx
    // making it therefore more scalable
    int currSize = blocks;
    valIdx *d_in = d_blockMaxes;
    valIdx *d_out;
    while(currSize > 1){
        int nextBlockSize = (currSize + numThreads-1)/numThreads;
        cudaMalloc(&d_out, nextBlockSize *sizeof(valIdx));

        argmax_reduce<<<nextBlockSize, numThreads>>>(d_in, currSize, d_out);
        cudaDeviceSynchronize();
        cudaFree(d_in);
        d_in = d_out;
        currSize = nextBlockSize;
    }

    cudaMemcpy(&h_out, d_in, sizeof(valIdx), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_in);
    return h_out.idx;

}

template <typename T>
struct TvalIdx{
    T val;
    int idx;
};

template<typename T, typename Operation> 
__global__ void reduce(const T *in, int n, TvalIdx<T> *out, Operation op){
    TvalIdx<T> local;
    int tid = threadIdx.x;
    int start  = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    local.val = -INFINITY;
    local.idx = -1;

    for(int i = start; i < n; i += stride){
        TvalIdx<T> c = {in[i], i};
        local = op(local, c);
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        TvalIdx<T> other;
        other.val = __shfl_down_sync(0xffffffff, local.val, offset);
        other.idx = __shfl_down_sync(0xffffffff, local.idx, offset);
        local = op(local, other);
    }

    __shared__ TvalIdx<T> warp_vals[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    if (lane == 0){
        warp_vals[warp_id] = local;
    }

    __syncthreads();

    if(warp_id == 0){
        local = (lane < blockDim.x/warpSize) // warps per block
            ? warp_vals[lane] : TvalIdx<T>{-INFINITY,-1};

        for (int offset = 16; offset > 0; offset >>= 1) {
            TvalIdx<T> other;
            other.val = __shfl_down_sync(0xffffffff, local.val, offset);
            other.idx = __shfl_down_sync(0xffffffff, local.idx, offset);
            local = op(local,other);
        }
        if(lane == 0){
            out[blockIdx.x] = local;
        }
    }
}

template <typename T>
struct argMax {
    __device__ __forceinline__
    TvalIdx<T> operator()(TvalIdx<T> a, TvalIdx<T> b) const {
        if (a.val > b.val) return a;
        if (b.val > a.val) return b;
        return(a.idx < b.idx) ? a : b; 
    }
};

template <typename T>
struct argMin {
    __device__ __forceinline__
    TvalIdx<T> operator()(TvalIdx<T> a, TvalIdx<T> b) const {
        if (a.val < b.val) return a;
        if (b.val < a.val) return b;
        return(a.idx < b.idx) ? a : b;
    }
};

template<typename T>
int template_argMax(span<T> arr){
    int n = arr.size();
    int numThreads = 256;
    int blocks = (n + numThreads - 1) / numThreads;

    T* d_input;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMemcpy(d_input, arr.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    
    TvalIdx<T>* d_out;
    cudaMalloc(&d_out, blocks * sizeof(TvalIdx<T>));
    
    reduce<<<blocks, numThreads>>>(
        d_input,
        n,
        d_out,
        argMax<T>()
    );

    TvalIdx<T>* curr_in = d_out;
    int curr_n = blocks;
    TvalIdx<T>* curr_out;


    while(curr_n > 1){
        blocks = (n + numThreads - 1) / numThreads;
        cudaMalloc(&curr_out, blocks * sizeof(TvalIdx<T>));
        reduce<<<blocks, numThreads>>>(
            curr_in,
            curr_n,
            curr_out,
            argMax<T>()
        );
        cudaDeviceSynchronize();
        cudaFree(curr_in);
        curr_in = curr_out;
        curr_n = blocks;
    }

    TvalIdx<T> host_out;
    cudaMemcpy(&host_out, curr_in, sizeof(TvalIdx<T>), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_out);
    if(curr_in != d_out){
        cudaFree(curr_in);
    }

    return host_out.idx;

}



int main(){
    return 0;
}
