#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>  
#include <span>
#include <vector>
using namespace std;

__global__ void rgbtogray(const unsigned char *in, unsigned char *out, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height){
        // [[r,g,b], [r,g,b]] row major order
        int rgb_idx = (y * width + x) * 3;
        
        unsigned char r = in[rgb_idx];
        unsigned char g = in[rgb_idx+1];
        unsigned char b = in[rgb_idx+2];

        out[y * width + x] = static_cast<unsigned char>(0.299f*r + 0.587f*g + 0.114f*b);
    }
}

__global__ void sobel(const unsigned char *in, unsigned char *out, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    auto lambda = [&](int nx, int ny) -> int{
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) return 0;
        return in[ny * width + nx];
    };

    int tl = lambda(x-1, y-1);
    int t = lambda(x, y-1);
    int tr = lambda(x+1, y-1);
    int l = lambda(x-1, y);
    int r = lambda(x+1, y);
    int bl = lambda(x-1, y+1);
    int b = lambda(x, y+1);
    int br = lambda(x+1, y+1);

    int gx = (-1 * tl) + (1 * tr) + 
             ( -2 * l) + (2 * r) +
             (-1 * bl) + (1 * br);

    int gy = (-1 * tl) + (-2 * t) + (-1 * tr) +
             (1 * bl) + (2 * b) + (1 * br);

    int mag = sqrt(gx * gx + gy * gy);

    if (mag > 255) mag = 255;

    out[y * width + x] = (unsigned char) mag;

}