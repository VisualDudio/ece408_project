#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <cstdio>
#define TILE_WIDTH 16
#define MASK_SIZE_ONE 12 * 1 * 5 * 5
#define MASK_SIZE_TWO 24 * 12 * 5 * 5
#define MASK_WIDTH 5
#define MAX_FILTER 12
namespace mxnet
{
  namespace op
  {
    __constant__ float weightsOne[12][1][5][5];
    __constant__ float weightsTwo[24][12][5][5];


    __global__ void forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
    {
      /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
      */

      const int H_out = H - K + 1;
      const int W_out = W - K + 1;

      int X_tile_width = TILE_WIDTH + K - 1;
      extern __shared__ float shmem[];
      float* X_shared = &shmem[0];

      int W_grid = ceil((float)W_out / TILE_WIDTH);
      int h_base = blockIdx.z / W_grid * TILE_WIDTH;
      int w_base = blockIdx.z % W_grid * TILE_WIDTH;
      int b = blockIdx.x;
      int m = blockIdx.y;
      int h0 = threadIdx.y;
      int w0 = threadIdx.x;
      int h = h_base + h0;
      int w = w_base + w0;
      float acc = 0.0f;
      
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
      
      for(int c = 0; c < C; c++) {
        // Load tile into shared memory
        if (h < H && w < W) {
          X_shared[((h - h_base) * X_tile_width) + (w - w_base)] = x4d(b, c, h, w);
        } else {
          X_shared[((h - h_base) * X_tile_width) + (w - w_base)] = 0;
        }
        __syncthreads();

        for(int p = 0; p < K; p++) {
          for(int q = 0; q < K; q++) {
            // Access correct constant memory weights array
            acc += x4d(b,c,h+p,w+q) * weightsOne[m][c][p][q];
          }
        }
      }  
    
      if (b < B && m < M && h < H_out && w < W_out)
        y4d(b, m, h, w) = acc;

#undef y4d
#undef x4d
#undef k4d
    }

__global__ void forward_kernel2(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
    {
      /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
      */

      const int H_out = H - K + 1;
      const int W_out = W - K + 1;

      int X_tile_width = TILE_WIDTH + K - 1;
      extern __shared__ float shmem[];
      float* X_shared = &shmem[0];

      int W_grid = ceil((float)W_out / TILE_WIDTH);
      int h_base = blockIdx.z / W_grid * TILE_WIDTH;
      int w_base = blockIdx.z % W_grid * TILE_WIDTH;
      int b = blockIdx.x;
      int m = blockIdx.y;
      int h0 = threadIdx.y;
      int w0 = threadIdx.x;
      int h = h_base + h0;
      int w = w_base + w0;
      float acc = 0.0f;
      
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
      
      for(int c = 0; c < C; c++) {
        // Load tile into shared memory
        if (h < H && w < W) {
          X_shared[((h - h_base) * X_tile_width) + (w - w_base)] = x4d(b, c, h, w);
        } else {
          X_shared[((h - h_base) * X_tile_width) + (w - w_base)] = 0;
        }
        __syncthreads();

        for(int p = 0; p < K; p++) {
          for(int q = 0; q < K; q++) {
            // Access correct constant memory weights array
            acc += x4d(b,c,h+p,w+q) * weightsTwo[m][c][p][q];
          }
        }
      }  
    
      if (b < B && m < M && h < H_out && w < W_out)
        y4d(b, m, h, w) = acc;

#undef y4d
#undef x4d
#undef k4d
    }

    /* 
       This function is called by new-inl.h
       Any code you write should be executed by this function.
       For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
    */
    template <>
      void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
    {

      // Use mxnet's CHECK_EQ to do assertions.
      // Remove this assertion when you do your implementation!
      // Extract the tensor dimensions into B,M,C,H,W,K
      // ...
      const int B = x.shape_[0];
      const int M = y.shape_[1];
      const int C = x.shape_[1];
      const int H = x.shape_[2];
      const int W = x.shape_[3];
      const int K = w.shape_[3];

      const int H_out = H - K + 1;
      const int W_out = W - K + 1;
      int W_grid = ceil((float)W_out / TILE_WIDTH);
      int H_grid = ceil((float)H_out / TILE_WIDTH);
      // Set the kernel dimensions
      dim3 gridDim(B, M, H_grid * W_grid);
      dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
      
      // Optimization 1: Convolution tiling with shared memory
      // Calculate shared memory size
      size_t shmem_size = sizeof(float) * (TILE_WIDTH + K-1)*(TILE_WIDTH + K-1);

      // Optimization 2: Weights in constant memory
      // Put weights into constant memory
      if (w.shape_[0] == 12) {
        cudaMemcpyToSymbol(weightsOne, w.dptr_, MASK_SIZE_ONE * sizeof(float));
        // Call the kernel
        forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);
      } else {
        cudaMemcpyToSymbol(weightsTwo, w.dptr_, MASK_SIZE_TWO * sizeof(float));
        // Call the kernel
        forward_kernel2<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);
      }
      // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
      MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }

    /* 
       This tells mxnet how to do an op when it's not a float.
       This is not used in the ECE408 project
    */
    template <typename gpu, typename DType>
      void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
    {
      CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
    }
  }
}

#endif