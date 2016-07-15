#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    double* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * patch_h * patch_w + (h - h_col * stride_h) * ksize
            + (w - w_col * stride_w);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset =
        (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);

template <typename Dtype>
__global__ void sparse_im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col, 
    const int * kernel_distri , const int * kernel_asum, const Dtype * kernel_map, 
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = kernel_asum[channel_in];
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;

    Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;


    for (int k = kernel_asum[channel_in]; k < kernel_asum[channel_in]+kernel_distri[channel_in]; ++k)
    {
      int c = kernel_map[k];
      c = c % (kernel_h * kernel_w);
      int i = c/kernel_w;
      int j = c%kernel_w;
      int h = h_in + i;
      int w = w_in + j;
      *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_im_ptr[i * width + j] : 0;
      data_col_ptr += height_col * width_col;
    }
  }
}

template <typename Dtype>
void sparse_im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, int kmap_length, const Dtype * kernel_map,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  int * kernel_distri;

  cudaMalloc((void**) &kernel_distri, sizeof(int) * channels);
  cudaMemset(kernel_distri, 0,  sizeof(Dtype) * channels);

  int * kernel_asum;
  cudaMalloc((void**) &kernel_asum, sizeof(int) * channels);
  //cudaMemset(kernel_distri, 0,  sizeof(Dtype) * channels);

  int kernel_dim = kernel_w * kernel_h;
  int channel_idx = 0;
  int up_limit = kernel_dim;

  for (int i = 0; i < kmap_length; ++i)
  {
    if (kernel_map[i] < up_limit)
    {
      kernel_distri[channel_idx] += 1;
    }
    else
    {
      up_limit += kernel_dim;
      channel_idx += 1;
      while (kernel_map[i] < up_limit)
      {
        up_limit += kernel_dim;
        channel_idx += 1;
      }
      //add in this way, otherwise lose it.
      kernel_distri[channel_idx] += 1;
    }
  }

  kernel_asum[0] = 0;
  for (int i = 1; i < channels; ++i)
  {
    kernel_asum[i] = (kernel_asum[i-1] + kernel_distri[i]);
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  sparse_im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col, width_col,
      kernel_distri, kernel_asum, kernel_map,
       data_col);

  cudaFree(kernel_distri);
  cudaFree(kernel_asum);

  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void sparse_im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    int kmap_length, const float * kernel_map, float* data_col);
template void sparse_im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    int kmap_length, const double * kernel_map, double* data_col);

template <typename Dtype>
__global__ void sparse_col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col, const int* kernel_distri,
    const int* kernel_asum, const Dtype* kernel_map, Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    // equivalent implementation

    int offset = kernel_asum[c] * height_col * width_col;
    // int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    // int coeff_w_col = (1 - stride_w * height_col * width_col);
    int idx = 0;
    int kernel_spatial = patch_w * patch_h;

    for (int h_col = h_col_start; h_col < h_col_end && idx < kernel_distri[c]; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int kernel_loc = (h - h_col * stride_h) * patch_w + w - w_col * stride_w;
        while (kernel_loc > int(kernel_map[kernel_asum[c]+idx])%kernel_spatial)
        {
          ++idx;
        }
        if (idx >= kernel_distri[c])
          break;
      
        if (kernel_loc == int(kernel_map[kernel_asum[c]+idx])%kernel_spatial)
        {
          val += data_col[offset + idx*height_col*width_col + h_col * width_col + w_col];
        }
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void sparse_col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, int kmap_length, const Dtype * kernel_map, Dtype* data_im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  int * kernel_distri;

  cudaMalloc((void**) &kernel_distri, sizeof(int) * channels);
  cudaMemset(kernel_distri, 0,  sizeof(Dtype) * channels);

  int * kernel_asum;
  cudaMalloc((void**) &kernel_asum, sizeof(int) * channels);

  int kernel_dim = patch_w * patch_h;
  int channel_idx = 0;
  int up_limit = kernel_dim;

  for (int i = 0; i < kmap_length; ++i)
  {
    if (kernel_map[i] < up_limit)
    {
      kernel_distri[channel_idx] += 1;
    }
    else
    {
      up_limit += kernel_dim;
      channel_idx += 1;
      while (kernel_map[i] < up_limit)
      {
        up_limit += kernel_dim;
        channel_idx += 1;
      }
      kernel_distri[channel_idx] += 1;
    }
  }

  kernel_asum[0] = 0;
  for (int i = 1; i < channels; ++i)
  {
    kernel_asum[i] = (kernel_asum[i-1] + kernel_distri[i]);
  }

  sparse_col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, kernel_distri, kernel_asum, kernel_map, data_im);

  cudaFree(kernel_distri);
  cudaFree(kernel_asum);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void sparse_col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w,  int kmap_length, const float * kernel_map, float* data_im);
template void sparse_col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w,  int kmap_length, const double* kernel_map, double* data_im);

}  // namespace caffe
