#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <pthread.h>
// extern "C" void caffe_gpu_lu(const Dtype* a, const Dtype* b, int a_nr,  int a_nc, int b_nr, int b_nc, int K, Dtype* c);

// template <typename Dtype>
// extern "C" void caffe_gpu_lu(const Dtype* a, const Dtype* b, int a_nr,  int a_nc, int b_nr, int b_nc, int K, Dtype* c);

namespace caffe {

template <typename Dtype>
void BaseCuriousLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  CuriousParameter curious_param = this->layer_param_.curious_param();
  CHECK(!curious_param.has_kernel_size() !=
      !(curious_param.has_kernel_h() && curious_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(curious_param.has_kernel_size() ||
      (curious_param.has_kernel_h() && curious_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!curious_param.has_pad() && curious_param.has_pad_h()
      && curious_param.has_pad_w())
      || (!curious_param.has_pad_h() && !curious_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!curious_param.has_stride() && curious_param.has_stride_h()
      && curious_param.has_stride_w())
      || (!curious_param.has_stride_h() && !curious_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (curious_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = curious_param.kernel_size();
  } else {
    kernel_h_ = curious_param.kernel_h();
    kernel_w_ = curious_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!curious_param.has_pad_h()) {
    pad_h_ = pad_w_ = curious_param.pad();
  } else {
    pad_h_ = curious_param.pad_h();
    pad_w_ = curious_param.pad_w();
  }
  if (!curious_param.has_stride_h()) {
    stride_h_ = stride_w_ = curious_param.stride();
  } else {
    stride_h_ = curious_param.stride_h();
    stride_w_ = curious_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.curious_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.curious_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.curious_param().bias_term();

  //#############################################
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(3); // book + indicator + lu_table + bias
    } else {
      this->blobs_.resize(2);
    }
    //remain to change
  //#############################################

    //>--------------------------------------------------------------------------------

    Cs = curious_param.cs(); //8
    CHECK_EQ(channels_ % Cs, 0);

    Ct = num_output_;
    M = channels_  / Cs;
    K = curious_param.k(); // 128

    vector<int> book_shape_;
    book_shape_.push_back(M);
    book_shape_.push_back(K);
    book_shape_.push_back(Cs);
    
    vector<int> indicator_shape;
    indicator_shape.push_back(M);
    indicator_shape.push_back(Ct);
    indicator_shape.push_back(kernel_h_*kernel_w_);

    // this->blobs_[0].reset(new Blob<Dtype>(M,Cs,K));// target
    this->blobs_[0].reset(new Blob<Dtype>(book_shape_));// target
    this->blobs_[1].reset(new Blob<Dtype>(indicator_shape));

    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.curious_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  // # no backward
  // this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseCuriousLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.

  // height_out and weight_out are computed here
  compute_output_shape();

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  if (reverse_dimensions()) {
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = height_ * width_;
  } else {
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = height_out_ * width_out_;
  }

  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // sub_output_offset_ = conv_out_spatial_dim_;

  book_offset_  = K * Cs;
  input_offset_ = Cs * height_ * width_;
  lu_table_offset_ = K * height_ * width_;
  lu_dim_ = K * kernel_h_ * kernel_w_;
  indicator_offset_ = Ct *  kernel_h_ * kernel_w_;
  kernel_count = kernel_h_*kernel_w_;

//##################################################

  lu_table.reset(new Blob<Dtype>(M,K,height_,width_)); // target, too

//##################################################

  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
    col_buffer_.Reshape(1, lu_dim_, height_, width_);
  } else {
    col_buffer_.Reshape(1, lu_dim_, height_out_, width_out_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseCuriousLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* quantized_book, const Dtype* quantized_indicator, Dtype* lu_table, Dtype* output, bool skip_im2col) {
  // Dtype is_begin_ = 0.;
  const Dtype* col_buff = input;

// at first generate lookuptable
  // Get the lu_table

  for (int m = 0; m < M; ++m)
  {
    col_buff = input;

    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, K, 
      height_*width_, Cs, (Dtype)1.,quantized_book + m * book_offset_, 
      col_buff + m*input_offset_,(Dtype)0.,lu_table+lu_table_offset_*m);
  }

  for (int m = 0; m < M; ++m)
  {
    col_buff = lu_table + m * lu_table_offset_;
    if (!is_1x1_) {
      curious_im2col_cpu(lu_table + m*lu_table_offset_, col_buffer_.mutable_cpu_data());
      col_buff = col_buffer_.cpu_data();
    }

    const Dtype* indicator_subspace = quantized_indicator + m*indicator_offset_;
    for (int i = 0; i  < Ct; ++i)
    {
      Dtype * start_point = (output + i*conv_out_spatial_dim_);
      // ith row
      const Dtype * start_indicator  = (indicator_subspace + i*kernel_count);
      for (int j = 0; j < kernel_count ; ++j) // one by one
      {
        const Dtype* start_col = col_buff + int(*(start_indicator + j));
        caffe_axpy(conv_out_spatial_dim_, (Dtype)1., start_col ,start_point);
      }
    }

    // int height_col = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    // int width_col = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
    //assume m is fixed
    // for (int c = 0; c < kernel_count; ++c)
    // {
    //   int w_offset = c % kernel_w_;
    //   int h_offset = c / kernel_w_;
    //   const Dtype * indicator_c = indicator_subspace + c*Ct;

    //   for (int h = 0; h < height_col; ++h)
    //     for (int w = 0; w < width_col; ++w)
    //     {
    //       int h_pad = h * stride_h_ - pad_h_ + h_offset;
    //       int w_pad = w * stride_w_ - pad_w_ + w_offset;
    //       if (h_pad >= 0 && h_pad <= height_ && w_pad >= 0 && w_pad <= width_)
    //       {
    //         // what does the book look like
    //         // Ct by Kernel_count
    //         // I can change the kernel_count into Kernel_count by Ct
    //         // assume changed!
    //         for (int k = 0; k < Ct; ++k)
    //         {
    //           output[(k * height_col + h) * width_col + w] += col_buff[int(indicator_c[k]) * conv_out_spatial_dim_ + h_pad * width_ + w_pad];
    //         }
    //       }
    //     }
    // }
  }
}

template <typename Dtype>
void BaseCuriousLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseCuriousLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* quantized_book, const Dtype* quantized_indicator, Dtype* lu_table, Dtype* output, bool skip_im2col) {
  // Dtype is_begin_ = 0.;
  const Dtype* col_buff = input;
  for (int m = 0; m < M; ++m)
  {
    col_buff = input;

    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, K, 
      height_*width_, Cs, (Dtype)1.,quantized_book + m * book_offset_, 
      col_buff + m*input_offset_,(Dtype)0.,lu_table+lu_table_offset_*m);

    col_buff = lu_table + m*lu_dim_;
    if (!is_1x1_) {
      curious_im2col_gpu(lu_table + m*lu_table_offset_, col_buffer_.mutable_gpu_data());
      col_buff = col_buffer_.gpu_data();
    }

    const Dtype* indicator_subspace = quantized_indicator + m*indicator_offset_;

    LOG(INFO)<<"before computation";
    caffe_gpu_lu(indicator_subspace, lu_table, Ct, K, lu_dim_, conv_out_spatial_dim_, K, output);
  }
}


template <typename Dtype>
void BaseCuriousLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseCuriousLayer);
// REGISTER_LAYER_CLASS(BaseCurious); 

}  // namespace caffe
