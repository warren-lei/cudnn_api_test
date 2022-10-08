#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include <memory>
#include <assert.h>

#define CUDNN_CHECK(error)                                              \
  do {                                                                  \
    if (error != CUDNN_STATUS_SUCCESS) {                                \
      printf("cuDNN error is : %d %s %d\n", error, __FILE__, __LINE__); \
    }                                                                   \
  } while(0)

#define CUDA_CHECK(error)                                              \
  do {                                                                 \
    if (error != cudaSuccess) {                                        \
      printf("cuda error is : %d %s %d\n", error, __FILE__, __LINE__); \
    }                                                                  \
  } while(0)

void PrintArray(float *array, int size, const char *name) {
  std::cout << name << "\n";
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

void TestFindConvolutionForwardAlgorithm() {
  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));
  assert(handle != nullptr);

  cudnnTensorFormat_t format_x = CUDNN_TENSOR_NHWC;
  cudnnDataType_t data_type_x = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format_y = CUDNN_TENSOR_NHWC;
  cudnnDataType_t data_type_y = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format_w = CUDNN_TENSOR_NHWC;
  cudnnConvolutionMode_t conv_mode = CUDNN_CROSS_CORRELATION;
  //cudnnConvolutionMode_t conv_mode = CUDNN_CONVOLUTION;
  cudnnDataType_t data_type_conv = CUDNN_DATA_FLOAT;

  int dim_x[4] = {1, 3, 3, 4};
  cudnnTensorDescriptor_t x_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  assert(x_desc != nullptr);
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, format_x, data_type_x, dim_x[0],
                                         dim_x[3], dim_x[1], dim_x[2]));

  int in_size = dim_x[0] * dim_x[1] * dim_x[2] * dim_x[3];
  void *in_data;
  cudaMalloc(&in_data, in_size * sizeof(float));
  std::unique_ptr<float[]> input_h(new float[in_size]);
  for (int i = 0; i < in_size; i++) {
    input_h[i] = 1.0f;
  }
  CUDA_CHECK(cudaMemcpy(in_data, input_h.get(), in_size * sizeof(float),
                        cudaMemcpyHostToDevice));

  int dim_w[4] = {1, 4, 2, 2};
  cudnnFilterDescriptor_t filter_desc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
  assert(filter_desc != nullptr);
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, data_type_x, format_w,
                                         dim_w[0], dim_w[1], dim_w[2], dim_w[3]));

  int w_size = dim_w[0] * dim_w[1] * dim_w[2] * dim_w[3];
  void *w_data;
  CUDA_CHECK(cudaMalloc(&w_data, w_size * sizeof(float)));
  std::unique_ptr<float[]> filter_h(new float[w_size]);
  for (int i = 0; i < w_size; i++) {
    filter_h[i] = 1.0f;
  }
  CUDA_CHECK(cudaMemcpy(w_data, filter_h.get(), w_size * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int pad[4] = {0, 0, 0, 0};
  const int stride[4] = {1, 1, 1, 1};
  const int dilation[4] = {1, 1, 1, 1};

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  assert(conv_desc != nullptr);
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad[0], pad[1],
                                              stride[0], stride[1], dilation[0],
                                              dilation[1], conv_mode, data_type_conv));

  int out_n, out_c, out_h, out_w;
  CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, filter_desc,
                                                    &out_n, &out_c, &out_h, &out_w));

  printf("output dims(nchw): %d, %d, %d, %d\n", out_n, out_c, out_h, out_w);
  int dim_y[4] = {out_n, out_c, out_h, out_w};
  cudnnTensorDescriptor_t y_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  assert(y_desc != nullptr);
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, format_y, data_type_y, dim_y[0],
                                         dim_y[1], dim_y[2], dim_y[3]));

  int out_size = out_n * out_c * out_h * out_w;
  void *out_data;
  CUDA_CHECK(cudaMalloc(&out_data, out_size * sizeof(float)));

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  size_t workspace_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, filter_desc,
                                                      conv_desc, y_desc, algo,
                                                      &workspace_size));
  //size_t workspace_size = 1024 * 1024 * 1024;
  void *workspace;
  CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

  const int conv_algo_cnt = 8;
  int cnt = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results[conv_algo_cnt];
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(handle, x_desc, in_data,
                                                     filter_desc, w_data,
                                                     conv_desc, y_desc, out_data,
                                                     conv_algo_cnt, &cnt,
                                                     perf_results, workspace,
                                                     workspace_size));

  assert(cnt > 0);
  std::cout << "cnt: " << cnt << std::endl;
  CUDNN_CHECK(perf_results[0].status);
  std::cout << "perf algo: " << perf_results[0].algo
            << ", time: " << perf_results[0].time
            << ", memory: " << perf_results[0].memory << std::endl;

  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
  CUDNN_CHECK(cudnnDestroy(handle));

  if (workspace_size > 0) CUDA_CHECK(cudaFree(workspace));
}


void TestBNFwdTraining() {
  int dim_x[4] = {1, 3, 3, 4};
  int n = dim_x[0], c = dim_x[3], h = dim_x[1], w = dim_x[2];
  // Generating random input_data
  int size = n * c * h * w;
  int input_data[size];
  for (int i = 0; i < size; i++) {
    input_data[i] = rand() % 255;
  }

  cudnnHandle_t handle_;
  CUDNN_CHECK(cudnnCreate(&handle_));

  // setting parameters for batchnormal API
  auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  const cudnnBatchNormOps_t bn_ops = CUDNN_BATCHNORM_OPS_BN;
  float one = 1.0;
  float zero = 0.0;

  int size_bytes = size * sizeof(float);
  int mean_size = c;
  int mean_size_bytes = mean_size * sizeof(float);

  // create the tensor descriptor
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w));
  cudnnTensorDescriptor_t y_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, format, dtype, n, c, h, w));

  float *x, *y;
  CUDA_CHECK(cudaMalloc(&x, size_bytes));
  CUDA_CHECK(cudaMalloc(&y, size_bytes));

  // initializing data
  CUDA_CHECK(cudaMemcpy(x, input_data, size_bytes, cudaMemcpyHostToDevice));

  float alpha[c] = {1};
  float beta[c] = {0.0};

  cudnnTensorDescriptor_t mean_descriptor;
  cudnnCreateTensorDescriptor(&mean_descriptor);
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(mean_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, 1, c, 1, 1));

  float *scale, *offset;
  float *running_mean, *running_var;
  float *saved_mean, *saved_inv_var;
  cudaMalloc(&scale, mean_size_bytes);
  cudaMalloc(&offset, mean_size_bytes);
  cudaMalloc(&running_mean, mean_size_bytes);
  cudaMalloc(&running_var, mean_size_bytes);
  cudaMalloc(&saved_mean, mean_size_bytes);
  cudaMalloc(&saved_inv_var, mean_size_bytes);

  // initialize scale, offset, running_mean, running_var
  float mean_val[mean_size];
  for (int i = 0; i < mean_size; i++) { mean_val[i] = 1.0f; }

  CUDA_CHECK(cudaMemcpy(scale, mean_val, mean_size_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(offset, mean_val, mean_size_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(running_mean, mean_val, mean_size_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(running_var, mean_val, mean_size_bytes,
                        cudaMemcpyHostToDevice));

  cudnnActivationDescriptor_t activation_desc;
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc));
  CUDNN_CHECK(cudnnSetActivationDescriptor(activation_desc,
                                           CUDNN_ACTIVATION_IDENTITY,
                                           CUDNN_PROPAGATE_NAN, 0.0));

  size_t workspace_size_bytes = 0;
  CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      handle_, mode, bn_ops, x_desc, NULL, y_desc, mean_descriptor,
      activation_desc, &workspace_size_bytes));

  void *workspace = nullptr;
  if (workspace_size_bytes > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size_bytes));
  }

  clock_t start, stop;
  start=clock();

  size_t reserve_space_size_bytes = 0;
  CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      handle_, mode, bn_ops, activation_desc, x_desc, &reserve_space_size_bytes));

  char *reserve_space;
  CUDA_CHECK(cudaMalloc(&reserve_space, reserve_space_size_bytes));

  CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      /*handle=*/handle_,
      /*mode=*/mode,
      /**alpha=*/&one,
      /**beta=*/&zero,
      /*xDesc=*/x_desc,
      /**x=*/x,
      /*yDesc=*/y_desc,
      /**y=*/y,
      /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
      /*bnScaleData=*/scale,
      /*bnBiasData=*/offset,
      /*exponentialAverageFactor=*/0.5,
      /*resultRunningMeanData=*/running_mean,
      /*resultRunningVarianceData=*/running_var,
      /*epsilon=*/0.001,
      /*resultSaveMean=*/saved_mean,
      /*resultSaveInvVariance=*/saved_inv_var));

  stop=clock();
  double flopsCoef = 2.0;
  std::cout << "Input n*c*h*w: " << size <<
      "\nLatancy: " << ((double)(stop - start))/CLOCKS_PER_SEC <<
      "\nThroughput: " << (1e-9 * flopsCoef * size) / (stop - start) << std::endl;

  CUDA_CHECK(cudaDeviceSynchronize());

  float out_h[size] = { 0 };
  CUDA_CHECK(cudaMemcpy(out_h, y, size_bytes, cudaMemcpyDeviceToHost));
  PrintArray(out_h, size, "output: ");

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  CUDA_CHECK(cudaFree(scale));
  CUDA_CHECK(cudaFree(offset));
  CUDA_CHECK(cudaFree(running_mean));
  CUDA_CHECK(cudaFree(running_var));
  CUDA_CHECK(cudaFree(saved_mean));
  CUDA_CHECK(cudaFree(saved_inv_var));
  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFree(reserve_space));

  CUDNN_CHECK(cudnnDestroy(handle_));
}

int main() {
  //TestFindConvolutionForwardAlgorithm();
  TestBNFwdTraining();
}
