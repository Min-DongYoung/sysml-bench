#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__)               \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define CHECK_KERNEL_LAUNCH()                                                \
  do {                                                                       \
    cudaError_t err__ = cudaGetLastError();                                  \
    if (err__ != cudaSuccess) {                                              \
      std::cerr << "Kernel launch error: " << cudaGetErrorString(err__)      \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void vadd_fp32(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

__global__ void vadd_fp16(const __half* a, const __half* b, __half* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = __hadd(a[i], b[i]);
  }
}

__global__ void copy_fp32(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i];
  }
}

__global__ void fma_rpt_fp32(const float* in, float* out, int n, int k, float a, float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float acc = in[i];
    for (int t = 0; t < k; ++t) {
      acc = fmaf(acc, a, b);
    }
    out[i] = acc;
  }
}

struct Args {
  std::string op = "vadd";
  std::string dtype = "fp32";
  int N = 1 << 24;
  int warmup = 10;
  int iters = 50;
  int block = 256;
  int seed = 123;
  int fma_k = 256;
  float fma_a = 1.0001f;
  float fma_b = 0.0001f;
  std::string csv_path;
};

bool parse_args(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const std::string& k) -> bool {
      return k == "--op" || k == "--N" || k == "--warmup" || k == "--iters" ||
             k == "--csv" || k == "--dtype" || k == "--block" || k == "--seed" ||
             k == "--fma-k" || k == "--fma-a" || k == "--fma-b";
    };
    if (need_value(key)) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << key << "\n";
        return false;
      }
      std::string val = argv[++i];
      if (key == "--op") args.op = val;
      else if (key == "--N") args.N = std::atoi(val.c_str());
      else if (key == "--warmup") args.warmup = std::atoi(val.c_str());
      else if (key == "--iters") args.iters = std::atoi(val.c_str());
      else if (key == "--csv") args.csv_path = val;
      else if (key == "--dtype") args.dtype = val;
      else if (key == "--block") args.block = std::atoi(val.c_str());
      else if (key == "--seed") args.seed = std::atoi(val.c_str());
      else if (key == "--fma-k") args.fma_k = std::atoi(val.c_str());
      else if (key == "--fma-a") args.fma_a = static_cast<float>(std::atof(val.c_str()));
      else if (key == "--fma-b") args.fma_b = static_cast<float>(std::atof(val.c_str()));
    } else {
      std::cerr << "Unknown argument: " << key << "\n";
      return false;
    }
  }
  return true;
}

std::string timestamp_utc_iso() {
  using clock = std::chrono::system_clock;
  auto now = clock::now();
  std::time_t t = clock::to_time_t(now);
  std::tm tm_utc;
#if defined(_WIN32)
  gmtime_s(&tm_utc, &t);
#else
  gmtime_r(&t, &tm_utc);
#endif
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday,
                tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec);
  return std::string(buf);
}

std::string timestamp_for_filename() {
  using clock = std::chrono::system_clock;
  auto now = clock::now();
  auto t = clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  std::tm tm_utc;
#if defined(_WIN32)
  gmtime_s(&tm_utc, &t);
#else
  gmtime_r(&t, &tm_utc);
#endif
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d_%03d",
                tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday,
                tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec,
                static_cast<int>(ms.count()));
  return std::string(buf);
}

template <typename T>
void fill_random(std::vector<T>& v, std::mt19937& rng);

template <>
void fill_random<float>(std::vector<float>& v, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) {
    x = dist(rng);
  }
}

template <>
void fill_random<__half>(std::vector<__half>& v, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) {
    x = __float2half(dist(rng));
  }
}

constexpr int kValidateSamples = 1024;

std::vector<int> make_validation_indices(int n) {
  std::vector<int> indices;
  if (n <= 0) {
    return indices;
  }
  int checks = std::min(n, kValidateSamples);
  indices.reserve(checks);
  if (checks <= 2) {
    indices.push_back(0);
    if (n > 1) {
      indices.push_back(n - 1);
    }
    return indices;
  }
  int stride = (n - 1) / (checks - 1);
  if (stride < 1) {
    stride = 1;
  }
  for (int j = 0; j < checks; ++j) {
    int idx = j * stride;
    if (idx > n - 1) {
      idx = n - 1;
    }
    indices.push_back(idx);
  }
  if (n > 1 && !indices.empty()) {
    indices.back() = n - 1;
  }
  return indices;
}

uint32_t float_bits(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

bool validate_fp32(const std::vector<float>& a,
                   const std::vector<float>& b,
                   const std::vector<float>& c,
                   const std::vector<int>& indices) {
  const float tol = 1e-5f;
  if (indices.empty()) return true;
  for (int idx : indices) {
    float expected = a[idx] + b[idx];
    float diff = std::abs(c[idx] - expected);
    if (diff > tol) {
      std::cerr << "Validation failed at idx " << idx
                << " expected " << expected
                << " got " << c[idx]
                << " diff " << diff << "\n";
      return false;
    }
  }
  return true;
}

bool validate_fp16(const std::vector<__half>& a,
                   const std::vector<__half>& b,
                   const std::vector<__half>& c,
                   const std::vector<int>& indices) {
  const float tol = 1e-2f;
  if (indices.empty()) return true;
  for (int idx : indices) {
    float expected = __half2float(a[idx]) + __half2float(b[idx]);
    float got = __half2float(c[idx]);
    float diff = std::abs(got - expected);
    if (diff > tol) {
      std::cerr << "Validation failed at idx " << idx
                << " expected " << expected
                << " got " << got
                << " diff " << diff << "\n";
      return false;
    }
  }
  return true;
}

bool validate_copy_fp32(const std::vector<float>& in,
                        const std::vector<float>& out,
                        const std::vector<int>& indices) {
  if (indices.empty()) return true;
  for (int idx : indices) {
    uint32_t in_bits = float_bits(in[idx]);
    uint32_t out_bits = float_bits(out[idx]);
    if (in_bits != out_bits) {
      std::cerr << "Validation failed at idx " << idx
                << " expected_bits " << in_bits
                << " got_bits " << out_bits << "\n";
      return false;
    }
  }
  return true;
}

bool validate_fma_rpt(const std::vector<float>& in,
                      const std::vector<float>& out,
                      int k,
                      float a,
                      float b,
                      const std::vector<int>& indices) {
  const float atol = 1e-4f;
  const float rtol = 1e-4f;
  if (indices.empty()) return true;
  for (int idx : indices) {
    float acc = in[idx];
    for (int t = 0; t < k; ++t) {
      acc = std::fma(acc, a, b);
    }
    float ref = acc;
    float got = out[idx];
    bool ref_nan = std::isnan(ref);
    bool got_nan = std::isnan(got);
    if (ref_nan || got_nan) {
      if (!(ref_nan && got_nan)) {
        std::cerr << "Validation failed at idx " << idx
                  << " ref_nan " << ref_nan
                  << " got_nan " << got_nan << "\n";
        return false;
      }
      continue;
    }
    bool ref_inf = std::isinf(ref);
    bool got_inf = std::isinf(got);
    if (ref_inf || got_inf) {
      if (!(ref_inf && got_inf && (std::signbit(ref) == std::signbit(got)))) {
        std::cerr << "Validation failed at idx " << idx
                  << " ref " << ref
                  << " got " << got << "\n";
        return false;
      }
      continue;
    }
    float diff = std::abs(got - ref);
    float thresh = atol + rtol * std::abs(ref);
    if (diff > thresh) {
      std::cerr << "Validation failed at idx " << idx
                << " expected " << ref
                << " got " << got
                << " diff " << diff << "\n";
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    return EXIT_FAILURE;
  }

  if (args.op != "vadd" && args.op != "copy" && args.op != "fma_rpt") {
    std::cerr << "Unsupported op: " << args.op << " (vadd|copy|fma_rpt)\n";
    return EXIT_FAILURE;
  }
  if (args.dtype != "fp32" && args.dtype != "fp16") {
    std::cerr << "Unsupported dtype: " << args.dtype << " (fp32|fp16)\n";
    return EXIT_FAILURE;
  }
  if ((args.op == "copy" || args.op == "fma_rpt") && args.dtype != "fp32") {
    std::cerr << "Unsupported dtype for op " << args.op << ": "
              << args.dtype << " (fp32 only)\n";
    return EXIT_FAILURE;
  }
  if (args.N <= 0 || args.block <= 0 || args.iters <= 0) {
    std::cerr << "Invalid N/block/iters\n";
    return EXIT_FAILURE;
  }
  if (args.warmup < 0) {
    std::cerr << "Invalid warmup\n";
    return EXIT_FAILURE;
  }

  int warmup = args.warmup;
  std::string notes;

  if (args.csv_path.empty()) {
    args.csv_path = "results_" + timestamp_for_filename() + ".csv";
  }

  std::vector<int> validate_indices = make_validation_indices(args.N);

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  int driver_version = 0;
  int runtime_version = 0;
  CHECK_CUDA(cudaDriverGetVersion(&driver_version));
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));

  std::cout << "GPU: " << prop.name << "\n";
  std::cout << "CC: " << prop.major << "." << prop.minor << "\n";

  dim3 block(args.block);
  dim3 grid((args.N + block.x - 1) / block.x);

  double flops_model = 0.0;
  double bytes_model = 0.0;

  float time_ms_total = 0.0f;
  float time_us_per_iter = 0.0f;

  if (args.dtype == "fp32") {
    if (args.op == "vadd") {
      std::vector<float> hA(args.N), hB(args.N), hC(args.N);
      std::mt19937 rng(args.seed);
      fill_random(hA, rng);
      fill_random(hB, rng);

      float* dA = nullptr;
      float* dB = nullptr;
      float* dC = nullptr;
      CHECK_CUDA(cudaMalloc(&dA, args.N * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dB, args.N * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dC, args.N * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(dA, hA.data(), args.N * sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, hB.data(), args.N * sizeof(float), cudaMemcpyHostToDevice));

      for (int i = 0; i < warmup; ++i) {
        vadd_fp32<<<grid, block>>>(dA, dB, dC, args.N);
        CHECK_KERNEL_LAUNCH();
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int i = 0; i < args.iters; ++i) {
        vadd_fp32<<<grid, block>>>(dA, dB, dC, args.N);
        CHECK_KERNEL_LAUNCH();
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&time_ms_total, start, stop));
      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));

      time_us_per_iter = (time_ms_total * 1000.0f) / args.iters;

      CHECK_CUDA(cudaMemcpy(hC.data(), dC, args.N * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));

      if (!validate_fp32(hA, hB, hC, validate_indices)) {
        return EXIT_FAILURE;
      }

      flops_model = static_cast<double>(args.N);
      bytes_model = 3.0 * args.N * sizeof(float);
    } else if (args.op == "copy") {
      std::vector<float> hIn(args.N), hOut(args.N);
      std::mt19937 rng(args.seed);
      fill_random(hIn, rng);

      float* dIn = nullptr;
      float* dOut = nullptr;
      CHECK_CUDA(cudaMalloc(&dIn, args.N * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dOut, args.N * sizeof(float)));
      if (dIn == dOut) {
        std::cerr << "Copy requires distinct input/output buffers\n";
        return EXIT_FAILURE;
      }
      CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), args.N * sizeof(float), cudaMemcpyHostToDevice));

      for (int i = 0; i < warmup; ++i) {
        copy_fp32<<<grid, block>>>(dIn, dOut, args.N);
        CHECK_KERNEL_LAUNCH();
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int i = 0; i < args.iters; ++i) {
        copy_fp32<<<grid, block>>>(dIn, dOut, args.N);
        CHECK_KERNEL_LAUNCH();
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&time_ms_total, start, stop));
      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));

      time_us_per_iter = (time_ms_total * 1000.0f) / args.iters;

      CHECK_CUDA(cudaMemcpy(hOut.data(), dOut, args.N * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaFree(dIn));
      CHECK_CUDA(cudaFree(dOut));

      if (!validate_copy_fp32(hIn, hOut, validate_indices)) {
        return EXIT_FAILURE;
      }

      flops_model = 0.0;
      bytes_model = 2.0 * args.N * sizeof(float);
    } else {
      std::vector<float> hIn(args.N), hOut(args.N);
      std::mt19937 rng(args.seed);
      fill_random(hIn, rng);

      float* dIn = nullptr;
      float* dOut = nullptr;
      CHECK_CUDA(cudaMalloc(&dIn, args.N * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dOut, args.N * sizeof(float)));
      if (dIn == dOut) {
        std::cerr << "fma_rpt requires distinct input/output buffers\n";
        return EXIT_FAILURE;
      }
      CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), args.N * sizeof(float), cudaMemcpyHostToDevice));

      for (int i = 0; i < warmup; ++i) {
        fma_rpt_fp32<<<grid, block>>>(dIn, dOut, args.N, args.fma_k, args.fma_a, args.fma_b);
        CHECK_KERNEL_LAUNCH();
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int i = 0; i < args.iters; ++i) {
        fma_rpt_fp32<<<grid, block>>>(dIn, dOut, args.N, args.fma_k, args.fma_a, args.fma_b);
        CHECK_KERNEL_LAUNCH();
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&time_ms_total, start, stop));
      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));

      time_us_per_iter = (time_ms_total * 1000.0f) / args.iters;

      CHECK_CUDA(cudaMemcpy(hOut.data(), dOut, args.N * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaFree(dIn));
      CHECK_CUDA(cudaFree(dOut));

      if (!validate_fma_rpt(hIn, hOut, args.fma_k, args.fma_a, args.fma_b, validate_indices)) {
        return EXIT_FAILURE;
      }

      flops_model = 2.0 * static_cast<double>(args.fma_k) * static_cast<double>(args.N);
      bytes_model = 2.0 * args.N * sizeof(float);
    }
  } else {
    std::vector<__half> hA(args.N), hB(args.N), hC(args.N);
    std::mt19937 rng(args.seed);
    fill_random(hA, rng);
    fill_random(hB, rng);

    __half* dA = nullptr;
    __half* dB = nullptr;
    __half* dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, args.N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB, args.N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC, args.N * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), args.N * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), args.N * sizeof(__half), cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
      vadd_fp16<<<grid, block>>>(dA, dB, dC, args.N);
      CHECK_KERNEL_LAUNCH();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < args.iters; ++i) {
      vadd_fp16<<<grid, block>>>(dA, dB, dC, args.N);
      CHECK_KERNEL_LAUNCH();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_ms_total, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    time_us_per_iter = (time_ms_total * 1000.0f) / args.iters;

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, args.N * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    if (!validate_fp16(hA, hB, hC, validate_indices)) {
      return EXIT_FAILURE;
    }

    flops_model = static_cast<double>(args.N);
    bytes_model = 3.0 * args.N * sizeof(__half);
  }

  double time_s = static_cast<double>(time_ms_total) / 1000.0;
  double flops_total = flops_model * static_cast<double>(args.iters);
  double bytes_total = bytes_model * static_cast<double>(args.iters);
  double achieved_gflops = (time_s > 0.0) ? (flops_total / time_s / 1e9) : 0.0;
  double achieved_gbs = (time_s > 0.0) ? (bytes_total / time_s / 1e9) : 0.0;

  std::string ts = timestamp_utc_iso();

  std::ifstream infile(args.csv_path);
  bool csv_exists = infile.good();
  if (csv_exists) {
    std::cerr << "CSV already exists: " << args.csv_path << "\n";
    return EXIT_FAILURE;
  }

  std::ofstream out(args.csv_path);
  if (!out) {
    std::cerr << "Failed to open CSV: " << args.csv_path << "\n";
    return EXIT_FAILURE;
  }
  out << "timestamp,device,gpu_name,cc_major,cc_minor,total_global_mem_bytes,";
  out << "driver_version,cuda_runtime_version,op,dtype,N,block,warmup,iters,";
  out << "time_ms_total,time_us_per_iter,flops_model,bytes_model,achieved_gflops,";
  out << "achieved_gbs,notes,fma_k,fma_a,fma_b\n";

  out << ts << ","
      << "gpu" << ","
      << prop.name << ","
      << prop.major << ","
      << prop.minor << ","
      << static_cast<unsigned long long>(prop.totalGlobalMem) << ","
      << driver_version << ","
      << runtime_version << ","
      << args.op << ","
      << args.dtype << ","
      << args.N << ","
      << args.block << ","
      << warmup << ","
      << args.iters << ","
      << std::fixed << std::setprecision(6) << time_ms_total << ","
      << std::fixed << std::setprecision(3) << time_us_per_iter << ","
      << std::fixed << std::setprecision(0) << flops_model << ","
      << std::fixed << std::setprecision(0) << bytes_model << ","
      << std::fixed << std::setprecision(6) << achieved_gflops << ","
      << std::fixed << std::setprecision(6) << achieved_gbs << ","
      << notes << ",";
  if (args.op == "fma_rpt") {
    out << args.fma_k << ","
        << std::fixed << std::setprecision(6) << args.fma_a << ","
        << std::fixed << std::setprecision(6) << args.fma_b;
  } else {
    out << ",,";
  }
  out << "\n";

  out.close();

  std::cout << "op=" << args.op << " dtype=" << args.dtype
            << " N=" << args.N << " block=" << args.block
            << " warmup=" << warmup << " iters=" << args.iters << "\n";
  std::cout << "time_ms_total=" << time_ms_total
            << " time_us_per_iter=" << time_us_per_iter << "\n";
  std::cout << "achieved_gflops=" << achieved_gflops
            << " achieved_gbs=" << achieved_gbs << "\n";
  std::cout << "CSV: " << args.csv_path << "\n";

  return EXIT_SUCCESS;
}
