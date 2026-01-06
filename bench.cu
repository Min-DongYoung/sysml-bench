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

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#endif

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

struct Args {
  std::string op = "vadd";
  std::string dtype = "fp32";
  int N = 1 << 24;
  int warmup = 10;
  int iters = 50;
  int block = 256;
  int seed = 123;
  std::string csv_path = "results.csv";
};

bool parse_args(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const std::string& k) -> bool {
      return k == "--op" || k == "--N" || k == "--warmup" || k == "--iters" ||
             k == "--csv" || k == "--dtype" || k == "--block" || k == "--seed";
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

bool validate_fp32(const std::vector<float>& a,
                   const std::vector<float>& b,
                   const std::vector<float>& c) {
  const float tol = 1e-5f;
  int n = static_cast<int>(c.size());
  int checks = std::min(n, 10);
  if (checks == 0) return true;
  for (int j = 0; j < checks; ++j) {
    int idx = (checks == 1) ? 0 : (j * (n - 1) / (checks - 1));
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
                   const std::vector<__half>& c) {
  const float tol = 1e-2f;
  int n = static_cast<int>(c.size());
  int checks = std::min(n, 10);
  if (checks == 0) return true;
  for (int j = 0; j < checks; ++j) {
    int idx = (checks == 1) ? 0 : (j * (n - 1) / (checks - 1));
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

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    return EXIT_FAILURE;
  }

  if (args.op != "vadd") {
    std::cerr << "Unsupported op: " << args.op << " (only vadd)\n";
    return EXIT_FAILURE;
  }
  if (args.dtype != "fp32" && args.dtype != "fp16") {
    std::cerr << "Unsupported dtype: " << args.dtype << " (fp32|fp16)\n";
    return EXIT_FAILURE;
  }
  if (args.N <= 0 || args.block <= 0 || args.iters <= 0) {
    std::cerr << "Invalid N/block/iters\n";
    return EXIT_FAILURE;
  }

  int warmup = args.warmup;
  std::string notes;
  if (warmup < 1) {
    warmup = 1;
    notes = "warmup_forced=1";
  }

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

  double flops_model = static_cast<double>(args.N);
  double bytes_model = 0.0;

  float time_ms_total = 0.0f;
  float time_us_per_iter = 0.0f;

  if (args.dtype == "fp32") {
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

    if (!validate_fp32(hA, hB, hC)) {
      return EXIT_FAILURE;
    }

    bytes_model = 3.0 * args.N * sizeof(float);
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

    if (!validate_fp16(hA, hB, hC)) {
      return EXIT_FAILURE;
    }

    bytes_model = 3.0 * args.N * sizeof(__half);
  }

  double time_s = static_cast<double>(time_ms_total) / 1000.0;
  double flops_total = flops_model * static_cast<double>(args.iters);
  double bytes_total = bytes_model * static_cast<double>(args.iters);
  double achieved_gflops = (time_s > 0.0) ? (flops_total / time_s / 1e9) : 0.0;
  double achieved_gbs = (time_s > 0.0) ? (bytes_total / time_s / 1e9) : 0.0;

  std::string ts = timestamp_utc_iso();

  bool write_header = true;
#if __has_include(<filesystem>)
  write_header = !fs::exists(args.csv_path);
#else
  std::ifstream infile(args.csv_path);
  write_header = !infile.good();
#endif

  std::ofstream out(args.csv_path, std::ios::app);
  if (!out) {
    std::cerr << "Failed to open CSV: " << args.csv_path << "\n";
    return EXIT_FAILURE;
  }
  if (write_header) {
    out << "timestamp,device,gpu_name,cc_major,cc_minor,total_global_mem_bytes,";
    out << "driver_version,cuda_runtime_version,op,dtype,N,block,warmup,iters,";
    out << "time_ms_total,time_us_per_iter,flops_model,bytes_model,achieved_gflops,";
    out << "achieved_gbs,notes\n";
  }

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
      << notes << "\n";

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
