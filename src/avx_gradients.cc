#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "avx_allocator.h"

struct bst_gpair {
  bst_gpair() {}
  bst_gpair(float grad, float hess) : grad(grad), hess(hess) {}
  float grad;
  float hess;
};

struct Timer {
  typedef std::chrono::high_resolution_clock ClockT;
  typedef std::chrono::high_resolution_clock::time_point TimePointT;
  typedef std::chrono::high_resolution_clock::duration DurationT;
  typedef std::chrono::duration<double> SecondsT;

  TimePointT start;
  DurationT elapsed;
  Timer() { Reset(); }
  void Reset() {
    elapsed = DurationT::zero();
    Start();
  }
  void Start() { start = ClockT::now(); }
  void Stop() { elapsed += ClockT::now() - start; }
  double ElapsedSeconds() const { return SecondsT(elapsed).count(); }
  void PrintElapsed(std::string label) {
    printf("%s:\t %fs\n", label.c_str(), SecondsT(elapsed).count());
    Reset();
  }
};

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct LinearSquareLoss {
  static float FirstOrderGradient(float predt, float label) {
    return predt - label;
  }
  static float SecondOrderGradient(float predt, float label) { return 1.0f; }
};

struct LogisticRegression {
  static float PredTransform(float x) { return Sigmoid(x); }
  static float FirstOrderGradient(float predt, float label) {
    return predt - label;
  }
  static float SecondOrderGradient(float predt, float label) {
    const float eps = 1e-16f;
    return std::max(predt * (1.0f - predt), eps);
  }
};

void BinaryLogisticGradients(
    const std::vector<float, AlignedAllocator<float>> &preds,
    const std::vector<float, AlignedAllocator<float>> &labels,
    const std::vector<float, AlignedAllocator<float>> &weights,
    float scale_pos_weight,
    std::vector<bst_gpair, AlignedAllocator<bst_gpair>> *out_gpair) {
  for (size_t i = 0; i < out_gpair->size(); i++) {
    float y = labels[i];
    float p = LogisticRegression::PredTransform(preds[i]);
    float w = weights[i];
    w += y * (scale_pos_weight * w - w);
    (*out_gpair)[i] =
        bst_gpair(LogisticRegression::FirstOrderGradient(p, y) * w,
                  LogisticRegression::SecondOrderGradient(p, y) * w);
  }
}

typedef __m256 Float8;
// Store 8 gradient pairs given vectors containing gradient and Hessian
void StoreGpair(bst_gpair *dst, Float8 grad, Float8 hess) {
  float *ptr = reinterpret_cast<float *>(dst);
  Float8 gpair_low = _mm256_unpacklo_ps(grad, hess);
  Float8 gpair_high = _mm256_unpackhi_ps(grad, hess);
  _mm256_store_ps(ptr, _mm256_permute2f128_ps(gpair_low, gpair_high, 0x20));
  _mm256_store_ps(ptr + 8, _mm256_permute2f128_ps(gpair_low, gpair_high, 0x31));
}

// https://codingforspeed.com/using-faster-exponential-approximation/
inline Float8 Exp4096(Float8 x) {
  Float8 fraction = _mm256_set1_ps(1.0/4096.0); 
  Float8 ones = _mm256_set1_ps(1.0f);
  x = _mm256_fmadd_ps(x, fraction, ones);  // x = 1.0 + x / 4096

  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  x = _mm256_mul_ps(x, x);
  return x;
}

inline Float8 ApproximateSigmoid(Float8 x) {
  Float8 negative = _mm256_set1_ps(-1.0f);
  Float8 ones = _mm256_set1_ps(1.0f);
  Float8 exp = Exp4096(_mm256_mul_ps(x, negative));
  x = _mm256_add_ps(exp, ones);
  return _mm256_div_ps(ones, x);
}

void BinaryLogisticGradientsAVX(
    const std::vector<float, AlignedAllocator<float>> &preds,
    const std::vector<float, AlignedAllocator<float>> &labels,
    const std::vector<float, AlignedAllocator<float>> &weights,
    float scale_pos_weight,
    std::vector<bst_gpair, AlignedAllocator<bst_gpair>> *out_gpair) {
  const size_t n = preds.size();
  auto gpair_ptr = out_gpair->data();
  std::vector<float, AlignedAllocator<float>> scale_pos_vec(8,
                                                            scale_pos_weight);
  Float8 scale = _mm256_load_ps(&scale_pos_vec[0]);

  const size_t remainder = n % 8;
  for (size_t i = 0; i < n - remainder; i += 8) {
    Float8 y = _mm256_load_ps(&labels[i]);
    Float8 p = _mm256_load_ps(&preds[i]);
    Float8 p_transformed = ApproximateSigmoid(p);
    Float8 first = _mm256_sub_ps(p_transformed, y);
    Float8 ones = _mm256_set1_ps(1.0f);
    Float8 second = _mm256_sub_ps(ones, p_transformed);
    second = _mm256_mul_ps(second, p_transformed);
    Float8 eps = _mm256_set1_ps(1e-16f);
    second = _mm256_max_ps(second, eps);

    // Adjust weight
    Float8 w = _mm256_load_ps(&weights[i]);
    Float8 w_mul = _mm256_mul_ps(w, scale);
    Float8 w_sub = _mm256_sub_ps(w_mul, w);
    Float8 w_tmp = _mm256_mul_ps(y, w_sub);
    w = _mm256_add_ps(w, w_tmp);

    first = _mm256_mul_ps(first, w);
    second = _mm256_mul_ps(second, w);

    StoreGpair(gpair_ptr + i, first, second);
  }

  // Process remainder
  for (size_t i = 8 * (n / 8); i < n; i++) {
    float y = labels[i];
    float p = LogisticRegression::PredTransform(preds[i]);
    float w = weights[i];
    w += y * (scale_pos_weight * w - w);
    (*out_gpair)[i] =
        bst_gpair(LogisticRegression::FirstOrderGradient(p, y) * w,
                  LogisticRegression::SecondOrderGradient(p, y) * w);
  }
}

void MSEGradientsAVX(
    const std::vector<float, AlignedAllocator<float>> &preds,
    const std::vector<float, AlignedAllocator<float>> &labels,
    const std::vector<float, AlignedAllocator<float>> &weights,
    float scale_pos_weight,
    std::vector<bst_gpair, AlignedAllocator<bst_gpair>> *out_gpair) {
  size_t n = preds.size();
  auto gpair_ptr = out_gpair->data();
  for (size_t i = 0; i < n; i += 8) {
    Float8 y = _mm256_load_ps(&labels[i]);
    Float8 p = _mm256_load_ps(&preds[i]);
    Float8 w = _mm256_load_ps(&weights[i]);
    Float8 first = _mm256_sub_ps(p, y);
    first = _mm256_mul_ps(first, w);
    StoreGpair(gpair_ptr + i, first, w);
  }

  for (size_t i = 8 * (n / 8); i < n; i++) {
    float y = labels[i];
    float p = preds[i];
    float w = weights[i];
    (*out_gpair)[i] =
        bst_gpair(LinearSquareLoss::FirstOrderGradient(p, y) * w,
                  LinearSquareLoss::SecondOrderGradient(p, y) * w);
  }
}

void MSEGradients(
    const std::vector<float, AlignedAllocator<float>> &preds,
    const std::vector<float, AlignedAllocator<float>> &labels,
    const std::vector<float, AlignedAllocator<float>> &weights,
    float scale_pos_weight,
    std::vector<bst_gpair, AlignedAllocator<bst_gpair>> *out_gpair) {
  for (size_t i = 0; i < out_gpair->size(); i++) {
    float y = labels[i];
    float p = preds[i];
    float w = weights[i];
    (*out_gpair)[i] =
        bst_gpair(LinearSquareLoss::FirstOrderGradient(p, y) * w,
                  LinearSquareLoss::SecondOrderGradient(p, y) * w);
  }
}

std::vector<float, AlignedAllocator<float>> RandomVector(size_t n,
                                                         float a = 0.0f,
                                                         float b = 1.0f) {
  std::vector<float, AlignedAllocator<float>> x(n);

  std::generate(x.begin(), x.end(),
                [=]() { return ((b - a) * ((float)rand() / RAND_MAX)) + a; });

  return x;
}

bool approx_equal(const float *v1, const float *v2, size_t n,
                  float abs_tolerance) {
  for (size_t i = 0; i < n; i++) {
    if (abs(v1[i] - v2[i]) > abs_tolerance) {
      printf("i=%d, %f vs %f\n", i, v1[i], v2[i]);
      return false;
    }
  }

  return true;
}

void main() {
  size_t n = 1 << 24;
  // size_t n = 16;
  auto tolerance = 1e-4;
  auto preds = RandomVector(n, -1000, 1000);
  auto labels = RandomVector(n);
  auto weights = RandomVector(n, 0.5, 5);
  // std::vector<float, AlignedAllocator<float>> weights(n, 1.0f);
  for (int i = 0; i < 1; i++) {
    std::vector<bst_gpair, AlignedAllocator<bst_gpair>> binary_gpair(n);
    Timer t;
    t.Start();
    BinaryLogisticGradients(preds, labels, weights, 1.0, &binary_gpair);
    t.Stop();
    t.PrintElapsed("BinaryLogisticGradients");
    t.Reset();

    std::vector<bst_gpair, AlignedAllocator<bst_gpair>> binary_gpair_avx(n);
    t.Start();
    BinaryLogisticGradientsAVX(preds, labels, weights, 1.0, &binary_gpair_avx);
    t.Stop();
    t.PrintElapsed("BinaryLogisticGradientsAVX");
    t.Reset();

    if (!approx_equal(reinterpret_cast<float *>(binary_gpair.data()),
                      reinterpret_cast<float *>(binary_gpair_avx.data()), n,
                      tolerance)) {
      std::cout << "Incorrect binary gradients!\n";
      for (int j = 0; j < 5; j++) {
        printf("%f/%f %f/%f\n", binary_gpair[j].grad, binary_gpair[j].hess,
               binary_gpair_avx[j].grad, binary_gpair_avx[j].hess);
      }
    } else {
      std::cout << "Correct binary gradients!\n";
    }

    std::vector<bst_gpair, AlignedAllocator<bst_gpair>> mse_gpair(n);
    t.Start();
    MSEGradients(preds, labels, weights, 1.0, &mse_gpair);
    t.Stop();
    t.PrintElapsed("MSEGradients");
    t.Reset();

    std::vector<bst_gpair, AlignedAllocator<bst_gpair>> mse_gpair_avx(n);
    t.Start();
    MSEGradientsAVX(preds, labels, weights, 1.0, &mse_gpair_avx);
    t.Stop();
    t.PrintElapsed("MSEGradientsAVX");
    t.Reset();

    if (!approx_equal(reinterpret_cast<float *>(mse_gpair.data()),
                      reinterpret_cast<float *>(mse_gpair_avx.data()), n,
                      tolerance)) {
      std::cout << "Incorrect mse gradients!\n";
    } else {
      std::cout << "Correct mse gradients!\n";
    }
  }
}