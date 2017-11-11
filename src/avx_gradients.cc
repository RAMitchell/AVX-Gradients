#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "avx_helpers.h"

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
  void Reset() { elapsed = DurationT::zero(); }
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
    const std::vector<float, avx::AlignedAllocator<float>> &preds,
    const std::vector<float, avx::AlignedAllocator<float>> &labels,
    const std::vector<float, avx::AlignedAllocator<float>> &weights,
    float scale_pos_weight,
    std::vector<bst_gpair, avx::AlignedAllocator<bst_gpair>> *out_gpair) {
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

//typedef __m256 Float8;
// Store 8 gradient pairs given vectors containing gradient and Hessian
void StoreGpair(bst_gpair *dst, const avx::Float8 &grad, const avx::Float8 &hess) {
  float *ptr = reinterpret_cast<float *>(dst);
  __m256 gpair_low = _mm256_unpacklo_ps(grad.x, hess.x);
  __m256 gpair_high = _mm256_unpackhi_ps(grad.x, hess.x);
  //_mm256_storeu_ps(ptr, _mm256_permute2f128_ps(gpair_low, gpair_high, 0x20));
  //_mm256_storeu_ps(ptr + 8, _mm256_permute2f128_ps(gpair_low, gpair_high, 0x31));
  _mm256_stream_ps(ptr, _mm256_permute2f128_ps(gpair_low, gpair_high, 0x20));
  _mm256_stream_ps(ptr + 8, _mm256_permute2f128_ps(gpair_low, gpair_high, 0x31));
}

// https://codingforspeed.com/using-faster-exponential-approximation/
inline avx::Float8 Exp4096(avx::Float8 x) {
  avx::Float8 fraction(1.0 / 4096.0);
  avx::Float8 ones(1.0f);
  x.x = _mm256_fmadd_ps(x.x, fraction.x, ones.x);  // x = 1.0 + x / 4096
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  return x;
}

inline avx::Float8 ApproximateSigmoid(avx::Float8 x) {
  avx::Float8 exp = Exp4096(x * avx::Float8(-1.0f));
  x = avx::Float8(1.0f) + exp;
  return avx::Float8(_mm256_rcp_ps(x.x));
}

void BinaryLogisticGradientsAVX(
    const std::vector<float, avx::AlignedAllocator<float>> &preds,
    const std::vector<float, avx::AlignedAllocator<float>> &labels,
    const std::vector<float, avx::AlignedAllocator<float>> &weights,
    const float scale_pos_weight,
    std::vector<bst_gpair, avx::AlignedAllocator<bst_gpair>> *out_gpair) {
  const size_t n = preds.size();
  auto gpair_ptr = out_gpair->data();
  std::vector<float, avx::AlignedAllocator<float>> scale_pos_vec(
      8, scale_pos_weight);
  avx::Float8 scale(scale_pos_weight);

  const size_t remainder = n % 8;
  for (size_t i = 0; i < n - remainder; i += 8) {
    avx::Float8 y(&labels[i]);
    avx::Float8 p(&preds[i]);
    avx::Float8 w(&weights[i]);
    // Adjust weight
    w += y * (scale * w - w);

    avx::Float8 p_transformed = ApproximateSigmoid(p);
    avx::Float8 grad = p_transformed - y;
    grad *= w;
    avx::Float8 hess = (avx::Float8(1.0f) - p_transformed) * p_transformed;
    hess *= w;
    hess.x = _mm256_max_ps(hess.x, _mm256_set1_ps(1e-16f));

    StoreGpair(gpair_ptr + i, grad, hess);
  }

  // Process remainder
  for (size_t i = n-  remainder; i < n; i++) {
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
    const std::vector<float, avx::AlignedAllocator<float>> &preds,
    const std::vector<float, avx::AlignedAllocator<float>> &labels,
    const std::vector<float, avx::AlignedAllocator<float>> &weights,
    float scale_pos_weight,
    std::vector<bst_gpair, avx::AlignedAllocator<bst_gpair>> *out_gpair) {
  size_t n = preds.size();
  auto gpair_ptr = out_gpair->data();
  const size_t remainder = n % 8;
  for (size_t i = 0; i < n - remainder; i += 8) {
    avx::Float8 y(&labels[i]);
    avx::Float8 p(&preds[i]);
    avx::Float8 w(&weights[i]);
    avx::Float8 grad = p- y;
    grad *= w;
    avx::Float8 hess = w;
    StoreGpair(gpair_ptr + i, grad, hess);
  }

  for (size_t i = n-remainder; i < n; i++) {
    float y = labels[i];
    float p = preds[i];
    float w = weights[i];
    (*out_gpair)[i] =
        bst_gpair(LinearSquareLoss::FirstOrderGradient(p, y) * w,
                  LinearSquareLoss::SecondOrderGradient(p, y) * w);
  }
}

void MSEGradients(
    const std::vector<float, avx::AlignedAllocator<float>> &preds,
    const std::vector<float, avx::AlignedAllocator<float>> &labels,
    const std::vector<float, avx::AlignedAllocator<float>> &weights,
    float scale_pos_weight,
    std::vector<bst_gpair, avx::AlignedAllocator<bst_gpair>> *out_gpair) {
  size_t n = out_gpair->size();
  for (size_t i = 0; i < n; i++) {
    float y = labels[i];
    float p = preds[i];
    float w = weights[i];
    (*out_gpair)[i] =
        bst_gpair(LinearSquareLoss::FirstOrderGradient(p, y) * w,
                  LinearSquareLoss::SecondOrderGradient(p, y) * w);
  }
}

std::vector<float, avx::AlignedAllocator<float>> RandomVector(size_t n,
                                                              float a = 0.0f,
                                                              float b = 1.0f) {
  std::vector<float, avx::AlignedAllocator<float>> x(n);

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

int main() {
  size_t n = 1 << 24;
   //size_t n = 13;
  auto tolerance = 1e-3;
  auto preds = RandomVector(n, -10, 10);
  auto labels = RandomVector(n);
  auto weights = RandomVector(n, 0.5, 5);
   //auto weights = std::vector<float,avx::AlignedAllocator<float > >(n, 1.0);
  for (int i = 0; i < 2; i++) {
    std::vector<bst_gpair, avx::AlignedAllocator<bst_gpair>> binary_gpair(n);
    Timer t;
    t.Start();
    BinaryLogisticGradients(preds, labels, weights, 1.0, &binary_gpair);
    t.Stop();
    t.PrintElapsed("BinaryLogisticGradients");
    t.Reset();

    std::vector<bst_gpair, avx::AlignedAllocator<bst_gpair>> binary_gpair_avx(
        n);
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

    std::vector<bst_gpair, avx::AlignedAllocator<bst_gpair>> mse_gpair(n);
    t.Start();
    MSEGradients(preds, labels, weights, 1.0, &mse_gpair);
    t.Stop();
    t.PrintElapsed("MSEGradients");
    t.Reset();

    std::vector<bst_gpair, avx::AlignedAllocator<bst_gpair>> mse_gpair_avx(n);
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
  return 0;
}