#pragma once
#include <cstddef>

namespace avx {

enum class Alignment : size_t {
  Normal = sizeof(void*),
  SSE = 16,
  AVX = 32,
};

namespace detail {
inline void* allocate_aligned_memory(size_t align, size_t size) {
  if (size == 0) {
    return nullptr;
  }

  void* ptr = nullptr;
#ifdef _MSC_VER
  ptr = _aligned_malloc(size, align);
#else
  int rc = posix_memalign(&ptr, align, size);
  if (rc != 0) {
    return nullptr;
  }
#endif

  return ptr;
}
inline void deallocate_aligned_memory(void* ptr) noexcept {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
};
}  // namespace detail

template <typename T, Alignment Align = Alignment::AVX>
class AlignedAllocator;

template <Alignment Align>
class AlignedAllocator<void, Align> {
 public:
  typedef void* pointer;
  typedef const void* const_pointer;
  typedef void value_type;

  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, Align> other;
  };
};

template <typename T, Alignment Align>
class AlignedAllocator {
 public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  typedef std::true_type propagate_on_container_move_assignment;

  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, Align> other;
  };

 public:
  AlignedAllocator() noexcept {}

  template <class U>
  AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

  size_type max_size() const noexcept {
    return (size_type(~0) - size_type(Align)) / sizeof(T);
  }

  pointer address(reference x) const noexcept { return std::addressof(x); }

  const_pointer address(const_reference x) const noexcept {
    return std::addressof(x);
  }

  pointer allocate(size_type n,
                   typename AlignedAllocator<void, Align>::const_pointer = 0) {
    const size_type alignment = static_cast<size_type>(Align);
    void* ptr = detail::allocate_aligned_memory(alignment, n * sizeof(T));
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }

    return reinterpret_cast<pointer>(ptr);
  }

  void deallocate(pointer p, size_type) noexcept {
    return detail::deallocate_aligned_memory(p);
  }

  template <class U, class... Args>
  void construct(U* p, Args&&... args) {
    ::new (reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
  }

  void destroy(pointer p) { p->~T(); }
};

template <typename T, Alignment Align>
class AlignedAllocator<const T, Align> {
 public:
  typedef T value_type;
  typedef const T* pointer;
  typedef const T* const_pointer;
  typedef const T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  typedef std::true_type propagate_on_container_move_assignment;

  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, Align> other;
  };

 public:
  AlignedAllocator() noexcept {}

  template <class U>
  AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

  size_type max_size() const noexcept {
    return (size_type(~0) - size_type(Align)) / sizeof(T);
  }

  const_pointer address(const_reference x) const noexcept {
    return std::addressof(x);
  }

  pointer allocate(size_type n,
                   typename AlignedAllocator<void, Align>::const_pointer = 0) {
    const size_type alignment = static_cast<size_type>(Align);
    void* ptr = detail::allocate_aligned_memory(alignment, n * sizeof(T));
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }

    return reinterpret_cast<pointer>(ptr);
  }

  void deallocate(pointer p, size_type) noexcept {
    return detail::deallocate_aligned_memory(p);
  }

  template <class U, class... Args>
  void construct(U* p, Args&&... args) {
    ::new (reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
  }

  void destroy(pointer p) { p->~T(); }
};

template <typename T, Alignment TAlign, typename U, Alignment UAlign>
inline bool operator==(const AlignedAllocator<T, TAlign>&,
                       const AlignedAllocator<U, UAlign>&) noexcept {
  return TAlign == UAlign;
}

template <typename T, Alignment TAlign, typename U, Alignment UAlign>
inline bool operator!=(const AlignedAllocator<T, TAlign>&,
                       const AlignedAllocator<U, UAlign>&) noexcept {
  return TAlign != UAlign;
}

#ifdef __GNUC__
#define ALIGN(x) x __attribute__((aligned(32)))
#elif defined(_MSC_VER)
#define ALIGN(x) __declspec(align(32))
#endif

struct Float8 {
  __m256 x;
  explicit Float8(const __m256& x) : x(x) {}
  explicit Float8(const float& val)
    :x(_mm256_broadcast_ss(&val)){}
  explicit Float8(const float* vec) : x(_mm256_load_ps(vec)) {}
  Float8() : x() {}
  Float8& operator+=(const Float8& rhs) {
    x = _mm256_add_ps(x, rhs.x);
    return *this;
  }
  Float8& operator-=(const Float8& rhs) {
    x = _mm256_sub_ps(x, rhs.x);
    return *this;
  }
  Float8& operator*=(const Float8& rhs) {
    x = _mm256_mul_ps(x, rhs.x);
    return *this;
  }
  Float8& operator/=(const Float8& rhs) {
    x = _mm256_div_ps(x, rhs.x);
    return *this;
  }
  void Print() { 
    float* f = (float*)&x;
    printf("%f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5],
           f[6], f[7]);
  }
};

inline Float8 operator+(Float8 lhs, const Float8& rhs) {
  lhs += rhs;
  return lhs;
}
inline Float8 operator-(Float8 lhs, const Float8& rhs) {
  lhs -= rhs;
  return lhs;
}
inline Float8 operator*(Float8 lhs, const Float8& rhs) {
  lhs *= rhs;
  return lhs;
}
inline Float8 operator/(Float8 lhs, const Float8& rhs) {
  lhs /= rhs;
  return lhs;
}
}  // namespace avx