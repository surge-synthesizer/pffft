// FFT wrapper class, to be used for C++ code.
// Contains an API that allocates its outputs, as well as an API that takes already-allocated
// arrays. For the tl;dr of how to use it, scroll down to the "FFT" class.
#pragma once

#include <complex>
#include <new>
#include <span>
#include <type_traits>

#include "pffft-detail.hpp"

static_assert(__cplusplus >= 202002L, "Surge team libraries have moved to C++ 20");

namespace pffft
{

// Class for performing a Fourier transform. This class is not thread safe; it uses a work array.
// Different threads should have their own thread-local copy of the class.
// To have a FFT size known at run time, set N to std::dynamic_extent.
template <typename T, std::size_t N> class FFT
{
    // Ensure we're either a float or complex<float>.
    static_assert(std::is_same_v<float, typename std::remove_cv<T>::type> ||
                      std::is_same_v<std::complex<float>, typename std::remove_cv<T>::type>,
                  "T parameter must be either float or std::complex<float>.");
    // Ensure that the size is a power of two, or zero for dynamic size.
    static_assert(internal::IsPowerOfTwo(N) || N == std::dynamic_extent,
                  "N parameter must be a power of two.");
    // pffft demands size of at least 32.
    static_assert(N >= 32, "N must be at least 32.");

    // Sanity check for std::complex.
    static_assert(sizeof(std::complex<float>) == 2 * sizeof(float));

  public:
    // Alignment requirement for inputs and outputs. SSE and co need 16 byte alignment so that's
    // what we set here. However, PFFFT likes to allocate at 64-byte alignment for L2 caches so if
    // you depend on this class's types instead, that's what you'll get.
    static constexpr std::size_t alignment = 16;
    template <typename U> using AlignedVector = internal::AlignedVector<U, 64>;

    // Size of the time array for the FFT.
    std::size_t size = N;

    // Size of the frequency array for the FFT. Since the spectrum is a std::complex which is twice
    // the size of a real, when T is real and not complex the spectrum type std::complex<T> needs to
    // be half the size.
    std::size_t spectrum_size =
        (N == std::dynamic_extent
             ? N
             : (std::is_same_v<float, typename std::remove_cv<T>::type> ? N / 2 : N));

    typedef float Real;
    typedef std::complex<float> Complex;

    using Time = T;
    using Frequency = Complex;

    // Helper types for specifying correctly-sized arrays. Use these if you want to create the
    // correct array sizes for the FFT.
    static constexpr std::size_t kConstSize = N;
    static constexpr std::size_t kConstSpectrumSize =
        std::is_same_v<float, typename std::remove_cv<T>::type> ? N / 2 : N;
    using TimeArray = std::conditional_t<N != std::dynamic_extent, std::array<T, N>, void>;
    using FreqArray =
        std::conditional_t<N != std::dynamic_extent, std::array<Complex, kConstSpectrumSize>, void>;

    // Helper types for specifying aligned vectors. Use these if you want to create vectors at
    // runtime for the FFT.
    using TimeVector = AlignedVector<T>;
    using FreqVector = AlignedVector<Complex>;
    using UnorderedTimeVector = AlignedVector<float>;
    using UnorderedFreqVector = AlignedVector<float>;

    // The use_stack parameter explicitly tells the class whether to allocate the work array on the
    // stack or the heap. For small transforms (N < 16384 or so), stack can be faster. However,
    // threads can have small stacks, so it doesn't hurt to use the heap instead if you're
    // concerned. No allocation is performed except during construction time and in resize(), so
    // even if it uses the heap you don't need to worry about allocations during the operation.
    explicit FFT(bool use_stack = false)
        requires(N != std::dynamic_extent);
    // Constructor when N = std::dynamic_extent.
    FFT(std::size_t size, bool use_stack = false)
    requires(N == std::dynamic_extent);
    ~FFT();

    // Change the FFT size. Only usable when N = std::dynamic_extent.
    void resize(std::size_t size)
        requires(N == std::dynamic_extent);

    // Functions to provide pre-allocated vectors in the exactly correct sizes for the FFT. This
    // will cause an allocation, so don't use it if that's bad for you.
    TimeVector createTimeVector() const;
    FreqVector createFreqVector() const;
    UnorderedTimeVector createUnorderedTimeVector() const;
    UnorderedFreqVector createUnorderedFreqVector() const;
    // As above, but arrays.
    std::unique_ptr<T[]> createTimeArray() const;
    std::unique_ptr<Complex[]> createFreqArray() const;

    // Perform a Fourier transform.
    // Output is in canonical form, AKA the familiar array of interleaved complex numbers:
    // [bin0_real, bin0_complex, bin1_real, bin1_complex, ...]
    //
    // The result is unscaled; call the scale() method if needed.
    //
    // Input and output may alias.
    //
    // The TimeVector/FreqVector API will perform allocations. If you're in a tight loop or
    // otherwise need to avoid heap allocations, use the array API instead. The array API will throw
    // if the input and output pointers are improperly aligned.
    FreqVector forward(const TimeVector &time);
    // Alternate API to use with preallocated storage, such as that created by this class or
    // existing arrays. The time span must have N elements, and the freq span must have
    // spectrum_size elements.
    void forward(std::span<const T> time, std::span<Complex> freq);
    // Raw pointer API. time must have N elements, and freq must have spectrum_size elements.
    void forward(const T *time, Complex *freq);

    // Inverse Fourier transform.
    //
    // The TimeVector/FreqVector API will perform allocations. If you're in a tight loop or
    // otherwise need to avoid heap allocations, use the array API instead. The array API will throw
    // if the input and output pointers are improperly aligned.
    TimeVector inverse(const FreqVector &freq);
    // Alternate API to use with preallocated storage, such as that created by this class or
    // existing arrays. The freq span must have spectrum_size elements, and the time span must have
    // N elements.
    void inverse(std::span<const Complex> freq, std::span<T> time);
    // Raw pointer API. freq must have spectrum_size elements, and time must have N elements.
    void inverse(const Complex *freq, T *time);

    // "Unordered" API. In these functions, the z-domain data is stored in the
    // most efficient order for transforming it back, or using it for
    // convolution. No extra work is done reordering it back to interleaved
    // complex numbers, like with the regular forward().
    //
    // In this case, freq.size() must be equal to spectrum_size * 2, since
    // we aren't storing them as std::complex<float>, halving the size. If
    // that confuses you, use createUnorderedTimeArray() and
    // createUnorderedFreqArray() and don't worry about it.
    void forward_unordered(std::span<T> time, std::span<float> freq);
    void inverse_unordered(std::span<float> freq, std::span<T> time);

    // Perform a multiplication of the frequency components of dft_a and
    // dft_b and accumulate them into dft_ab. The arrays should have
    // been obtained with forward_unordered() (otherwise just perform the
    // operation yourself as the dft coefficients come out as regular
    // std::complex<float> values).
    //
    // The operation performed is: dft_ab += (dft_a * fdt_b)*scale
    void zconvolve_accumulate(std::span<float> dft_a, std::span<float> dft_b,
                              std::span<float> dft_ab, float scale);
    // Convenience method if you're treating a FreqVector or other typed Complex
    // array source as an un-interleaved float array. Note that the actual
    // values that end up here will not be correct complex numbers.
    void zconvolve_accumulate(std::span<Complex> dft_a, std::span<Complex> dft_b,
                              std::span<Complex> dft_ab, float scale);

    // Helper methods for scaling the output of the forward transform.
    void scale(std::span<Complex> freq) const;
    void scale_unordered(std::span<float> freq) const;

  private:
    typedef internal::pffft_transform_t TransformType;

    static constexpr TransformType FftType{
        std::is_same_v<std::complex<float>, typename std::remove_cv<T>::type>
            ? internal::PFFFT_COMPLEX
            : internal::PFFFT_REAL};

    const internal::aligned_allocator<float, alignment> aligned_float_allocator_;
    bool use_stack_{false};
    float *work_{nullptr};
    internal::PFFFT_Setup *setup_{nullptr};

    // Disable assignment and copy.
    FFT(const FFT<T, N> &fft) = delete;
    FFT<T, N> operator=(const FFT<T, N> &fft) = delete;
};

template <typename T, std::size_t N>
FFT<T, N>::FFT(bool use_stack)
    requires(N != std::dynamic_extent)
{
    use_stack_ = use_stack;

    if (!use_stack)
    {
        // We use the aligned_allocator to create and destroy the work array, instead of the regular
        // aligned new[], because of a bug on MSVC (compiler error C2956). This works around it.
        work_ = aligned_float_allocator_.allocate(spectrum_size * 2);
    }
    setup_ = pffft_new_setup(N, FftType);
}

template <typename T, std::size_t N>
FFT<T, N>::FFT(std::size_t size, bool use_stack)
    requires(N == std::dynamic_extent)
{
    use_stack_ = use_stack;
    resize(size);
}

template <typename T, std::size_t N> FFT<T, N>::~FFT()
{
    if (work_)
    {
        aligned_float_allocator_.deallocate(work_, spectrum_size * 2);
    }
    pffft_destroy_setup(setup_);
}

template <typename T, std::size_t N>
void FFT<T, N>::resize(std::size_t size)
    requires(N == std::dynamic_extent)
{
    if (!internal::IsPowerOfTwo(size))
        throw std::invalid_argument("size must be a power of two");
    if (N < 32)
        throw std::invalid_argument("size must be at least 32");

    if (setup_)
        pffft_destroy_setup(setup_);

    this->size = size;
    this->spectrum_size = std::is_same_v<float, typename std::remove_cv<T>::type> ? size / 2 : size;

    if (!use_stack_)
    {
        // We use the aligned_allocator to create and destroy the work array, instead of the regular
        // aligned new[], because of a bug on MSVC (compiler error C2956). This works around it.
        work_ = aligned_float_allocator_.allocate(spectrum_size * 2);
    }
    setup_ = pffft_new_setup(size, FftType);
}

template <typename T, std::size_t N>
typename FFT<T, N>::TimeVector FFT<T, N>::createTimeVector() const
{
    if constexpr (N != std::dynamic_extent)
        return TimeVector(N);
    else
        return TimeVector(size);
}

template <typename T, std::size_t N>
typename FFT<T, N>::FreqVector FFT<T, N>::createFreqVector() const
{
    if constexpr (N != std::dynamic_extent)
        return FreqVector(kConstSpectrumSize);
    else
        return FreqVector(spectrum_size);
}

template <typename T, std::size_t N>
typename FFT<T, N>::UnorderedTimeVector FFT<T, N>::createUnorderedTimeVector() const
{
    constexpr int M = std::is_same_v<float, typename std::remove_cv<T>::type> ? 1 : 2;
    if constexpr (N != std::dynamic_extent)
        return UnorderedTimeVector(N * M);
    else
        return UnorderedTimeVector(size * M);
}

template <typename T, std::size_t N>
typename FFT<T, N>::UnorderedFreqVector FFT<T, N>::createUnorderedFreqVector() const
{
    constexpr int M = std::is_same_v<float, typename std::remove_cv<T>::type> ? 1 : 2;
    if constexpr (N != std::dynamic_extent)
        return UnorderedFreqVector(N * M);
    else
        return UnorderedFreqVector(size * M);
}

template <typename T, std::size_t N> std::unique_ptr<T[]> FFT<T, N>::createTimeArray() const
{
    return std::unique_ptr<T[]>(new (static_cast<std::align_val_t>(alignment)) T[size]);
}

template <typename T, std::size_t N>
std::unique_ptr<typename FFT<T, N>::Complex[]> FFT<T, N>::createFreqArray() const
{
    return std::unique_ptr<Complex[]>(new (static_cast<std::align_val_t>(alignment))
                                          Complex[spectrum_size]);
}

template <typename T, std::size_t N>
typename FFT<T, N>::FreqVector FFT<T, N>::forward(const TimeVector &time)
{
    FreqVector out = createFreqVector();
    forward(time.data(), out.data());
    return out;
}

template <typename T, std::size_t N>
void FFT<T, N>::forward(const std::span<const T> time, std::span<Complex> freq)
{
    if (time.size() < size) [[unlikely]]
        throw std::invalid_argument("time is not large enough");
    if (freq.size() < spectrum_size) [[unlikely]]
        throw std::invalid_argument("freq is not large enough");
    forward(time.data(), freq.data());
}

template <typename T, std::size_t N> void FFT<T, N>::forward(const T *time, Complex *freq)
{
    if (!internal::is_aligned(time, alignment)) [[unlikely]]
    {
        throw std::invalid_argument("input not aligned");
    }
    if (!internal::is_aligned(freq, alignment)) [[unlikely]]
    {
        throw std::invalid_argument("output not aligned");
    }

    internal::pffft_transform_ordered(setup_, reinterpret_cast<const float *>(time),
                                      reinterpret_cast<float *>(freq), work_,
                                      internal::PFFFT_FORWARD);
}

template <typename T, std::size_t N>
typename FFT<T, N>::TimeVector FFT<T, N>::inverse(const FreqVector &freq)
{
    TimeVector out = createTimeVector();
    inverse(freq.data(), out.data());
    return out;
}

template <typename T, std::size_t N>
void FFT<T, N>::inverse(const std::span<const Complex> freq, std::span<T> time)
{
    if (time.size() < size) [[unlikely]]
        throw std::invalid_argument("time is not large enough");
    if (freq.size() < spectrum_size) [[unlikely]]
        throw std::invalid_argument("freq is not large enough");
    inverse(freq.data(), time.data());
}

template <typename T, std::size_t N> void FFT<T, N>::inverse(const Complex *freq, T *time)
{
    if (!internal::is_aligned(time, alignment)) [[unlikely]]
    {
        throw std::invalid_argument("input not aligned");
    }
    if (!internal::is_aligned(freq, alignment)) [[unlikely]]
    {
        throw std::invalid_argument("output not aligned");
    }

    internal::pffft_transform_ordered(setup_, reinterpret_cast<const float *>(freq),
                                      reinterpret_cast<float *>(time), work_,
                                      internal::PFFFT_BACKWARD);
}

template <typename T, std::size_t N>
void FFT<T, N>::forward_unordered(const std::span<T> time, std::span<float> freq)
{
    if (time.size() < size) [[unlikely]]
        throw std::invalid_argument("time is not large enough");
    if (freq.size() < spectrum_size * 2) [[unlikely]]
        throw std::invalid_argument("freq is not large enough");
    internal::pffft_transform(setup_, reinterpret_cast<const float *>(time.data()), freq.data(),
                              work_, internal::PFFFT_FORWARD);
}

template <typename T, std::size_t N>
void FFT<T, N>::inverse_unordered(const std::span<float> freq, std::span<T> time)
{
    if (time.size() < size) [[unlikely]]
        throw std::invalid_argument("time is not large enough");
    if (freq.size() < spectrum_size * 2) [[unlikely]]
        throw std::invalid_argument("freq is not large enough");
    internal::pffft_transform(setup_, freq.data(), reinterpret_cast<float *>(time.data()), work_,
                              internal::PFFFT_BACKWARD);
}

template <typename T, std::size_t N>
void FFT<T, N>::zconvolve_accumulate(const std::span<float> dftA, const std::span<float> dftB,
                                     std::span<float> dftAB, float scale)
{
    if (dftA.size() < spectrum_size * 2) [[unlikely]]
        throw std::invalid_argument("dftA is not large enough");
    if (dftB.size() < spectrum_size * 2) [[unlikely]]
        throw std::invalid_argument("dftB is not large enough");
    if (dftAB.size() < spectrum_size * 2) [[unlikely]]
        throw std::invalid_argument("dftAB is not large enough");
    internal::pffft_zconvolve_accumulate(setup_, dftA.data(), dftB.data(), dftAB.data(), scale);
}

template <typename T, std::size_t N>
void FFT<T, N>::zconvolve_accumulate(const std::span<Complex> dftA, const std::span<Complex> dftB,
                                     std::span<Complex> dftAB, float scale)
{
    if (dftA.size() < spectrum_size) [[unlikely]]
        throw std::invalid_argument("dftA is not large enough");
    if (dftB.size() < spectrum_size) [[unlikely]]
        throw std::invalid_argument("dftB is not large enough");
    if (dftAB.size() < spectrum_size) [[unlikely]]
        throw std::invalid_argument("dftAB is not large enough");
    float *a = reinterpret_cast<float *>(dftA.data());
    float *b = reinterpret_cast<float *>(dftB.data());
    float *ab = reinterpret_cast<float *>(dftAB.data());
    internal::pffft_zconvolve_accumulate(setup_, a, b, ab, scale);
}

template <typename T, std::size_t N> void FFT<T, N>::scale(std::span<Complex> freq) const
{
    for (Complex &f : freq)
    {
        f /= static_cast<Real>(size);
    }
}

template <typename T, std::size_t N> void FFT<T, N>::scale_unordered(std::span<float> freq) const
{
    for (float &f : freq)
    {
        f /= static_cast<float>(size);
    }
}

} // namespace pffft
