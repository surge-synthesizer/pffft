// FFT wrapper class, to be used for C++ code.
// Contains an API that allocates its outputs, as well as an API that takes already-allocated
// arrays. For the tl;dr of how to use it, scroll down to the "FFT" class.
#pragma once

#include <array>
#include <complex>
#include <span>
#include <type_traits>

#include "pffft-detail.hpp"

static_assert(__cplusplus >= 202002L, "Surge team libraries have moved to C++ 20");

namespace pffft {

// Class for performing a Fourier transform. This class is not thread safe; it uses a work array.
// Different threads should have their own thread-local copy of the class.
template <typename T, std::size_t N> class FFT
{
    // Ensure we're either a float or complex<float>.
    static_assert(std::is_same_v<float, typename std::remove_cv<T>::type> ||
                      std::is_same_v<std::complex<float>, typename std::remove_cv<T>::type>,
                  "T parameter must be either float or std::complex<float>.");
    // Ensure that the size is a power of two.
    static_assert(internal::IsPowerOfTwo(N), "N parameter must be a power of two.");

    // Sanity check for std::complex.
    static_assert(sizeof(std::complex<float>) == 2 * sizeof(float));

  public:
    // Alignment requirement for inputs and outputs. SSE and co need 16 byte alignment so that's
    // what we set here. However, PFFFT likes to allocate at 64-byte alignment for L2 caches so if
    // you depend on this class's types instead, that's what you'll get.
    static constexpr std::size_t alignment = 16;
    template <typename U> using AlignedVector = internal::AlignedVector<U, 64>;

    // Helper for enumerating the spectrum size. Since the spectrum is a std::complex which
    // is twice the size of a real, when T is real and not complex the spectrum type std::complex<T>
    // needs to be half the size.
    static constexpr std::size_t spectrum_size =
        std::is_same_v<float, typename std::remove_cv<T>::type> ? N / 2 : N;

    typedef internal::pffft_transform_t TransformType;
    typedef float Real;
    typedef std::complex<float> Complex;

    using TimeArray = std::array<T, N>;
    using FreqArray = std::array<Complex, spectrum_size>;
    using TimeVector = AlignedVector<T>;
    using FreqVector = AlignedVector<Complex>;

    static constexpr TransformType FftType{
        std::is_same_v<std::complex<float>, typename std::remove_cv<T>::type>
            ? internal::PFFFT_COMPLEX
            : internal::PFFFT_REAL};

    // The use_stack parameter explicitly tells the class whether to allocate the work array on the
    // stack or the heap. For small transforms (N < 16384 or so), stack can be faster. However,
    // threads can have small stacks, so it doesn't hurt to use the heap instead if you're
    // concerned. No allocation is performed except during construction time, so even if it uses the
    // heap you don't need to worry about allocations during the operation.
    explicit FFT(bool use_stack = false);
    ~FFT();

    // Functions to provide pre-allocated vectors in the exactly correct sizes for the FFT.
    TimeVector createTimeVector() const;
    FreqVector createFreqVector() const;

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

    // Helper methods for scaling the output of the forward transform.
    void scale(FreqVector &freq) const;
    void scale(FreqArray &freq) const;
    // freq must have spectrum_size elements.
    void scale(Complex *freq) const;

  private:
    const internal::aligned_allocator<float, alignment> aligned_float_allocator_;
    float *work_{nullptr};
    internal::PFFFT_Setup *setup_{nullptr};
};

template <typename T, std::size_t N> FFT<T, N>::FFT(bool use_stack)
{
    if (!use_stack)
    {
        // We use the aligned_allocator to create and destroy the work array, instead of the regular
        // aligned new[], because of a bug on MSVC (compiler error C2956). This works around it.
        work_ = aligned_float_allocator_.allocate(spectrum_size * 2);
    }
    setup_ = pffft_new_setup(N, FftType);
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
typename FFT<T, N>::TimeVector FFT<T, N>::createTimeVector() const
{
    return TimeVector(N);
}

template <typename T, std::size_t N>
typename FFT<T, N>::FreqVector FFT<T, N>::createFreqVector() const
{
    return FreqVector(spectrum_size);
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
    if (time.size() < N)
        throw std::invalid_argument("time is not large enough");
    if (freq.size() < spectrum_size)
        throw std::invalid_argument("freq is not large enough");
    forward(time.data(), freq.data());
}

template <typename T, std::size_t N> void FFT<T, N>::forward(const T *time, Complex *freq)
{
    if (!internal::is_aligned(time, alignment))
    {
        throw std::invalid_argument("input not aligned");
    }
    if (!internal::is_aligned(freq, alignment))
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
    if (time.size() < N)
        throw std::invalid_argument("time is not large enough");
    if (freq.size() < spectrum_size)
        throw std::invalid_argument("freq is not large enough");
    inverse(freq.data(), time.data());
}

template <typename T, std::size_t N> void FFT<T, N>::inverse(const Complex *freq, T *time)
{
    if (!internal::is_aligned(time, alignment))
    {
        throw std::invalid_argument("input not aligned");
    }
    if (!internal::is_aligned(freq, alignment))
    {
        throw std::invalid_argument("output not aligned");
    }

    internal::pffft_transform_ordered(setup_, reinterpret_cast<const float *>(freq),
                                      reinterpret_cast<float *>(time), work_,
                                      internal::PFFFT_BACKWARD);
}

template <typename T, std::size_t N> void FFT<T, N>::scale(FreqVector &freq) const
{
    scale(freq.data());
}

template <typename T, std::size_t N> void FFT<T, N>::scale(FreqArray &freq) const
{
    scale(freq.data());
}

template <typename T, std::size_t N> void FFT<T, N>::scale(Complex *freq) const
{
    for (std::size_t i = 0; i < spectrum_size; i++)
    {
        freq[i] /= N;
    }
}

} // namespace pffft
