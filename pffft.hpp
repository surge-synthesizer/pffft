// C++ wrapper for PFFFT, taken from the marton78 fork.
// Modified to check for power-of-two sizing at compile time, rather than runtime.

/* Copyright (c) 2020  Dario Mambro ( dario.mambro@gmail.com )
   Copyright (c) 2020  Hayati Ayguen ( h_ayguen@web.de )

   Redistribution and use of the Software in source and binary forms,
   with or without modification, is permitted provided that the
   following conditions are met:

   - Neither the names of PFFFT, nor the names of its
   sponsors or contributors may be used to endorse or promote products
   derived from this Software without specific prior written permission.

   - Redistributions of source code must retain the above copyright
   notices, this list of conditions, and the disclaimer below.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the disclaimer below in the
   documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
   SOFTWARE.
*/

#pragma once

#include <complex>
#include <vector>
#include <limits>
#include <cassert>
#include <type_traits>

namespace pffft
{
namespace detail
{
#include "pffft.h"

// Utility function to make sure our inputs are powers of two.
// Can't use the Juce one because we're in a split-out library.
static constexpr bool IsPowerOfTwo(size_t x) { return x && (x & (x - 1)) == 0; }
} // namespace detail
} // namespace pffft

namespace pffft
{

// enum { PFFFT_REAL, PFFFT_COMPLEX }
typedef detail::pffft_transform_t TransformType;

// define 'Scalar' and 'Complex' (in namespace pffft) with template Types<>
// and other type specific helper functions
template <typename T> struct Types
{
};
template <> struct Types<float>
{
    typedef float Scalar;
    typedef std::complex<Scalar> Complex;
    static int simd_size() { return detail::pffft_simd_size(); }
};
template <> struct Types<std::complex<float>>
{
    typedef float Scalar;
    typedef std::complex<float> Complex;
    static int simd_size() { return detail::pffft_simd_size(); }
};

// Allocator
template <typename T> class PFAlloc;

namespace detail
{
template <typename T> class Setup;
}

// define AlignedVector<T> utilizing 'using' in C++11
template <typename T> using AlignedVector = typename std::vector<T, PFAlloc<T>>;

// T can be float or std::complex<float>.
template <typename T, std::size_t N> class Fft
{
    // Ensure we're either a float or complex<float>.
    static_assert(std::is_same_v<float, typename std::remove_cv<T>::type> ||
                      std::is_same_v<std::complex<float>, typename std::remove_cv<T>::type>,
                  "T parameter must be either float or std::complex<float>.");

    // Ensure that the size is a power of two.
    static_assert(detail::IsPowerOfTwo(N), "N parameter must be a power of two.");

  public:
    // define types value_type, Scalar and Complex
    typedef T value_type;
    typedef typename Types<T>::Scalar Scalar;
    typedef typename Types<T>::Complex Complex;

    // static retrospection functions
    static bool isComplexTransform() { return sizeof(T) == sizeof(Complex); }
    static bool isFloatScalar() { return sizeof(Scalar) == sizeof(float); }
    static bool isDoubleScalar() { return sizeof(Scalar) == sizeof(double); }

    static int simd_size() { return Types<T>::simd_size(); }

    //////////////////

    /*
     * Contructor, preparing transforms.
     *
     * For N <= stackThresholdLen, the stack is used for the internal
     * work memory. for bigger N, the heap is used.
     *
     * Using the stack is probably the best strategy for small
     * FFTs, say for N <= 4096). Threads usually have a small stack, that
     * there's no sufficient amount of memory, usually leading to a crash!
     */
    Fft(int stackThresholdLen = 4096);

    /*
     * constructor produced a valid FFT instance?
     */
    bool isValid() const;

    ~Fft();

    /*
     * retrieve size of complex spectrum vector,
     * the output of forward()
     */
    int getSpectrumSize() const { return isComplexTransform() ? length : (length / 2); }

    /*
     * retrieve size of spectrum vector - in internal layout;
     * the output of forwardToInternalLayout()
     */
    int getInternalLayoutSize() const { return isComplexTransform() ? (2 * length) : length; }

    ////////////////////////////////////////////
    ////
    //// API 1, with std::vector<> based containers,
    ////   which free the allocated memory themselves (RAII).
    ////
    //// uses an Allocator for the alignment of SIMD data.
    ////
    ////////////////////////////////////////////

    // create suitably preallocated aligned vector for one FFT
    AlignedVector<T> valueVector() const;
    AlignedVector<Complex> spectrumVector() const;
    AlignedVector<Scalar> internalLayoutVector() const;

    ////////////////////////////////////////////
    // although using Vectors for output ..
    // they need to have resize() applied before!

    // core API, having the spectrum in canonical order

    /*
     * Perform the forward Fourier transform.
     *
     * Transforms are not scaled: inverse(forward(x)) = N*x.
     * Typically you will want to scale the backward transform by 1/N.
     *
     * The output 'spectrum' is canonically ordered - as expected.
     *
     * a) for complex input isComplexTransform() == true,
     *    and transformation length N  the output array is complex:
     *   index k in 0 .. N/2 -1  corresponds to frequency k * Samplerate / N
     *   index k in N/2 .. N -1  corresponds to frequency (k -N) * Samplerate / N,
     *     resulting in negative frequencies
     *
     * b) for real input isComplexTransform() == false,
     *    and transformation length N  the output array is 'mostly' complex:
     *   index k in 1 .. N/2 -1  corresponds to frequency k * Samplerate / N
     *   index k == 0 is a special case:
     *     the real() part contains the result for the DC frequency 0,
     *     the imag() part contains the result for the Nyquist frequency Samplerate/2
     *   both 0-frequency and half frequency components, which are real,
     *   are assembled in the first entry as  F(0)+i*F(N/2).
     *   with the output size N/2 complex values, it is obvious, that the
     *   result for negative frequencies are not output, cause of symmetry.
     *
     * input and output may alias - if you do nasty type conversion.
     * return is just the given output parameter 'spectrum'.
     */
    AlignedVector<Complex> &forward(const AlignedVector<T> &input,
                                    AlignedVector<Complex> &spectrum);

    /*
     * Perform the inverse Fourier transform, see forward().
     * return is just the given output parameter 'output'.
     */
    AlignedVector<T> &inverse(const AlignedVector<Complex> &spectrum, AlignedVector<T> &output);

    // provide additional functions with spectrum in some internal Layout.
    // these are faster, cause the implementation omits the reordering.
    // these are useful in special applications, like fast convolution,
    // where inverse() is following anyway ..

    /*
     * Perform the forward Fourier transform - similar to forward(), BUT:
     *
     * The z-domain data is stored in the most efficient order
     * for transforming it back, or using it for convolution.
     * If you need to have its content sorted in the "usual" canonical order,
     * either use forward(), or call reorderSpectrum() after calling
     * forwardToInternalLayout(), and before the backward fft
     *
     * return is just the given output parameter 'spectrum_internal_layout'.
     */
    AlignedVector<Scalar> &forwardToInternalLayout(const AlignedVector<T> &input,
                                                   AlignedVector<Scalar> &spectrum_internal_layout);

    /*
     * Perform the inverse Fourier transform, see forwardToInternalLayout()
     *
     * return is just the given output parameter 'output'.
     */
    AlignedVector<T> &
    inverseFromInternalLayout(const AlignedVector<Scalar> &spectrum_internal_layout,
                              AlignedVector<T> &output);

    /*
     * Reorder the spectrum from internal layout to have the
     * frequency components in the correct "canonical" order.
     * see forward() for a description of the canonical order.
     *
     * input and output should not alias.
     */
    void reorderSpectrum(const AlignedVector<Scalar> &input, AlignedVector<Complex> &output);

    /*
     * Perform a multiplication of the frequency components of
     * spectrum_internal_a and spectrum_internal_b
     * into spectrum_internal_ab.
     * The arrays should have been obtained with forwardToInternalLayout)
     * and should *not* have been reordered with reorderSpectrum().
     *
     * the operation performed is:
     *  spectrum_internal_ab = (spectrum_internal_a * spectrum_internal_b)*scaling
     *
     * The spectrum_internal_[a][b], pointers may alias.
     * return is just the given output parameter 'spectrum_internal_ab'.
     */
    AlignedVector<Scalar> &convolve(const AlignedVector<Scalar> &spectrum_internal_a,
                                    const AlignedVector<Scalar> &spectrum_internal_b,
                                    AlignedVector<Scalar> &spectrum_internal_ab,
                                    const Scalar scaling);

    /*
     * Perform a multiplication and accumulation of the frequency components
     * - similar to convolve().
     *
     * the operation performed is:
     *  spectrum_internal_ab += (spectrum_internal_a * spectrum_internal_b)*scaling
     *
     * The spectrum_internal_[a][b], pointers may alias.
     * return is just the given output parameter 'spectrum_internal_ab'.
     */
    AlignedVector<Scalar> &convolveAccumulate(const AlignedVector<Scalar> &spectrum_internal_a,
                                              const AlignedVector<Scalar> &spectrum_internal_b,
                                              AlignedVector<Scalar> &spectrum_internal_ab,
                                              const Scalar scaling);

    ////////////////////////////////////////////
    ////
    //// API 2, dealing with raw pointers,
    //// which need to be deallocated using alignedFree()
    ////
    //// the special allocation is required cause SIMD
    //// implementations require aligned memory
    ////
    //// Method descriptions are equal to the methods above,
    //// having  AlignedVector<T> parameters - instead of raw pointers.
    //// That is why following methods have no documentation.
    ////
    ////////////////////////////////////////////

    static void alignedFree(void *ptr);

    static T *alignedAllocType(int length);
    static Scalar *alignedAllocScalar(int length);
    static Complex *alignedAllocComplex(int length);

    // core API, having the spectrum in canonical order

    Complex *forward(const T *input, Complex *spectrum);

    T *inverse(const Complex *spectrum, T *output);

    // provide additional functions with spectrum in some internal Layout.
    // these are faster, cause the implementation omits the reordering.
    // these are useful in special applications, like fast convolution,
    // where inverse() is following anyway ..

    Scalar *forwardToInternalLayout(const T *input, Scalar *spectrum_internal_layout);

    T *inverseFromInternalLayout(const Scalar *spectrum_internal_layout, T *output);

    void reorderSpectrum(const Scalar *input, Complex *output);

    Scalar *convolve(const Scalar *spectrum_internal_a, const Scalar *spectrum_internal_b,
                     Scalar *spectrum_internal_ab, const Scalar scaling);

    Scalar *convolveAccumulate(const Scalar *spectrum_internal_a, const Scalar *spectrum_internal_b,
                               Scalar *spectrum_internal_ab, const Scalar scaling);

  private:
    detail::Setup<T> setup;
    Scalar *work;
    int length;
    int stackThresholdLen;
};

template <typename T> inline T *alignedAlloc(int length)
{
    return (T *)detail::pffft_aligned_malloc(length * sizeof(T));
}

inline void alignedFree(void *ptr) { detail::pffft_aligned_free(ptr); }

////////////////////////////////////////////////////////////////////

// implementation

namespace detail
{

template <typename T> class Setup
{
};

template <> class Setup<float>
{
    PFFFT_Setup *self;

  public:
    typedef float value_type;
    typedef Types<value_type>::Scalar Scalar;

    Setup() : self(NULL) {}

    ~Setup() { pffft_destroy_setup(self); }

    void prepareLength(int length)
    {
        if (self)
        {
            pffft_destroy_setup(self);
        }
        self = pffft_new_setup(length, PFFFT_REAL);
    }

    bool isValid() const { return (self); }

    void transform_ordered(const Scalar *input, Scalar *output, Scalar *work,
                           pffft_direction_t direction)
    {
        pffft_transform_ordered(self, input, output, work, direction);
    }

    void transform(const Scalar *input, Scalar *output, Scalar *work, pffft_direction_t direction)
    {
        pffft_transform(self, input, output, work, direction);
    }

    void reorder(const Scalar *input, Scalar *output, pffft_direction_t direction)
    {
        pffft_zreorder(self, input, output, direction);
    }

    void convolveAccumulate(const Scalar *dft_a, const Scalar *dft_b, Scalar *dft_ab,
                            const Scalar scaling)
    {
        pffft_zconvolve_accumulate(self, dft_a, dft_b, dft_ab, scaling);
    }
};

template <> class Setup<std::complex<float>>
{
    PFFFT_Setup *self;

  public:
    typedef std::complex<float> value_type;
    typedef Types<value_type>::Scalar Scalar;

    Setup() : self(NULL) {}

    ~Setup() { pffft_destroy_setup(self); }

    void prepareLength(int length)
    {
        if (self)
        {
            pffft_destroy_setup(self);
        }
        self = pffft_new_setup(length, PFFFT_COMPLEX);
    }

    bool isValid() const { return (self); }

    void transform_ordered(const Scalar *input, Scalar *output, Scalar *work,
                           pffft_direction_t direction)
    {
        pffft_transform_ordered(self, input, output, work, direction);
    }

    void transform(const Scalar *input, Scalar *output, Scalar *work, pffft_direction_t direction)
    {
        pffft_transform(self, input, output, work, direction);
    }

    void reorder(const Scalar *input, Scalar *output, pffft_direction_t direction)
    {
        pffft_zreorder(self, input, output, direction);
    }
};

} // namespace detail

template <typename T, std::size_t N>
inline Fft<T, N>::Fft(int stackThresholdLen)
    : work(NULL), length(N), stackThresholdLen(stackThresholdLen)
{
    static_assert(sizeof(Complex) == 2 * sizeof(Scalar),
                  "pffft requires sizeof(std::complex<>) == 2 * sizeof(Scalar)");

    const bool useHeap = N > stackThresholdLen;
    setup.prepareLength(N);
    if (useHeap)
    {
        work = reinterpret_cast<Scalar *>(alignedAllocType(length));
    }
}

template <typename T, std::size_t N> inline Fft<T, N>::~Fft()
{
    if (work)
    {
        alignedFree(work);
    }
}

template <typename T, std::size_t N> inline bool Fft<T, N>::isValid() const
{
    return setup.isValid();
}

template <typename T, std::size_t N> inline AlignedVector<T> Fft<T, N>::valueVector() const
{
    return AlignedVector<T>(length);
}

template <typename T, std::size_t N>
inline AlignedVector<typename Fft<T, N>::Complex> Fft<T, N>::spectrumVector() const
{
    return AlignedVector<Complex>(getSpectrumSize());
}

template <typename T, std::size_t N>
inline AlignedVector<typename Fft<T, N>::Scalar> Fft<T, N>::internalLayoutVector() const
{
    return AlignedVector<Scalar>(getInternalLayoutSize());
}

template <typename T, std::size_t N>
inline AlignedVector<typename Fft<T, N>::Complex> &
Fft<T, N>::forward(const AlignedVector<T> &input, AlignedVector<Complex> &spectrum)
{
    forward(input.data(), spectrum.data());
    return spectrum;
}

template <typename T, std::size_t N>
inline AlignedVector<T> &Fft<T, N>::inverse(const AlignedVector<Complex> &spectrum,
                                            AlignedVector<T> &output)
{
    inverse(spectrum.data(), output.data());
    return output;
}

template <typename T, std::size_t N>
inline AlignedVector<typename Fft<T, N>::Scalar> &
Fft<T, N>::forwardToInternalLayout(const AlignedVector<T> &input,
                                   AlignedVector<Scalar> &spectrum_internal_layout)
{
    forwardToInternalLayout(input.data(), spectrum_internal_layout.data());
    return spectrum_internal_layout;
}

template <typename T, std::size_t N>
inline AlignedVector<T> &
Fft<T, N>::inverseFromInternalLayout(const AlignedVector<Scalar> &spectrum_internal_layout,
                                     AlignedVector<T> &output)
{
    inverseFromInternalLayout(spectrum_internal_layout.data(), output.data());
    return output;
}

template <typename T, std::size_t N>
inline void Fft<T, N>::reorderSpectrum(const AlignedVector<Scalar> &input,
                                       AlignedVector<Complex> &output)
{
    reorderSpectrum(input.data(), output.data());
}

template <typename T, std::size_t N>
inline AlignedVector<typename Fft<T, N>::Scalar> &
Fft<T, N>::convolveAccumulate(const AlignedVector<Scalar> &spectrum_internal_a,
                              const AlignedVector<Scalar> &spectrum_internal_b,
                              AlignedVector<Scalar> &spectrum_internal_ab, const Scalar scaling)
{
    convolveAccumulate(spectrum_internal_a.data(), spectrum_internal_b.data(),
                       spectrum_internal_ab.data(), scaling);
    return spectrum_internal_ab;
}

template <typename T, std::size_t N>
inline AlignedVector<typename Fft<T, N>::Scalar> &
Fft<T, N>::convolve(const AlignedVector<Scalar> &spectrum_internal_a,
                    const AlignedVector<Scalar> &spectrum_internal_b,
                    AlignedVector<Scalar> &spectrum_internal_ab, const Scalar scaling)
{
    convolve(spectrum_internal_a.data(), spectrum_internal_b.data(), spectrum_internal_ab.data(),
             scaling);
    return spectrum_internal_ab;
}

template <typename T, std::size_t N>
inline typename Fft<T, N>::Complex *Fft<T, N>::forward(const T *input, Complex *spectrum)
{
    assert(isValid());
    setup.transform_ordered(reinterpret_cast<const Scalar *>(input),
                            reinterpret_cast<Scalar *>(spectrum), work, detail::PFFFT_FORWARD);
    return spectrum;
}

template <typename T, std::size_t N>
inline T *Fft<T, N>::inverse(Complex const *spectrum, T *output)
{
    assert(isValid());
    setup.transform_ordered(reinterpret_cast<const Scalar *>(spectrum),
                            reinterpret_cast<Scalar *>(output), work, detail::PFFFT_BACKWARD);
    return output;
}

template <typename T, std::size_t N>
inline typename pffft::Fft<T, N>::Scalar *
Fft<T, N>::forwardToInternalLayout(const T *input, Scalar *spectrum_internal_layout)
{
    assert(isValid());
    setup.transform(reinterpret_cast<const Scalar *>(input), spectrum_internal_layout, work,
                    detail::PFFFT_FORWARD);
    return spectrum_internal_layout;
}

template <typename T, std::size_t N>
inline T *Fft<T, N>::inverseFromInternalLayout(const Scalar *spectrum_internal_layout, T *output)
{
    assert(isValid());
    setup.transform(spectrum_internal_layout, reinterpret_cast<Scalar *>(output), work,
                    detail::PFFFT_BACKWARD);
    return output;
}

template <typename T, std::size_t N>
inline void Fft<T, N>::reorderSpectrum(const Scalar *input, Complex *output)
{
    assert(isValid());
    setup.reorder(input, reinterpret_cast<Scalar *>(output), detail::PFFFT_FORWARD);
}

template <typename T, std::size_t N>
inline typename pffft::Fft<T, N>::Scalar *
Fft<T, N>::convolveAccumulate(const Scalar *dft_a, const Scalar *dft_b, Scalar *dft_ab,
                              const Scalar scaling)
{
    assert(isValid());
    setup.convolveAccumulate(dft_a, dft_b, dft_ab, scaling);
    return dft_ab;
}

template <typename T, std::size_t N>
inline typename pffft::Fft<T, N>::Scalar *
Fft<T, N>::convolve(const Scalar *dft_a, const Scalar *dft_b, Scalar *dft_ab, const Scalar scaling)
{
    assert(isValid());
    setup.convolve(dft_a, dft_b, dft_ab, scaling);
    return dft_ab;
}

template <typename T, std::size_t N> inline void Fft<T, N>::alignedFree(void *ptr)
{
    pffft::alignedFree(ptr);
}

template <typename T, std::size_t N> inline T *pffft::Fft<T, N>::alignedAllocType(int length)
{
    return alignedAlloc<T>(length);
}

template <typename T, std::size_t N>
inline typename pffft::Fft<T, N>::Scalar *pffft::Fft<T, N>::alignedAllocScalar(int length)
{
    return alignedAlloc<Scalar>(length);
}

template <typename T, std::size_t N>
inline typename Fft<T, N>::Complex *Fft<T, N>::alignedAllocComplex(int length)
{
    return alignedAlloc<Complex>(length);
}

////////////////////////////////////////////////////////////////////

// Allocator - for std::vector<>:
// origin: http://www.josuttis.com/cppcode/allocator.html
// http://www.josuttis.com/cppcode/myalloc.hpp
//
// minor renaming and utilizing of pffft (de)allocation functions
// are applied to Jossutis' allocator

/* The following code example is taken from the book
 * "The C++ Standard Library - A Tutorial and Reference"
 * by Nicolai M. Josuttis, Addison-Wesley, 1999
 *
 * (C) Copyright Nicolai M. Josuttis 1999.
 * Permission to copy, use, modify, sell and distribute this software
 * is granted provided this copyright notice appears in all copies.
 * This software is provided "as is" without express or implied
 * warranty, and with no claim as to its suitability for any purpose.
 */

template <class T> class PFAlloc
{
  public:
    // type definitions
    typedef T value_type;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T &reference;
    typedef const T &const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    // rebind allocator to type U
    template <class U> struct rebind
    {
        typedef PFAlloc<U> other;
    };

    // return address of values
    pointer address(reference value) const { return &value; }
    const_pointer address(const_reference value) const { return &value; }

    /* constructors and destructor
     * - nothing to do because the allocator has no state
     */
    PFAlloc() throw() {}
    PFAlloc(const PFAlloc &) throw() {}
    template <class U> PFAlloc(const PFAlloc<U> &) throw() {}
    ~PFAlloc() throw() {}

    // return maximum number of elements that can be allocated
    size_type max_size() const throw()
    {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }

    // allocate but don't initialize num elements of type T
    pointer allocate(size_type num, const void * = 0)
    {
        pointer ret = (pointer)(alignedAlloc<T>(int(num)));
        return ret;
    }

    // initialize elements of allocated storage p with value value
    void construct(pointer p, const T &value)
    {
        // initialize memory with placement new
        new ((void *)p) T(value);
    }

    // destroy elements of initialized storage p
    void destroy(pointer p)
    {
        // destroy objects by calling their destructor
        p->~T();
    }

    // deallocate storage p of deleted elements
    void deallocate(pointer p, size_type num)
    {
        // deallocate memory with pffft
        alignedFree((void *)p);
    }
};

// return that all specializations of this allocator are interchangeable
template <class T1, class T2> bool operator==(const PFAlloc<T1> &, const PFAlloc<T2> &) throw()
{
    return true;
}
template <class T1, class T2> bool operator!=(const PFAlloc<T1> &, const PFAlloc<T2> &) throw()
{
    return false;
}

} // namespace pffft
