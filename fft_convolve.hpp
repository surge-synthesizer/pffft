// FFT-based zero-latency convolution.
// Based on FFTConvolver, via HiFi-LoFi
//
// Copyright (c) 2017 HiFi-LoFi, 2025 Surge Synth Team
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include <span>
#include <vector>

#include "pffft.hpp"

/**
 * Implementation of a partitioned FFT convolution algorithm with uniform block size
 *
 * Some notes on how to use it:
 *
 * - After initialization with an impulse response, subsequent data portions of
 *   arbitrary length can be convolved. The convolver internally can handle
 *   this by using appropriate buffering.
 *
 * - The convolver works without "latency" (except for the required
 *   processing time, of course), i.e. the output always is the convolved
 *   input for each processing call.
 *
 * - The convolver is suitable for real-time processing which means that no
 *   "unpredictable" operations like allocations, locking, API calls, etc. are
 *   performed during processing (all necessary allocations and preparations take
 *   place during initialization).
 */
namespace pffft
{
class Convolver
{
  public:
    Convolver();

    /**
     * Initialize the convolver.
     *   block_size: Block size internally used by the convolver (partition size).
     *   ir: The impulse response. A copy will be made.
     * Returns true on successful initialization, false otherwise.
     */
    bool init(std::size_t blockSize, std::span<float> ir);

    /**
     * Convolves the the given input samples and immediately outputs the result.
     *  input: The input samples.
     *  output: The convolution result.
     */
    void process(std::span<float> input, std::span<float> output);

    // Resets the convolver and discards the set inpulse response.
    void reset();

  private:
    friend class TwoStageConvolver;
    using RTFFT = FFT<float, std::dynamic_extent>;
    std::size_t blockSize_;
    std::size_t segSize_;
    std::size_t segCount_;
    std::size_t currentSegment_;
    std::size_t position_;
    RTFFT fft_;
    RTFFT::UnorderedTimeVector fftBuffer_;
    std::vector<RTFFT::UnorderedFreqVector> segments_;
    std::vector<RTFFT::UnorderedFreqVector> irSegments_;
    RTFFT::UnorderedFreqVector preMultiplied_;
    RTFFT::UnorderedFreqVector conv_;
    RTFFT::UnorderedTimeVector inputBuffer_;
    RTFFT::UnorderedTimeVector overlap_;
    std::size_t inputBufferFill_;
};

/**
 * FFT convolver using two different block sizes
 *
 * The 2-stage convolver consists internally of two convolvers:
 *
 * - A head convolver, which processes the only the begin of the impulse response.
 * - A tail convolver, which processes the rest and major amount of the impulse response.
 *
 * Using a short block size for the head convolver and a long block size for
 * the tail convolver results in much less CPU usage, while keeping the
 * calculation time of each processing call short.
 *
 * As well as the basic FFTConvolver class, the 2-stage convolver is suitable
 * for real-time processing which means that no "unpredictable" operations like
 * allocations, locking, API calls, etc. are performed during processing (all
 * necessary allocations and preparations take place during initialization).
 */
class TwoStageConvolver
{
  public:
    TwoStageConvolver();

    bool init(std::size_t headBlockSize, std::size_t tailBlockSize, std::span<float> ir);
    void process(std::span<float> input, std::span<float> output);
    void reset();

  private:
    using FloatVec = Convolver::RTFFT::UnorderedTimeVector;
    std::size_t headBlockSize_{0};
    std::size_t tailBlockSize_{0};
    Convolver headConvolver_;
    Convolver tailConvolver0_;
    FloatVec tailOutput0_;
    FloatVec tailPrecalculated0_;
    Convolver tailConvolver_;
    FloatVec tailOutput_;
    FloatVec tailPrecalculated_;
    FloatVec tailInput_;
    std::size_t tailInputFill_{0};
    std::size_t precalculatedPos_{0};
    FloatVec backgroundProcessingInput_;

    // Prevent uncontrolled usage
    TwoStageConvolver(const TwoStageConvolver &) = delete;
    TwoStageConvolver &operator=(const TwoStageConvolver &) = delete;
};

} // namespace pffft
