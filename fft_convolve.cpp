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

#include "fft_convolve.hpp"

#include <algorithm>
#include <functional>

namespace pffft
{

Convolver::Convolver() : fft_(32, false) { reset(); }

bool Convolver::init(std::size_t blockSize, std::span<float> ir)
{
    reset();

    if (blockSize == 0)
        return false;

    if (ir.size() == 0) // process() will do nothing.
        return true;

    if (!pffft::internal::IsPowerOfTwo(blockSize))
        return false;

    // FIXME TEMPORARY Ignore zeros at the end of the impulse response because they only waste
    // computation time
    while (ir.size() > 0 && ::fabs(ir[ir.size() - 1]) < 0.000001f)
    {
        ir = ir.first(ir.size() - 1);
    }

    blockSize_ = blockSize;
    segSize_ = 2 * blockSize;
    segCount_ = static_cast<std::size_t>(
        std::ceil(static_cast<float>(ir.size()) / static_cast<float>(blockSize)));
    fft_.resize(segSize_);
    fftBuffer_ = fft_.createUnorderedTimeVector();

    // Prepare segments.
    for (auto i = 0; i < segCount_; i++)
    {
        segments_.push_back(fft_.createUnorderedFreqVector());
    }

    // Prepare IR
    for (auto i = 0; i < segCount_; i++)
    {
        auto block = fft_.createUnorderedTimeVector();
        if (ir.size() > blockSize)
        {
            std::copy(ir.begin(), ir.begin() + blockSize, block.begin());
            ir = ir.subspan(blockSize);
        }
        else
        {
            std::copy(ir.begin(), ir.end(), block.begin());
        }
        irSegments_.push_back(fft_.createUnorderedFreqVector());
        fft_.forward_unordered(block, irSegments_.back());
    }

    preMultiplied_ = fft_.createUnorderedFreqVector();
    conv_ = fft_.createUnorderedFreqVector();
    overlap_.resize(blockSize_);
    // Input buffer is twice as long as the block size; this is for the
    // necessary zero-padding.
    inputBuffer_ = fft_.createUnorderedTimeVector();
    inputBufferFill_ = 0;
    return true;
}

void Convolver::process(std::span<float> input, std::span<float> output)
{
    if (segCount_ == 0)
    {
        std::fill(output.begin(), output.end(), 0);
        return;
    }

    std::size_t processed = 0;
    while (processed < input.size())
    {
        const bool inputBufferWasEmpty = (inputBufferFill_ == 0);
        // How much of a bite to take off the input. We can only take up to
        // (1) the currently available input (if it's less than a full block),
        // or (2) enough to (possibly partially) fill up a partial block we've
        // been left with on a prior call to process().
        const std::size_t processing =
            std::min(input.size() - processed, blockSize_ - inputBufferFill_);
        const std::size_t inputBufferPos = inputBufferFill_;
        std::copy_n(input.begin() + processed, processing, inputBuffer_.begin() + inputBufferPos);

        fft_.forward_unordered(inputBuffer_, segments_[currentSegment_]);
        fft_.scale_unordered(segments_[currentSegment_]);

        if (inputBufferWasEmpty)
        {
            // This means we previously filled up a block (or haven't started
            // yet). Recalculate the convolution stack.
            std::ranges::fill(preMultiplied_, 0);
            for (std::size_t i = 1; i < segCount_; i++)
            {
                // The IR segment is always in the same order, but the input
                // segments are wrapped around as a ring buffer.
                const std::size_t iA = (currentSegment_ + i) % segCount_;
                fft_.zconvolve_accumulate(irSegments_[i], segments_[iA], preMultiplied_, 1.f);
            }
        }
        // Generate result for the current block (or portion thereof).
        std::copy_n(preMultiplied_.begin(), preMultiplied_.size(), conv_.begin());
        fft_.zconvolve_accumulate(segments_[currentSegment_], irSegments_[0], conv_, 1.f);
        fft_.inverse_unordered(conv_, fftBuffer_);

        // Add overlap
        std::transform(fftBuffer_.begin() + inputBufferPos,
                       fftBuffer_.begin() + inputBufferPos + processing,
                       overlap_.begin() + inputBufferPos, output.begin() + processed, std::plus());

        inputBufferFill_ += processing;
        // Input buffer (current block) filled up? => next block. The
        // accumulation will be done in the if (inputBufferWasEmpty) branch on
        // the next call to process().
        if (inputBufferFill_ == blockSize_)
        {
            std::ranges::fill(inputBuffer_, 0);
            inputBufferFill_ = 0;

            // Save the overlap.
            std::copy_n(fftBuffer_.begin() + blockSize_, blockSize_, overlap_.begin());

            // Update current segment (remember that ring buffer comment earlier?)
            currentSegment_ = (currentSegment_ > 0) ? (currentSegment_ - 1) : (segCount_ - 1);
        }

        processed += processing;
    }
}

void Convolver::reset()
{
    blockSize_ = segSize_ = segCount_ = currentSegment_ = position_ = inputBufferFill_ = 0;
    fftBuffer_.clear();
    segments_.clear();
    irSegments_.clear();
    inputBuffer_.clear();
    preMultiplied_.clear();
    conv_.clear();
    overlap_.clear();
}

} // namespace pffft
