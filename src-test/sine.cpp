#include <cmath>
#include <iostream>
#include <catch2.hpp>
#include "pffft.hpp"

namespace internal = pffft::internal;
using FFT = pffft::FFT<float, 256>;

namespace
{

float hanning(int i, int n)
{
    if (i >= n)
        return 0.f;
    return 0.5f *
           (1.f - std::cos(2.f * 3.14159 * static_cast<float>(i) / (static_cast<float>(n) - 1.f)));
}

} // namespace

TEST_CASE("Sine", "Sine Wave Wrapper Equivalence")
{
    internal::PFFFT_Setup *setup = internal::pffft_new_setup(256, internal::PFFFT_REAL);
    FFT fft;
    FFT::TimeVector time = fft.createTimeVector();
    FFT::FreqVector freq = fft.createFreqVector();

    // Create sampled sine wave at 16 Hz with sample rate = 1024 samples/sec.
    for (std::size_t i = 0; i < 256; i++)
    {
        time[i] = std::sin(2.f * 3.14159f * (16.f / 1024.f) * static_cast<float>(i));
    }

    FFT::TimeVector timeC = time;
    FFT::FreqVector freqC = freq;

    fft.forward(time, freq);
    internal::pffft_transform_ordered(setup, timeC.data(), reinterpret_cast<float *>(freqC.data()),
                                      nullptr, internal::PFFFT_FORWARD);
    fft.scale(freq);
    fft.scale(freqC);

    // Check that we see a peak at 16 Hz and nowhere else. Should be around 0.5.
    for (std::size_t i = 0; i < fft.spectrum_size; i++)
    {
        if (i == 4)
        {
            REQUIRE_THAT(std::abs(freq[i]), Catch::Matchers::WithinRel(0.5f));
        }
        else if (i == 3 || i == 5)
        {
            REQUIRE_THAT(std::abs(freq[i]), Catch::Matchers::WithinAbs(0.f, 2e-6f));
        }
        else
        {
            REQUIRE_THAT(std::abs(freq[i]), Catch::Matchers::WithinAbs(0.f, 1e-6f));
        }
    }

    // Ensure that what we got from the wrapper is the same that we got from the underlying.
    for (std::size_t i = 0; i < fft.spectrum_size; i++)
    {
        REQUIRE_THAT(std::abs(freq[i]), Catch::Matchers::WithinRel(std::abs(freqC[i])));
    }

    // Exercise the array and ensure they all come to the same value.
    alignas(16) FFT::TimeArray timeA;
    alignas(16) FFT::FreqArray freqA;
    std::copy(time.begin(), time.end(), timeA.begin());
    fft.forward(timeA, freqA);
    fft.scale(freqA);
    for (std::size_t i = 0; i < fft.spectrum_size; i++)
    {
        REQUIRE_THAT(std::abs(freqA[i]), Catch::Matchers::WithinRel(std::abs(freqC[i])));
    }

    // Inverse transforms on all the data. Make sure it's equivalent.
    FFT::TimeVector orig = time;
    std::fill(time.begin(), time.end(), 0.f);
    std::fill(timeC.begin(), timeC.end(), 0.f);
    std::fill(timeA.begin(), timeA.end(), 0.f);
    fft.inverse(freq, time);
    fft.inverse(freqA, timeA);
    internal::pffft_transform_ordered(setup, reinterpret_cast<float *>(freqC.data()), timeC.data(),
                                      nullptr, internal::PFFFT_BACKWARD);
    for (std::size_t i = 0; i < 256; i++)
    {
        REQUIRE_THAT(orig[i], Catch::Matchers::WithinAbs(time[i], 1e-5f));
        REQUIRE_THAT(orig[i], Catch::Matchers::WithinAbs(timeA[i], 1e-5f));
        REQUIRE_THAT(orig[i], Catch::Matchers::WithinAbs(timeC[i], 1e-5f));
    }

    REQUIRE(true);
}
