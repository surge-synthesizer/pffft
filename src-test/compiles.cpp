#include <catch2.hpp>
#include "pffft.hpp"

TEST_CASE("PFFT Compiles", "[basics]")
{
   pffft::FFT<float, 256> val;
   REQUIRE(true);
}