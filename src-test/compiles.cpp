#include <catch2.hpp>
#include "pffft.hpp"

TEST_CASE("PFFFT Compiles", "[basics]")
{
   pffft::FFT<float, 256> val;
   REQUIRE(true);
   pffft::FFT<float, std::dynamic_extent> dval(32);
   dval.resize(64);
}
