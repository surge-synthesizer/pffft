//
// Created by Paul Walker on 12/29/22.
//


#define CATCH_CONFIG_RUNNER
#include "catch2.hpp"

int main( int argc, char* argv[] ) {
   int result = Catch::Session().run( argc, argv );

   return result;
}