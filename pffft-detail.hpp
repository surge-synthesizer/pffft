#pragma once

#include <cstdint>
#include <memory>
#include <new>
#include <vector>

namespace pffft
{
namespace internal
{
#include "pffft.h"

// Utility function to make sure our inputs are powers of two.
// Can't use the Juce one because we're in a split-out library.
static constexpr bool IsPowerOfTwo(size_t x) { return x && (x & (x - 1)) == 0; }

// Utility function to check whether a given pointer is aligned on the given boundary.
template <typename T> inline bool is_aligned(T *ptr, std::size_t alignment)
{
    std::uintptr_t orig = reinterpret_cast<std::uintptr_t>(ptr);
    return !(orig % alignment);
}

// Aligned allocator. Taken from the Seqan3 library, licensed under BSD 3-clause.
// Copyright (c) 2006-2022, Knut Reinert & Freie Universität Berlin
// Copyright (c) 2016-2022, Knut Reinert & MPI für molekulare Genetik
template <typename value_t, std::size_t alignment_v = __STDCPP_DEFAULT_NEW_ALIGNMENT__>
class aligned_allocator
{
  public:
    static constexpr std::size_t alignment = alignment_v;

    using value_type = value_t;
    using pointer = value_type *;
    using difference_type = typename std::pointer_traits<pointer>::difference_type;
    using size_type = std::make_unsigned_t<difference_type>;

    using is_always_equal = std::true_type;

    aligned_allocator() = default;
    aligned_allocator(aligned_allocator const &) = default;
    aligned_allocator(aligned_allocator &&) = default;
    aligned_allocator &operator=(aligned_allocator const &) = default;
    aligned_allocator &operator=(aligned_allocator &&) = default;
    ~aligned_allocator() = default;

    template <class other_value_type, std::size_t other_alignment>
    constexpr aligned_allocator(
        aligned_allocator<other_value_type, other_alignment> const &) noexcept
    {
    }

    [[nodiscard]] pointer allocate(size_type const n) const
    {
        constexpr size_type max_size = std::numeric_limits<size_type>::max() / sizeof(value_type);
        if (n > max_size)
            throw std::bad_alloc{};

        std::size_t bytes_to_allocate = n * sizeof(value_type);
        if constexpr (alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__)
            return static_cast<pointer>(::operator new(bytes_to_allocate));
        else // Use alignment aware allocator function.
            return static_cast<pointer>(
                ::operator new(bytes_to_allocate, static_cast<std::align_val_t>(alignment)));
    }

    void deallocate(pointer const p, size_type const n) const noexcept
    {
        std::size_t bytes_to_deallocate = n * sizeof(value_type);

        // Clang doesn't have __cpp_sized_deallocation defined by default even though this is a
        // C++14! feature > In Clang 3.7 and later, sized deallocation is only enabled if the user
        // passes the `-fsized-deallocation` > flag. see also
        // https://clang.llvm.org/cxx_status.html#n3778
#if __cpp_sized_deallocation >= 201309
        // gcc
        if constexpr (alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__)
            ::operator delete(p, bytes_to_deallocate);
        else // Use alignment aware deallocator function.
            ::operator delete(p, bytes_to_deallocate, static_cast<std::align_val_t>(alignment));
#else  /*__cpp_sized_deallocation >= 201309*/
        // e.g. clang++
        if constexpr (alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__)
            ::operator delete(p);
        else // Use alignment aware deallocator function.
            ::operator delete(p, static_cast<std::align_val_t>(alignment));
#endif // __cpp_sized_deallocation >= 201309
    }

    template <typename new_value_type> struct rebind
    {
        static constexpr std::size_t other_alignment = std::max(alignof(new_value_type), alignment);
        using other = aligned_allocator<new_value_type, other_alignment>;
    };

    template <class value_type2, std::size_t alignment2>
    constexpr bool operator==(aligned_allocator<value_type2, alignment2> const &) noexcept
    {
        return alignment == alignment2;
    }

    template <class value_type2, std::size_t alignment2>
    constexpr bool operator!=(aligned_allocator<value_type2, alignment2> const &) noexcept
    {
        return alignment != alignment2;
    }
};

// Easy reference for aligned vectors.
template <typename T, std::size_t N>
using AlignedVector = typename std::vector<T, internal::aligned_allocator<T, N>>;

// Annoying MSVC bug work-around where it doesn't realize it has to call the
// aligned deleter. Use this for aligned arrays.
template <typename T, std::size_t Alignment>
struct AlignedArrayDeleter
{
    void operator()(T *p) const
    {
        if (p)
            ::operator delete[](p, std::align_val_t{Alignment});
    }
};

} // namespace internal

} // namespace pffft
