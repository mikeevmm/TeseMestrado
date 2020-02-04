/**
 * This file defines the Vector collection structure, akin to C++'s
 * `std::Vector`, Python's `list`, etc.
 * Because C does not support template metaprogramming (or otherwise
 * generic type definitions), this Vector structure operates on `void *`
 * pointers, and it is the user's responsability to cast to the
 * appropriate types.
 **/

#ifndef QOP_VECTOR_H_
#define QOP_VECTOR_H_

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "include/iter.h"

typedef struct Vector {
  bool init;
  void *data;
  size_t capacity;
  size_t size;
  size_t obj_size;
} Vector;

// Initializes the specified vector to accommodate objects of the given
// size, and allocating enough memory to hold `init_capacity` objects
// without needing to reallocate memory.
// Note that initializing an already initialized vector will leak memory;
// if reusing the same location in memory, call `vector_free` first.
// Note also that it's always ok to specify an initial capacity of 0.
// This will just cause a greater number of allocations.
Result vector_init(Vector *v, size_t object_size, size_t init_capacity);

// Gets a pointer to the element at position `index` in the vector.
// No checks are made regarding the bounds validity of `index`, making
// this an unsafe function. If possible, consider making an iterator
// over the contents of the vector (see `iter.h`).
void *vector_get_raw(Vector *v, size_t index);

// Reallocates memory so that the vector can hold `size` objects.
// This function is automatically called by `vector_push`, `vector_pop`
// and `vector_extend`, so it's best used if just called when about to
// push a known number of elements exceeding the vector's capacity.
Result vector_resize(Vector *v, size_t size);

// Pushes a new element into the vector.
// This is essentially a wrapper around `vector_raw_push` with some
// safety (bounds, initialization, size) checks, so if the extra cycles
// are critical and you can ensure that the vector can accommodate the
// new element, consider using `vector_raw_push` instead.
// Note also that the object is `memcpy`d into the vector's memory, so
// that the object may be safely freed or dropped out of scope after
// pushing.
Result vector_push(Vector *v, void *object);

// Pushes a new element into the vector without any safety checks.
// Consider using `vector_push` instead, which performs some safety
// checks, and will allocate memory for the new object if needed.
// Note that the object is `memcpy`d into the vector, so that it may be
// safely freed or dropped from scope after pushing.
Result vector_raw_push(Vector *v, void *object);

// Extends the vector by all the elements in the given collection.
// The collection is specified by it's starting point in memory (`head`),
// and the amount of objects in the collection. The `i`th object is taken
// by striding `i` times from `head` in `v->obj_size` increments.
// Note that, like in pushing to the vector, the elements are `memcpy`d
// into the array, and may subsequently be freed or dropped from scope.
// This function is a wrapper to `vector_extend_raw`, with safety checks
// and allocating memory if needed. If performance is critical, and you
// can ensure that a raw `memcpy` into the vector memory region is valid,
// consider using `vector_extend_raw`.
Result vector_extend(Vector *v, void *head, size_t obj_count);

// Extends the vector by all the elements in the given collection,
// without performing any kind of safety checks regarding vector state
// or memory.
// Consider using `vector_extend` instead, as it wraps this function but
// assuring that the operation is legal.
// The given collection is specified by its starting position in memory
// and the object count, such that the `i`th element is ta taken by
// striding `i` times from `head` in `v->obj_size` increments.
Result vector_extend_raw(Vector *v, void *head, size_t obj_count);

// Moves the last allocated element in the vector into the provided location,
// returning the pointer to that location in memory, and
// resizes the vector accordingly.
// Note that it is the user's responsability to ensure that the target
// location is appropriate, and then free the popped element memory block.
// If the copy is not to be made at all (and the popped value is to be
// discarded), it is safe to pass `NULL` as `into`.
Result vector_pop(Vector *v, void *into);

// Removes all the elements from the vector (freeing the corresponding
// heap memory), zeroing the capacity of the vector.
// This function may be useful to reuse the same `Vector` variable,
// whereas one would first `vector_clean` it before calling `vector_init`.
Result vector_clean(Vector *v);

// Cleans the vector and marks it as non-initialized.
// There is little difference between this function and `vector_clean`,
// but by explicitly freeing the vector (rather than cleaning),
// subsequent operations are not allowed as a sanity check measure.
void vector_free(Vector *v);

// Creates a new iterator from a vector.
// Creates an iterator with the head matching the vector's head, with a
// stride matching the vector's object size, and with length matching
// the vector's `size`.
// The iterator will point to the vector's memory space, so that
// modifying these elements will modify the vector.
// See `iter.h` for more information on iterators.
Iter vector_iter_create(Vector *v);

// Makes a vector out of all of the elements of a filter.
// This is done by creating a new vector and iterating over the filter,
// pushing yielded elements into the vector; there's no optimization to
// it, and you may choose to do this explicitly, in order to increase
// control.
// Nonetheless this is a common operation and so is abstracted into this
// convenience function.
Result filter_into_vector(Filter *filter, Vector *vector);

#endif  // QOP_VECTOR_H