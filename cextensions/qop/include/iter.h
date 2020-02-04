/**
 * Declaration of the generic iterator structure and associated types.
 **/

#ifndef QOP_ITER_H_
#define QOP_ITER_H_

#include <stdbool.h>
#include <stdlib.h>
#include "include/option.h"

// Represents an iteration over an iterable memory region.
typedef struct Iter {
  unsigned int position;
  void *context;  // A pointer to an object which is contextually relevant
  int context_values[4];  // values of arbitrary "storage"
  // (Iter *, position)
  Option (*next_fn)(struct Iter *, unsigned int);
  // (Iter *)
  void (*free_fn)(struct Iter *);
} Iter;

// Moves the iterator object `iter` one iteration forward, returning a
// pointer to the object at the new iterator position.
// Note that the result is returned in the form of an `Option` object,
// such that if the iterator is not empty, the `Option` object's `data`
// property will be a pointer to the iterator's yielded pointer; a
// dereference of the `Result`'s data pointer will yield a pointer to
// the iteration data.
Option iter_next(Iter *iter);

// Frees any internal memory that the iterator object may have allocated.
// Note that the iterator may not have allocated any memory, in which case
// this function will not do anything and is in principle optimized out of
// the compiled program.
void iter_free(Iter *iter);

// Returns an empty (length-zero) iterator.
// This can be useful if, for example, an iterator is made from an empty
// `Vector` (see `Vector.h`).
Iter iter_get_empty(void);

// Returns an iterator over a contiguous region in memory.
Iter iter_create_contiguous_memory(void *head, unsigned int stride,
                                   unsigned int size);

// Filter function pointer signature. Of signature
//    `<pointer to> ((void *)<iteration element pointer>) -> <include?>`
typedef bool (*FilterFn)(void *);

// Iterator that refers to another iterator, and only yields items that
// evaluate to `true` under the filter function.
typedef struct Filter {
  Iter iter;
  FilterFn filter_fn;
  unsigned int position;
} Filter;

// Creates a `Filter` struct from an iterator object and a filter function.
// The iterator is passed by value into the `Filter`, and so modifying
// the original iterator (such as iterating over it) has no effect on
// the filter.
// This is not true of `filter_fn`, as it is merely a pointer to a
// filtering function.
Filter filter_create(Iter iter, FilterFn filter_fn);

// Iterate the given filter's associated iterator until either
//  - a yielded element of the iterator passes the filter function
//    (evaluates to `true`), or
//  - the iterator runs out if elements
// In the former case, an `Option` object with `data` pointing to
// *the pointer* to the valid element is given, while in the latter
// a none `Option` is yielded.
// Once the underlying iterator runs out, the none `Option` is always
// yielded.
Option filter_next(Filter *filter);

#endif  // QOP_ITER_H_