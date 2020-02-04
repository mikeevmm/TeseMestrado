/**
 * Declaration of the `Option` and `Result` abstractions.
 * These patterns are meant to emulate the Rust `Some`, `None` and
 * `Option<>` patterns, with the added benefit that they allow for
 * elegant error handling, since the user can always decide whether
 * to return an invalid `Result` upstream, or `result_unwrap`, resulting
 * in a signalling exit that can be debugged via the `Result`'s
 * properties. (This `unwrap`/return `Result` pattern is also borrowed
 * from Rust).
 **/

#ifndef QOP_OPTION_H_
#define QOP_OPTION_H_

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct ErrorDetails {
  const char *reason;
  const char *file;
  unsigned int line;
} ErrorDetails;

typedef union ResultContent {
  void *data;
  ErrorDetails error_details;
} ResultContent;

typedef struct Result {
  bool valid;
  ResultContent content;
} Result;

// Creates a new valid `Result` object pointing to nowhere (`data = NULL`).
// This is useful for signalling, e.g., that a void function returned
// normally.
Result result_get_empty_valid(void);

// Creates a new valid result pointing to the given pointer.
Result result_get_valid_with_data(void *data);

// Creates a new invalid `Result` object containing a reason for failure,
// and the file and line number at which the error occurred.
// It is discouraged to use this function directly. Consider instead
// using the macro `result_get_invalid_reason`, which will specify the
// file and line no. automatically using the predefined `__FILE__` and
// `__LINE__` macros.
Result result_get_invalid_reason_raw(const char *reason, const char *file,
                                     unsigned int line);

// Creates a new invalid `Result` object containing the given reason for
// failure, and pointing to the file and line location at which the call
// for this macro was made.
// The `reason` string should be a constant literal.
// This is the preferred way to create an invalid `Result` object.
#define result_get_invalid_reason(reason) \
  result_get_invalid_reason_raw(reason, __FILE__, __LINE__)

// Examines the given `Result` object, returning the `void *` pointer it
// contains if it's valid, or, if it's invalid, reporting the reason and
// location before exiting with a non-zero exit code.
void *result_unwrap(Result result);

// Represents an optional `void *` pointer.
// If `some` is `false`, then `*data` is unspecified (and likely garbage).
typedef struct Option {
  bool some;
  void *data;
} Option;

typedef struct Option_Uint {
  bool some;
  unsigned int data;
} Option_Uint;

typedef struct Option_Int {
  bool some;
  int data;
} Option_Int;

typedef struct Option_Double {
  bool some;
  double data;
} Option_Double;

// Creates an `Option` without data.
// This is akin to Rust's `None`.
Option option_none(void);
Option_Int option_none_int(void);
Option_Uint option_none_uint(void);
Option_Double option_none_double(void);

// Create an `Option` pointing to the given `void *` pointer.
// This is akin to Rust's `Some(data)`.
Option option_some_with_data(void *data);
Option_Int option_from_int(int data);
Option_Uint option_from_uint(unsigned int data);
Option_Double option_from_double(double data);

#endif  // QOP_OPTION_H_