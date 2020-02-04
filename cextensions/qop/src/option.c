#include "include/option.h"

Result result_get_empty_valid() {
  Result empty_valid;
  empty_valid.valid = true;
  empty_valid.content.data = NULL;
  return empty_valid;
}

Result result_get_valid_with_data(void *data) {
  Result data_valid;
  data_valid.valid = true;
  data_valid.content.data = data;
  return data_valid;
}

Result result_get_invalid_reason_raw(const char *reason, const char *file,
                                     unsigned int line) {
  Result invalid;
  invalid.valid = false;
  invalid.content.error_details.reason = reason;
  invalid.content.error_details.file = file;
  invalid.content.error_details.line = line;
  return invalid;
}

void *result_unwrap(Result result) {
  if (!result.valid) {
    ErrorDetails err_det = result.content.error_details;
    printf("%s at %s:%d\n", err_det.reason, err_det.file, err_det.line);
    exit(1);
  }

  return result.content.data;
}

Option option_none() {
  Option new_option;
  new_option.some = false;
  new_option.data = NULL;
  return new_option;
}

Option_Int option_none_int() {
  Option_Int new_option;
  new_option.some = false;
  return new_option;
}

Option_Uint option_none_uint() {
  Option_Uint new_option;
  new_option.some = false;
  return new_option;
}

Option_Double option_none_double() {
  Option_Double new_option;
  new_option.some = false;
  return new_option;
}

Option option_some_with_data(void *data) {
  Option new_option;
  new_option.some = true;
  new_option.data = data;
  return new_option;
}

Option_Int option_from_int(int data) {
  Option_Int new_option;
  new_option.some = true;
  new_option.data = data;
  return new_option;
}

Option_Uint option_from_uint(unsigned int data) {
  Option_Uint new_option;
  new_option.some = true;
  new_option.data = data;
  return new_option;
}

Option_Double option_from_double(double data) {
  Option_Double new_option;
  new_option.some = true;
  new_option.data = data;
  return new_option;
}