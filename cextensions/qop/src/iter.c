#include "include/iter.h"

Option iter_next(Iter *iter) {
  Option next = iter->next_fn(iter, iter->position);
  iter->position += 1;
  return next;
}

void iter_free(Iter *iter) {
  if (iter->free_fn != NULL) iter->free_fn(iter);
}

static Option iter_empty_next_fn(Iter *iter, unsigned int pos) {
  return option_none();
}

Iter iter_get_empty() {
  Iter empty_iter;
  empty_iter.position = 0;
  empty_iter.next_fn = &iter_empty_next_fn;
  empty_iter.free_fn = NULL;
  return empty_iter;
}

static Option iter_contiguous_memory_next_fn(Iter *iter, unsigned int pos) {
  unsigned int size = (unsigned int)iter->context_values[1];
  if (pos >= size) return option_none();
  unsigned int stride = (unsigned int)iter->context_values[0];
  return option_some_with_data((void *)((char *)iter->context + stride * pos));
}

Iter iter_create_contiguous_memory(void *head, unsigned int stride,
                                   unsigned int size) {
  Iter iter;
  iter.position = 0;
  iter.context = head;
  iter.context_values[0] = (int)stride;
  iter.context_values[1] = (int)size;
  iter.next_fn = &iter_contiguous_memory_next_fn;
  iter.free_fn = NULL;
  return iter;
}

Filter filter_create(Iter iter, FilterFn filter_fn) {
  Filter new_filter;
  new_filter.iter = iter;
  new_filter.filter_fn = filter_fn;
  new_filter.position = 0;
  return new_filter;
}

Option filter_next(Filter *filter) {
  Option next = iter_next(&filter->iter);
  if (!next.some) {
    return option_none();
  }

  while (next.some && !((filter->filter_fn)(next.data))) {
    next = iter_next(&filter->iter);
  }
  filter->position += 1;
  return next;
}
