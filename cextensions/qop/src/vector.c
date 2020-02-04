#include "include/vector.h"

Result vector_init(Vector *v, size_t object_size, size_t init_capacity) {
  if (init_capacity > 0) {
    v->data = (void *)malloc(init_capacity * object_size);
    if (!v->data) return result_get_invalid_reason("could not malloc");
  } else {
    v->data = NULL;
  }

  v->size = 0;
  v->capacity = init_capacity;
  v->obj_size = object_size;
  v->init = true;

  return result_get_valid_with_data(v);
}

void *vector_get_raw(Vector *v, size_t index) {
  void *object = (void *)((char *)v->data + index * v->obj_size);
  return object;
}

Result vector_resize(Vector *v, size_t size) {
  if (!v->init) {
    return result_get_invalid_reason("tried to resize uninitialized vector");
  }

  if (size == v->capacity) {
    return result_get_valid_with_data(v);
  } else if (size > v->capacity) {
    // Resize to next power of 2
    size_t new_capacity = (v->capacity == 0 ? 1 : v->capacity);
    while (new_capacity < size) new_capacity <<= 1;

    void *resized;
    {
      if (v->data == NULL) {
        resized = malloc(new_capacity * v->obj_size);
        if (resized == NULL) {
          return result_get_invalid_reason(
              "could not resize vector; malloc failed");
        }
      } else {
        resized = realloc(v->data, new_capacity * v->obj_size);
        if (resized == NULL) {
          return result_get_invalid_reason(
              "could not resize vector; realloc failed");
        }
      }
    }
    v->data = resized;
    v->capacity = new_capacity;
    return result_get_valid_with_data(v);
  } else if (size <= v->capacity / 2) {
    size_t new_capacity = (v->capacity == 0 ? 1 : v->capacity);
    while (new_capacity / 2 > size) new_capacity >>= 1;

    void *resized;
    if (new_capacity == 0) {
      free(v->data);
      resized = NULL;
    } else {
      {
        if (v->data == NULL)
          resized = malloc(new_capacity * v->obj_size);
        else
          resized = realloc(v->data, new_capacity * v->obj_size);
      }
      if (resized == NULL) {
        free(v->data);
        v->init = false;
        return result_get_invalid_reason(
            "could not resize vector; realloc failed");
      }
    }
    v->data = resized;
    v->capacity = new_capacity;
    return result_get_valid_with_data(v);
  } else {  // Already at good size
    return result_get_valid_with_data(v);
  }
}

Result vector_push(Vector *v, void *object_ptr) {
  if (!v->init) {
    return result_get_invalid_reason("tried to push to uninitialized vector");
  }

  if (v->size + 1 > v->capacity) {
    Result resize = vector_resize(v, v->size + 1);
    if (!resize.valid) {
      return resize;
    }
  }
  return vector_raw_push(v, object_ptr);
}

Result vector_raw_push(Vector *v, void *object_ptr) {
  void *moved =
      memcpy((char *)v->data + v->obj_size * v->size, object_ptr, v->obj_size);
  if (!moved) {
    return result_get_invalid_reason("could not memcpy");
  }
  v->size += 1;
  return result_get_valid_with_data(moved);
}

Result vector_extend(Vector *v, void *head, size_t obj_count) {
  if (!v->init) {
    return result_get_invalid_reason("tried to extend uninitialized vector");
  }

  if (v->size + obj_count > v->capacity) {
    Result resize = vector_resize(v, v->size + obj_count);
    if (!resize.valid) {
      return resize;
    }
  }
  return vector_extend_raw(v, head, obj_count);
}

Result vector_extend_raw(Vector *v, void *head, size_t obj_count) {
  void *moved = memcpy((char *)v->data + v->obj_size * v->size, head,
                       v->obj_size * obj_count);
  if (!moved) {
    return result_get_invalid_reason("failed to memcpy");
  }
  v->size += obj_count;
  return result_get_valid_with_data(moved);
}

Result vector_pop(Vector *v, void *into) {
  if (!v->init) {
    return result_get_invalid_reason("tried to pop uninitialized vector");
  }

  if (into == NULL) {
    void *copied =
        memcpy(into, (char *)v->data + v->size * v->obj_size, v->obj_size);
    if (!copied) {
      return result_get_invalid_reason("could not memcpy");
    }
  }
  v->size -= 1;
  vector_resize(v, v->size);

  return result_get_valid_with_data(into);
}

Result vector_clean(Vector *v) {
  free(v->data);
  v->data = NULL;
  v->capacity = 0;
  v->size = 0;
  return result_get_empty_valid();
}

void vector_free(Vector *v) {
  vector_clean(v);
  v->init = false;
}

static Option vector_iter_next_fn(Iter *iter, unsigned int pos) {
  Vector *v = (Vector *)iter->context;
  if (iter->position >= v->size) return option_none();
  void *object = (void *)((char *)v->data + iter->position * v->obj_size);
  return option_some_with_data(object);
}

Iter vector_iter_create(Vector *v) {
  if (!v->init) {
    return iter_get_empty();
  }

  Iter new_iter;
  new_iter.context = (void *)v;
  new_iter.position = 0;
  new_iter.next_fn = vector_iter_next_fn;
  new_iter.free_fn = NULL;
  return new_iter;
}

Result filter_into_vector(Filter *filter, Vector *vector) {
  if (filter == NULL || vector == NULL) {
    return result_get_invalid_reason("filter and/or vector are NULL");
  }

  if (!vector->init) {
    return result_get_invalid_reason(
        "tried to filter into uninitialized vector");
  }

  Option next;
  while ((next = filter_next(filter)).some) {
    Result push_r = vector_push(vector, next.data);
    if (!push_r.valid) {
      vector_clean(vector);
      return push_r;
    }
  }

  return result_get_valid_with_data(vector);
}
