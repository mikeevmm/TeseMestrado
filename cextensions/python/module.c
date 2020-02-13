#include "include/module.h"

#define EARLY_EXIT              \
  {                             \
    printf("Exiting early!\n"); \
    Py_INCREF(Py_None);         \
    return Py_None;             \
  }

static void reprint(PyObject *obj) {
  PyObject *repr = PyObject_Repr(obj);
  PyObject *str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
  const char *bytes = PyBytes_AS_STRING(str);

  printf("%s\n", bytes);

  Py_XDECREF(repr);
  Py_XDECREF(str);
}

int sort_ham_term_by_syslen(const void *a_addr, const void *b_addr) {
  PyObject *a = *(PyObject **)a_addr;
  PyObject *b = *(PyObject **)b_addr;

  // Each hamiltonian term is of the form (coef, [operators], [systems])
  // We are interested in oredering the terms from greatest number of
  // systems to least number
  Py_ssize_t a_len = PyList_GET_SIZE(PyList_GetItem(a, 2));
  Py_ssize_t b_len = PyList_GET_SIZE(PyList_GetItem(b, 2));

  return (int)b_len - (int)a_len;
}

int sort_partition_by_score(const void *a, const void *b) {
  return (int)((PartitionScore *)b)->score - (int)((PartitionScore *)a)->score;
}

int sort_uint_ascending(const void *a, const void *b) {
  return (int)(*(long int *)a - *(long int *)b);
}

// Initialization function of the module.
// Here we initialize not only the module, but also import the NumPy
// functions (via `import_array()`), and instantiate the internal error
// module exception type, `ModuleError`.
PyMODINIT_FUNC PyInit_cextension() {
  PyObject *mod;

  mod = PyModule_Create(&module);
  if (mod == NULL) return NULL;

  import_array();
  if (PyErr_Occurred()) {
    return NULL;
  }

  ModuleError = PyErr_NewException("cextension.error", NULL, NULL);
  Py_XINCREF(ModuleError);
  if (PyModule_AddObject(mod, "error", ModuleError) < 0) {
    Py_XDECREF(ModuleError);
    Py_CLEAR(ModuleError);
    Py_DECREF(mod);
    return NULL;
  }

  return mod;
}

static PyObject *find_used_partitions(PyObject *self, PyObject *args) {
  // Parse arguments; we are expecting
  // - a hamiltonian as described by a collection of terms
  //    of structure
  //      (coef, [operators], [corresponding systems])
  // - the number of systems in the hamiltonian
  // - the maximum locality allowed for the partitions

  PyObject **hamiltonian;
  unsigned int term_count;
  unsigned int system_num, locality;
  {
    PyObject *hamiltonian_object, *hamiltonian_iter;
    if (!PyArg_ParseTuple(args, "OII", &hamiltonian_object, &system_num,
                          &locality))
      return NULL;

    // Check the integer parameters
    if (locality > system_num) {
      PyErr_SetString(ModuleError,
                      "locality cannot be greater than system_num");
    }

    // Create an array of references to the elements of the hamiltonian

    // PySequence_Fast returns a new reference; this should be decref'd
    // at some point
    hamiltonian_iter =
        PySequence_Fast(hamiltonian_object, "argument must be iterable");
    if (!hamiltonian_iter) return NULL;

    term_count = PySequence_Fast_GET_SIZE(hamiltonian_iter);
    hamiltonian = malloc(term_count * sizeof(PyObject *));
    if (!hamiltonian) {
      Py_DECREF(hamiltonian_iter);
      return PyErr_NoMemory();
    }

    for (unsigned int i = 0; i < term_count; i++) {
      // This returns a borrowed reference to the element of the
      // given hamiltonian list; because this reference should only
      // live for the scope of this function, this is fine and allows
      // us not to have to decref the reference.
      PyObject *item = PySequence_Fast_GET_ITEM(hamiltonian_iter, i);
      if (!item) {
        Py_DECREF(hamiltonian_iter);
        free(hamiltonian);
        return NULL;
      }

      hamiltonian[i] = item;
    }

    Py_DECREF(hamiltonian_iter);
  }

  // Get all combinations of `system_num` choose `locality`
  // A partition is described by an array of indexes
  unsigned long long partition_count =
      choose((unsigned long long)system_num, (unsigned long long)locality);
  unsigned int *partitions;
  {
    void *allocation =
        malloc(partition_count * locality * sizeof(unsigned int));
    if (!allocation) {
      free(hamiltonian);
      return PyErr_NoMemory();
    }
    partitions = (unsigned int *)allocation;
  }

  {
    // Prepare to run over all `locality` sized subsets of `0..system_num`
    int x = 0, y = 0, z = 0;

    unsigned int subset[locality];
    for (unsigned int i = 0; i < locality; ++i) {
      subset[i] = system_num - locality + i;
      *(partitions + i) = subset[i];
    }

    qsort(partitions, locality, sizeof(unsigned int), sort_uint_ascending);

    int p[system_num + 2];
    inittwiddle(locality, system_num, p);

    unsigned int partition_index = 1;
    while (!twiddle(&x, &y, &z, p)) {
      subset[z] = (unsigned int)x;

      for (unsigned int j = 0; j < locality; ++j) {
        unsigned int *partition = partitions + partition_index * locality + j;
        *(partition) = subset[j];
      }

      // Save the partitions in sorted order
      qsort(partitions + partition_index * locality, locality,
            sizeof(unsigned int), sort_uint_ascending);

      partition_index++;
    }

    assert(partition_index == partition_count);
  }  // Finished calculating all possible partitions

  // Define partition score
  unsigned int best_score = 0;
  PartitionScore *partition_score;
  {
    void *allocation = malloc(partition_count * sizeof(PartitionScore));
    if (!allocation) {
      free(hamiltonian);
      free(partitions);
      return PyErr_NoMemory();
    }
    partition_score = (PartitionScore *)allocation;

    for (unsigned int i = 0; i < partition_count; ++i) {
      partition_score[i] = (PartitionScore){
          .partition_index = i,
          .score = 0,
      };
    }
  }

  // Vector of the terms split into the corresponding partition
  // (per index of partition)
  Vector *split_terms;
  {
    void *allocation = malloc(partition_count * sizeof(Vector));
    if (!allocation) {
      free(hamiltonian);
      free(partitions);
      free(partition_score);
      return PyErr_NoMemory();
    }
    split_terms = allocation;
  }

  for (unsigned int i = 0; i < partition_count; ++i) {
    Vector empty;
    Result init_result = vector_init(&empty, sizeof(CoefTermIndexPair), 0);
    if (!init_result.valid) {
      free(hamiltonian);
      PyErr_SetString(ModuleError, init_result.content.error_details.reason);
      return NULL;
    }
    split_terms[i] = empty;
  }

  // Sort the terms from greater number of systems to least number of systems
  // NOTE: WE ASSUME THE HAMILTONIAN IS ALREADY SORTED LIKE THIS
  // qsort(hamiltonian, term_count, sizeof(PyObject *),
  // sort_ham_term_by_syslen);

  // Now that the terms are sorted from most # of involved systems to least,
  // attribute score to the top scoring partitions that fit the terms
  for (unsigned int term_index = 0; term_index < term_count; ++term_index) {
    PyObject *term = hamiltonian[term_index];

    // Extract info from term
    double coef;
    PyObject *operators;
    PyObject *systems;
    {
      coef = PyFloat_AS_DOUBLE(PyList_GetItem(term, 0));
      operators = PyList_GetItem(term, 1);
      systems = PyList_GetItem(term, 2);
    }

    // Used to iterate over the systems in `sytems`;
    // `PySequence_Fast` returns a new reference
    PyObject *systems_fastseq =
        PySequence_Fast(systems, "systems must be iterable");
    if (!systems_fastseq) {
      // Could not instantiate iterable over `systems`
      for (unsigned int i = 0; i < partition_count; ++i) {
        Vector *split_term = split_terms + i;
        vector_free(split_term);
      }
      free(split_terms);
      free(hamiltonian);
      free(partitions);
      free(partition_score);
      return NULL;
    }

    Py_ssize_t systems_len = PySequence_Fast_GET_SIZE(systems_fastseq);

    //printf("SYSTEMS: [");
    //for (int i = 0; i < systems_len; ++i)
    //  printf("%li, ", PyNumber_AsSsize_t(
    //                      PySequence_Fast_GET_ITEM(systems_fastseq, i), NULL));
    //printf("]\n");

    // Go through the partitions in descending score
    qsort(partition_score, partition_count, sizeof(PartitionScore),
          sort_partition_by_score);

    // Partitions of top score in which the term fits
    Vector best;
    vector_init(&best, sizeof(unsigned int), 0);

    Option_Uint fitting_score = option_none_uint();
    for (unsigned int partscore_index = 0; partscore_index < partition_count;
         ++partscore_index) {
      unsigned int partition_index =
          partition_score[partscore_index].partition_index;
      unsigned int score = partition_score[partscore_index].score;
      unsigned int *partition = partitions + partition_index * locality;

      // Skip this partition if it can't contain the term.
      // Because the indices in partition are sorted, we
      // perform a binary search of partition for each system
      bool skip_partition = false;
      {
        // Check each system in `systems`
        for (unsigned int sys_index = 0; sys_index < systems_len; ++sys_index) {
          unsigned int system = (unsigned int)PyNumber_AsSsize_t(
              PySequence_Fast_GET_ITEM(systems_fastseq, sys_index), NULL);

          // We can skip the binary search immediately if the system is out of
          // bounds to the partition
          if (system < partition[0] || system > partition[locality - 1]) {
            skip_partition = true;
            break;  // From checking other systems
          }

          // Perform actual binary search in the partition for the system
          {
            unsigned int low = 0;
            unsigned int high = locality;

            while (true) {
              if (low > high) {
                // Unsuccessful, partition does not contain `system`
                skip_partition = true;
                break;  // From binary search loop
              }

              unsigned int middle = (low + high) / 2;
              unsigned int middle_elem = partition[middle];

              if (middle_elem < system) {
                low = middle + 1;
                continue;  // To next binary search loop
              }

              if (middle_elem > system) {
                high = middle - 1;
                continue;  // To next binary search loop
              }

              if (middle_elem == system) {
                // The partition contains the system; seach for next system
                break;  // From searching this system
              }
            }  // End of binary search loop

            if (skip_partition) break;  // From searching others systems
          }                             // Done performing binary search
        }  // Searched the partition for all systems

      }  // Finished assessing whether to skip this partition

      if (skip_partition) {
        //printf("\tSKIPPING [");
        //for (int i = 0; i < locality; ++i) printf("%u, ", partition[i]);
        //printf("]\n");
        continue;  // To next partition
      }

      //printf("\tVALID [");
      //for (int i = 0; i < locality; ++i) printf("%u, ", partition[i]);
      //printf("]\n");

      // This partition is valid;
      // If it's the first valid partition we see, continue onto adding
      // it to the `split_terms`, otherwise, only add it if the score
      // matches last seen fitting partition.
      // Because partitions are in descending score order, if this partition
      // has a lower score than previously seen, immediately stop iterating
      // over the partitions
      if (!fitting_score.some) {
        fitting_score = option_from_uint(score);
      } else if (score < fitting_score.data) {
        break;
      }

      vector_push(&best, &partition_index);

      // Increase the partition's score
      // Note that because `partition_score` was sorted in advance, this
      // doesn't interfere with the iteration over partitions
      partition_score[partscore_index].score += 1;

    }  // Done iterating over partitions (in descending score)

    // "Free" (decref) the iterator over the systems
    Py_DECREF(systems_fastseq);

    // Could not find any partitions to fit the term into? This should not
    // happen...
    if (best.size == 0) {
      PyErr_SetString(ModuleError,
                      "Could not find any partition to fit a term? This should "
                      "not happen!");
      for (unsigned int i = 0; i < partition_count; ++i) {
        Vector *split_term = split_terms + i;
        vector_free(split_term);
      }
      free(split_terms);
      free(hamiltonian);
      free(partitions);
      free(partition_score);
      return NULL;
    }

    // Split the coefficient over all possible partitions
    for (unsigned int best_index = 0; best_index < best.size; ++best_index) {
      unsigned int partition_index =
          *(unsigned int *)vector_get_raw(&best, best_index);

      CoefTermIndexPair value = (CoefTermIndexPair){
          .coef = coef / (double)(best.size),
          .term_index = term_index,
      };

      Vector *split_term = split_terms + partition_index;
      vector_push(split_term, &value);
    }

  }  // Done iterating over terms (in descending system count)

  // Cull partitions that did not have terms

  // Count how many non-empty split_terms exist
  unsigned int non_empty_count = 0;
  for (unsigned int partition_index = 0; partition_index < partition_count;
       ++partition_index) {
    if (split_terms[partition_index].size != 0) non_empty_count++;
  }

  // Prepare tuple to return; each element of the tuple will be another
  // 2-tuple of strucure (<partition>, (<coef, term> pairs))
  PyObject *return_tuple = PyTuple_New(non_empty_count);

  unsigned int return_tuple_index = 0;
  for (unsigned int partition_index = 0; partition_index < partition_count;
       ++partition_index) {
    Vector *split_term = split_terms + partition_index;
    if (split_term->size == 0) continue;

    PyObject *item_tuple = PyTuple_New(2);

    // Create a partition as a tuple
    PyObject *partition_tuple = PyTuple_New(locality);
    {
      unsigned int *partition = partitions + partition_index * locality;
      for (unsigned int i = 0; i < locality; ++i) {
        PyTuple_SetItem(partition_tuple, i, PyLong_FromSsize_t(partition[i]));
      }
    }

    // Create the split_term vector as tuple
    PyObject *split_term_pairs = PyTuple_New(split_term->size);
    for (unsigned int i = 0; i < split_term->size; ++i) {
      PyObject *coef_term_pair_tuple = PyTuple_New(2);
      CoefTermIndexPair pair =
          *(CoefTermIndexPair *)vector_get_raw(split_term, i);
      PyTuple_SetItem(coef_term_pair_tuple, 0, PyFloat_FromDouble(pair.coef));
      PyTuple_SetItem(coef_term_pair_tuple, 1,
                      PyLong_FromSsize_t(pair.term_index));
      PyTuple_SetItem(split_term_pairs, i, coef_term_pair_tuple);
    }

    PyTuple_SetItem(item_tuple, 0, partition_tuple);
    PyTuple_SetItem(item_tuple, 1, split_term_pairs);
    PyTuple_SetItem(return_tuple, return_tuple_index, item_tuple);
    return_tuple_index++;
  }

  // Free the heap allocations
  for (unsigned int i = 0; i < partition_count; ++i) {
    Vector *split_term = split_terms + i;
    vector_free(split_term);
  }
  free(split_terms);
  free(hamiltonian);
  free(partitions);
  free(partition_score);

  return return_tuple;
}

static PyObject *decimate(PyObject *self, PyObject *args) {
  
}