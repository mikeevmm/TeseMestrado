#ifndef _MODULE_H_
#define _MODULE_H_

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stdbool.h>
#include <stdlib.h>
#include <complex.h>
#include <numpy/arrayobject.h>
#include "include/combinations.h"
#include "include/option.h"
#include "include/twiddle.h"
#include "include/vector.h"

typedef struct {
  unsigned int partition_index;
  unsigned int score;
} PartitionScore;

typedef struct {
  double coef;
  unsigned int term_index;
} CoefTermIndexPair;

static PyObject *find_used_partitions(PyObject *self, PyObject *args);

static PyObject *ModuleError;

PyMODINIT_FUNC PyInit_cextension(void);

static PyMethodDef mod_methods[] = {
    {"find_used_partitions", find_used_partitions, METH_VARARGS,
     "Find what partitions are used to fit a hamiltonian in "
     "the least number of partitions possible.\n"
     "\n"
     "Usage: \n"
     "\n"
     "find_used_partitions(hamiltonian, system_num, locality)\n"
     "\n"
     "NOTE: hamiltonian is assumed to be sorted in descending "
     "number of interacting systems! If this is not the case, "
     "behaviour is undefined."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "cextension",
    "Implementation of routines in C for use in python project.", -1, mod_methods};

#endif  // _MODULE_H_
