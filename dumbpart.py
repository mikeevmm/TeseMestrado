#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""dumbpart.py

Calculates a lower bound to the ground state energy of a Hamiltonian by
adding up the ground state of each of the terms in the hamiltonian.

Usage:
  dumbpart.py <directory> <glob_pattern> [--outfile=<of>]
  dumbpart.py (-h | --help)

Options:
  -h --help       Show this text.
  --outfile=<of>  The file to write the lowerbounds to [default: dumbpart.txt].

"""

import sys
import os
import numpy as np
import numpy.linalg
from lowerbound import *
from docopt import docopt
from glob import glob

if __name__ == "__main__":
    arguments = docopt(__doc__, version="Dumbpart 1.0") 

    # Find filename; verify it exists
    directory = arguments['<directory>'] 
    if not os.path.exists(directory):
        print(f"{directory} does not exist")
        exit(1)

    # For each .molecule file in the directory, calculate a ground
    # state lower bound.

    glob_pattern = arguments['<glob_pattern>']
    filename_lb_pairs = []

    for filename in glob(f"{directory}/{glob_pattern}"):
        with open(filename) as infile_io:
            lower_bound = 0
            for line in infile_io:
                if line.startswith('#'):
                    continue

                term = parse_hamiltonian_line(line)
                if len(term[2]) == 0:
                    gs = term[0]
                else:
                    # Key the term so that indexes are those of the term
                    term[2] = [i for i in range(len(term[2]))]
                    matrix = matrix_from_terms([term], len(term[2]))
                    gs = np.min(np.linalg.eigvals(matrix.toarray())).real
                lower_bound += gs
            
            filename_lb_pairs.append((filename, lower_bound))

    # Write the results into the outfile
    outfile = arguments['--outfile']
    with open(outfile, 'w') as outfile_io:
        outfile_io.write("# <filename> \t <lower bound> \n")
        for filename, lb in filename_lb_pairs:
            outfile_io.write(f"{filename}\t{lb}\n")
