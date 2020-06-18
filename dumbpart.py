#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""dumbpart.py

Calculates a lower bound to the ground state energy of a Hamiltonian by
adding up the ground state of each of the terms in the hamiltonian.

Usage:
  dumbpart.py <directory> [--outfile=<of>] [--postprocess=<pp>] [--max-loc=<maxloc>]
  dumbpart.py (-h | --help)

Options:
  -h --help            Show this text.
  --outfile=<of>       The file to write the lowerbounds to [default: dumbpart.txt].
  --postprocess=<pp>   The regex element to apply to the filename.
                       The first capture group is considered.
                       Will print '-' if no group is found
                       [default: .+?\\(([\\d\\.]+),\\).+].
  --max-loc=<maxloc>   The max locality to allow [default: 0].

"""

import sys
import os
import numpy as np
import numpy.linalg
import re
import json_tricks as json
from lowerbound import *
from docopt import docopt
from glob import glob

if __name__ == "__main__":
    arguments = docopt(__doc__, version="Dumbpart 1.0") 
    print(arguments)

    # Find filename; verify it exists
    directory = arguments['<directory>'] 
    if not os.path.exists(directory):
        print(f"{directory} does not exist")
        exit(1)

    # For each .molecule file in the directory, calculate a ground
    # state lower bound.

    post_process = None
    if arguments['--postprocess']:
        post_process = re.compile(arguments['--postprocess'])
    filename_lb_pairs = []

    for filename in glob(f"{directory}/*.hamiltonian"):
        lower_bound = 0
        by_discarding = 0
        with open(filename + ".sw") as infile_io:
            hamiltonian = json.load(infile_io)
            for term in hamiltonian:
                if len(term[2]) == 0:
                    gs = term[0]
                else:
                    # Key the term so that indexes are those of the term
                    term[2] = [i for i in range(len(term[2]))]
                    matrix = matrix_from_terms([term], len(term[2]))
                    gs = np.min(np.linalg.eigvals(matrix.toarray())).real
                lower_bound += gs
                #if len(term[2]) <= int(arguments['--max-loc']):
                #    by_discarding += gs

        with open(filename) as infile_io:
            for line in infile_io:
                if line.startswith('#'):
                    continue
                term = parse_hamiltonian_line(line)
                if len(term[2]) > int(arguments['--max-loc']):
                    continue
                if len(term[2]) == 0:
                    gs = term[0]
                else:
                    # Key the term so that indexes are those of the term
                    term[2] = [i for i in range(len(term[2]))]
                    matrix = matrix_from_terms([term], len(term[2]))
                    gs = np.min(np.linalg.eigvals(matrix.toarray())).real
                by_discarding += gs

        # Post-process the filename
        proc_filename = filename
        if post_process:
            match = post_process.match(filename)
            try:
                proc_filename = match.group(1)
            except:
                proc_filename = '-'

        filename_lb_pairs.append((proc_filename, lower_bound, by_discarding))

    # Write the results into the outfile
    outfile = arguments['--outfile']
    with open(outfile, 'w') as outfile_io:
        for argument in arguments:
            outfile_io.write(f"# {argument}: {arguments[argument]}\n")
        outfile_io.write("# <filename> \t <lower bound>\t<discarding> \n")
        for filename, lb, dsc in filename_lb_pairs:
            outfile_io.write(f"{filename}\t{lb}\t{dsc}\n")

    print("Done, have a nice day!")
