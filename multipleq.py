#!/usr/bin/env python3

import os
import multiprocessing
import subprocess
import json
import gc
from copy import deepcopy

__doc__ = """\
Calculate lower bounds for different max localities.

Usage:
  multipleq.py [--force | -f] [flags] <template file>
  multipleq.py --template <new template file>

Options:
    --force -f      Do not ask about overriding files/
                    folders.
    --template      Create new template file.
    --parallel      Run the processes in parallel.
    --no-fail       Do not raise an exception if stderr
                    from subprocess is non empty.
    [flags]         Flags to pass to lowerbound.py.
                    See lowerbound.py for details.
"""

if __name__ == '__main__':
    arguments = os.sys.argv
    if len(arguments) == 1 or '-h' in arguments or '--help' in arguments:
        print(__doc__)
        exit(0)
    arguments = arguments[1:]
    root_path = os.path.dirname(os.path.realpath(__file__))

    if '--template' in arguments:
        try:
            target_file = next(
                filter(lambda x: not x.startswith('-'), arguments))
        except StopIteration:
            print("Need a target file.")
            exit(1)
        if os.path.exists(target_file):
            print("File already exists.")
            exit(1)
        # Write a single qubit template
        template_write = subprocess.Popen(
            [f'{root_path}/lowerbound.py', '--template', target_file])
        template_write.communicate()

        # Change the qubit field to be an array of qubits
        with open(target_file, 'r') as target_fileio:
            template_json = target_fileio.read()
            template_json = template_json.replace(
                r'"qubits": ', r'"qubits": []')
        with open(target_file, 'w') as target_fileio:
            target_fileio.write(template_json)
        exit(0)

    do_force = '-f' in arguments or '--force' in arguments

    # Get the molecule file
    try:
        molecule_json = next(
            filter(lambda x: not x.startswith('-'), arguments))
    except StopIteration:
        print("Need a molecule file.")
        exit(1)

    if not os.path.exists(molecule_json):
        print(f"Could not find {molecule_json}")
        exit(1)

    with open(molecule_json, "r") as file:
        molecule_json = json.load(file)

    # Extract some used properties from the json
    qubits = molecule_json['circuit']['qubits']
    molecule_name = molecule_json['molecule']['name']

    # Create the output directory if needed
    output_directory = molecule_json['output']['directory']
    if not do_force and os.path.exists(output_directory):
        print("WARNING: Output directory already exists.")
        answer = input("Proceed? [y/N] ")
        if answer != 'y':
            exit(1)
    else:
        os.makedirs(output_directory, exist_ok=True)

    # Create a specfile for each locality and run the simulations
    # in parallel

    specfiles = []

    for locality in qubits:
        # Make a subdirectory for this locality
        subdirectory = f'{output_directory}/{molecule_name}_{locality}q'
        os.makedirs(subdirectory, exist_ok=True)

        # Copy the given file to the subdirectory; replace with
        # correct locality and queue to run in parallel
        with open(f'{subdirectory}/{molecule_name}_{locality}q.json', 'w') as specfile:
            local_json = deepcopy(molecule_json)
            local_json['circuit']['qubits'] = locality
            local_json['molecule']['name'] = molecule_json['molecule']['name'] + \
                f'{locality}q'
            json.dump(local_json, specfile)
            specfiles.append(os.path.realpath(specfile.name))

    flags = list(filter(lambda arg: arg.startswith('-'), arguments))

    def run_specfile(file):
        simulation = subprocess.Popen(
            [f'{root_path}/lowerbound.py', *flags, file],
            cwd=os.path.dirname(file))
        return simulation.communicate()

    def on_simulation_complete(*args):
        pass

    if '--parallel' not in arguments:
        for specfile in specfiles:
            stdout, stderr = run_specfile(specfile)
            if not '--no-fail' in arguments and stderr:
                print("FAILED")
                raise Exception(stderr)
            on_simulation_complete(specfile)
            gc.collect()
    else:
        pool = multiprocessing.Pool(None)
        r = pool.map_async(run_specfile, specfiles,
                           callback=on_simulation_complete)
        r.wait()
