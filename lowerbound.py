#!/usr/bin/env python3

import itertools
import json
import os
import random
import re
from bisect import bisect_left
from collections import defaultdict
from functools import reduce
from glob import glob
from itertools import chain

import h5py
import numpy as np
import pathos.multiprocessing as mp
import qop
import scipy
import scipy.optimize
from more_itertools import partition
from numpy import abs, ceil, sqrt
from openfermion.hamiltonians import MolecularData
from openfermion.ops import QubitOperator
from openfermion.transforms import (get_fermion_operator, get_sparse_operator,
                                    jordan_wigner)
from openfermion.utils import get_ground_state
from openfermionpsi4 import run_psi4
from tqdm import tqdm

import cextension

STRETCH = 1

__doc__ = """\
Calculate a lower bound energy for a molecule.

Usage:
  lowerbound.py <parameters file>
              [-f | --force]
              [-h | --help]
              [-so | --skip-orbital]
              [-sjw | --skip-jordan-wigner]
              [-ssw | --skip-schrieffer-wolff]
              [-sp | --skip-partitioning]
              [-slb | --skip-lower-bound]
              [--debug-decomposition]
  molecule.py --template <file>

Options:
  -h  --help                    Show this screen.
  -f  --force                   Do not ask about overriding.
  -so --skip-orbital            Skip the orbital calculations.
  -sjw --skip-jordan-wigner     Skip the Jordan-Wigner decomposition
                                of the hamiltonians.
  -ssw --skip-schrieffer-wolff  Skip the Schrieffer-Wolff decomposition
                                of the previously created J.W.
                                hamiltonians.
  -sp --skip-partitioning       Skip the calculation of the heuristical
                                partitioning of systems for the S.W.
                                hamiltonians.
  -slb --skip-lower-bound       Skip the calculation of the lower bound
                                associated with the S.W. hamiltonians under
                                the defined partitions (.partititon file).
      --template                Create a template json file, rather
                                than processing an existing one.
      --debug-decomposition     Diagonalize both the full Jordan-
                                -Wigner hamiltonian, and the the
                                k-local hamiltonian produced with the
                                Schrieffer-Wolff transform.
                                WARNING: For any moderate size molecule,
                                this will likely crash the computer in
                                trying to diagonalize the exact
                                hamiltonian!
"""


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# Define parsing a line of a hamiltonian file


term_re = re.compile(
    r"\(?(.+?)\)? *\[([XYZ\d ]*)\](?: \+)?")
operator_re = re.compile(r"([XYZ])(\d+)")


def parse_hamiltonian_line(line):
    try:
        coef, operators_str = term_re.match(line).groups()
    except Exception as e:
        print(line)
        raise e

    # A hamiltonian term is identified as
    # (coef, (<systems envolved>), (<respective operators>))
    systems = []
    operators = []

    for xn in operators_str.strip().split():
        try:
            op, sys = operator_re.match(xn.strip()).groups()
        except Exception as e:
            print(xn)
            raise e
        systems.append(int(sys))
        operators.append(op.lower())

    return [float(coef), operators, systems]

# Produce a k-local hamiltonian via Schrieffer-Wolff transforms


epsilon = 0.1


def schrieffer_wolff(hamiltonian_file, max_locality):
    if max_locality == 2:
        raise Exception("Max locality of 2 not implemented!")

    available_ancilla = None

    # TODO: Go over this code very carefully. I'm not entirely sure
    # it's correct.
    def determine_square(terms):
        squared = [[sum(coef**2 for coef, _, _ in terms), [], []]]

        # Find the cross terms contributions
        # (`itertools.combinations` is in lexical order)
        # `left` and `right` correspond to the `i` and `j` terms in the sum
        # for left, right in itertools.combinations(terms, r=2):
        for i, left in enumerate(terms):
            for right in terms[:i]:
                # If the terms differ in an even number of
                # operators, 2k, there is a term ~(-1)^k
                # We need to build the "common" set of operators/systems
                systems = []
                left_ops = []
                right_ops = []

                left_ptr = 0
                right_ptr = 0
                while left_ptr < len(left[1]) or right_ptr < len(right[1]):
                    if left_ptr < len(left[2]):
                        # Within bounds to left
                        if right_ptr < len(right[2]):
                            # Within bounds to right
                            if left[2][left_ptr] < right[2][right_ptr]:
                                systems.append(left[2][left_ptr])
                                left_ops.append(left[1][left_ptr])
                                right_ops.append('i')
                                left_ptr += 1
                            elif right[2][right_ptr] < left[2][left_ptr]:
                                systems.append(right[2][right_ptr])
                                left_ops.append('i')
                                right_ops.append(right[1][right_ptr])
                                right_ptr += 1
                            else:
                                systems.append(left[2][left_ptr])
                                left_ops.append(left[1][left_ptr])
                                right_ops.append(right[1][right_ptr])
                                left_ptr += 1
                                right_ptr += 1
                        else:
                            # Outside bounds to right;
                            # add the rest of left
                            systems.extend(left[2][left_ptr:])
                            right_ops.extend(
                                ('i',)*(len(left[2]) - left_ptr))
                            left_ops.extend(left[1][left_ptr:])
                            break
                    else:
                        # Outside bounds to left;
                        # add the rest of right
                        systems.extend(right[2][right_ptr:])
                        left_ops.extend(('i',)*(len(right[2]) - right_ptr))
                        right_ops.extend(right[1][right_ptr:])
                        break

                if len(left_ops) == 0:  # This is a scalar term
                    assert(len(right_ops) == 0)
                    squared.append([2*left[0]*right[0], [], []])
                    continue

                # If the two terms differ in an uneven number of operators,
                # they will not contribute to the cross-term of the square
                # Note that identities do not "count" in terms of flipping
                # the sign, so there must be an even number of differences
                # *not* with the identity for the term not to disappear
                ops_arr = np.array([left_ops, right_ops])
                systems = np.array(systems)
                not_in_common = ((ops_arr[0, :] != 'i') &
                                 (ops_arr[1, :] != 'i') &
                                 (ops_arr[0, :] != ops_arr[1, :])).sum()
                if not_in_common % 2 != 0:
                    continue

                # Build the contribution
                # TODO: I am doing this naive-ly, because I couldn't find
                # any clever way to do it.
                different = (ops_arr[0, :] != ops_arr[1, :])
                contrib_sign = 1 if ((not_in_common//2) %
                                     2 == 0) else -1  # from i^(2k)
                contrib_coef = contrib_sign*2*left[0]*right[0]
                contrib_ops = []
                contrib_sys = []
                for op_a, op_b, sys in zip(ops_arr[0, different],
                                           ops_arr[1, different],
                                           systems[different]):
                    contrib_sys.append(sys)
                    if op_a == 'i':
                        contrib_ops.append(op_b)
                    elif op_b == 'i':
                        contrib_ops.append(op_a)
                    elif op_a == 'x':
                        if op_b == 'y':
                            contrib_ops.append('z')
                        else:
                            assert(op_b == 'z')
                            contrib_sign *= -1
                            contrib_ops.append('y')
                    elif op_a == 'y':
                        if op_b == 'z':
                            contrib_ops.append('x')
                        else:
                            assert(op_b == 'x')
                            contrib_sign *= -1
                            contrib_ops.append('z')
                    elif op_a == 'z':
                        if op_b == 'x':
                            contrib_ops.append('y')
                        else:
                            assert(op_b == 'y')
                            contrib_sign *= -1
                            contrib_ops.append('x')
                squared.append([contrib_coef, contrib_ops, contrib_sys])

        # Remove redundant terms!
        squared.sort(key=lambda term: term[1:])
        squared = list(
            map(
                lambda kg: [sum(x[0] for x in kg[1]), *kg[0]],
                itertools.groupby(
                    squared,
                    key=lambda coef_ops_sys: coef_ops_sys[1:])))
        return squared

    def decimate(terms):
        nonlocal max_locality, available_ancilla
        # `terms` is a collection of items with structure
        # (<coef>, [<operator>], [<system index>])
        # representing a hamiltonian that is the sum of all
        # these items.
        # These terms are **not reduced**

        # We can ignore all terms that are below the intended locality
        not_involved, involved = partition(
            lambda term: len(term[2]) > max_locality, terms)
        not_involved = [list(x) for x in not_involved]
        involved = list(involved)

        # Find out what systems are involved in the terms to be reduced
        systems_to_reduce = np.unique(
            list(sum((term[2] for term in involved), [])))

        try:
            involved_coefs, involved_ops, involved_sys = zip(*involved)
        except ValueError:
            # If not enough values to unpack, that means involved was empty
            return not_involved

        sw_hamiltonian = not_involved
        involved_coefs = np.fromiter(involved_coefs, dtype=float)
        involved_opssys = [(tuple(x), tuple(y))
                           for x, y in zip(involved_ops, involved_sys)]

        # Filter out very small contributions
        mask = (np.abs(involved_coefs)/np.max(involved_coefs) > 1e-6)
        involved_coefs = involved_coefs[mask]
        involved_opssys = [involved_opssys[sv_index]
                           for sv_index in range(len(involved_opssys)) if mask[sv_index]]

        # Perform the actual reduction
        system_cutoff = len(systems_to_reduce)//2
        low_space = systems_to_reduce[:system_cutoff]
        high_space = systems_to_reduce[system_cutoff:]
        operators = ('x', 'y', 'z', 'i')

        # Build the matrix C
        row = []
        col = []
        values = []
        for low_index, low_operators in enumerate(itertools.product(operators, repeat=len(low_space))):
            for high_index, high_operators in enumerate(itertools.product(operators, repeat=len(high_space))):
                # We can skip the full identity, as we know this will
                # never be in the list of terms to reduce!
                full_operators = low_operators + high_operators
                if len(full_operators) * ('i',) == full_operators:
                    continue

                # Build the (<operators>, <systems>) pair to look for in `involved`
                ops_sys = tuple(zip(*filter(lambda x: x[0] != 'i', zip(
                    full_operators, systems_to_reduce))))

                try:
                    term_index = involved_opssys.index(ops_sys)
                except ValueError:
                    continue

                row.append(low_index)
                col.append(high_index)
                values.append(involved_coefs[term_index])

        if sum(value**2 for value in values) < 1e-5:
            return sw_hamiltonian

        # We will need an ancilla for each singular value
        c = scipy.sparse.coo_matrix((values, (row, col)), shape=(
            len(operators)**len(low_space), len(operators)**len(high_space))).tocsr()
        u, s, vt = scipy.sparse.linalg.svds(c, return_singular_vectors=True, ncv=25)

        # The operator can be written as Sum(Tensor(u[i], s[i]*vt[i].T) for i)
        for sv_index in range(s.shape[0]):
            if s[sv_index] == 0:
                continue

            # ui and vi contain the coefficients for the low and high spaces
            uk = u[:, sv_index]
            vk = vt[sv_index, :] * s[sv_index]

            # Convert uk, vk into the operator form
            # See that if our "components" are ('x', 'y', 'z', 'i')
            # And we're producing e.g. 3-sized combinatitions in order
            # 0 --> xxx
            # 1 --> xxy
            # 2 --> xxz
            # 3 --> xxi
            # 4 --> xyx
            # etc.
            # Then the it works essentially as a `components`-sized decimal system
            # and the "digit" at position `j` (right-to-left) of number `i` is
            # given by `components[(i // len(components)**j) % len(components)]`
            coupling = np.max((np.max(np.abs(uk)), np.max(np.abs(vk))))
            low_component = [
                [uki, [operators[(low_index // len(operators)**(pos - 1)) % len(operators)]
                       for pos in range(len(low_space), 0, -1)], list(low_space)]
                for low_index, uki in enumerate(uk)
                if uki != 0
            ]
            high_component = [
                [vki, [operators[(high_index // len(operators)**(pos - 1)) % len(operators)]
                       for pos in range(len(high_space), 0, -1)], list(high_space)]
                for high_index, vki in enumerate(vk)
                if vki != 0
            ]

            # Filter both `low_component` and `high_component` from identities
            low_component = [
                (
                    [elem[0], *(list(x)
                                for x in zip(*filter(
                                    lambda x: x[0] != 'i',
                                    zip(elem[1], elem[2])
                                ))
                                )]
                    if elem[1] != ['i']*len(elem[1]) else
                    [elem[0], [], []]
                )
                for elem in low_component
            ]
            high_component = [
                (
                    [elem[0], *(list(x)
                                for x in zip(*filter(
                                    lambda x: x[0] != 'i',
                                    zip(elem[1], elem[2])
                                ))
                                )]
                    if elem[1] != ['i']*len(elem[1]) else
                    [elem[0], [], []]
                )
                for elem in high_component
            ]

            # The ancilla for this term
            used_ancilla = available_ancilla
            available_ancilla += 1

            local_terms = [
                [coupling / epsilon**2, ['|1><1|'], [used_ancilla]],
                *(
                    [-sqrt(coupling/2)/epsilon*uki,
                     [*ops, 'x'], [*space, used_ancilla]]
                    for uki, ops, space in low_component
                ),
                *(
                    [sqrt(coupling/2)/epsilon*vki,
                     [*ops, 'x'], [*space, used_ancilla]]
                    for vki, ops, space in high_component
                ),
                *(
                    [coef/2, ops, space]
                    for coef, ops, space in determine_square(low_component)
                ),
                *(
                    [coef/2, ops, space]
                    for coef, ops, space in determine_square(high_component)
                )
            ]

            local_terms = [elem for elem in local_terms if abs(elem[0]) > 1e-4]
            local_terms.sort(key=lambda term: term[1:])
            local_terms = list(
                map(
                    lambda kg: [sum(x[0] for x in kg[1]), *kg[0]],
                    itertools.groupby(
                        local_terms,
                        key=lambda coef_ops_sys: coef_ops_sys[1:])))
            local_terms = decimate(local_terms)
            sw_hamiltonian.extend(local_terms)

        sw_hamiltonian.sort(key=lambda term: term[1:])
        sw_hamiltonian = list(
            map(
                lambda kg: [sum(x[0] for x in kg[1]), *kg[0]],
                itertools.groupby(
                    sw_hamiltonian,
                    key=lambda coef_ops_sys: coef_ops_sys[1:])))

        return sw_hamiltonian

    # Read the hamiltonian file;
    # Get from the file:
    #  - Number of interacting systems (to determine ancilla index)
    #  - Hamiltonian in [(coef, operators, systems)] form
    # We immediately decimate each of the terms and return all the
    # terms
    hamiltonian_terms = []

    with open(hamiltonian_file) as hamiltonian_fileio:
        for line in hamiltonian_fileio:
            if line.startswith('#!Ancilla Qubits Start At'):
                available_ancilla = int(
                    line[len('#!Ancilla Qubits Start At'):])
            if line.startswith('#'):
                continue
            term = parse_hamiltonian_line(line)
            hamiltonian_terms.append(term)

    # Returns: Hamiltonian terms, number of systems involved
    return decimate(hamiltonian_terms), available_ancilla


def schrieffer_wolff_legacy(hamiltonian_file, max_locality):
    # The next available ancilla qubit index
    # This variable needs to be set once hamiltonian_file is read
    available_ancilla = None

    # Produce a ceil(k/2), where k is the locality, of a given hamiltonian
    # This is a recursive function; will return a list of new hamiltonians!
    # Hamiltonian terms are described by a common coefficient, a list of
    # operators (in string representation, e.g., 'x', 'y', '|1><1|') and
    # the corresponding systems (by their index).
    # The function returns a list of new terms, in this same representation.
    def decimate(coef, operators, systems):
        nonlocal max_locality
        nonlocal available_ancilla

        # Projection operators are unchanged when squared;
        # Pauli operators, on the other hand, become the identity.
        # This will break if other operators are involved!
        # def square_hamiltonian(operators, systems):
        #    operators_systems = filter(lambda op_sys: op_sys[0] not in (
        #        'x', 'y', 'z'), zip(operators, systems))
        #    operators_systems = list(zip(*operators_systems))
        #    if len(operators_systems) == 0:
        #        return (), ()
        #    else:
        #        operators, systems = operators_systems
        #        return operators, systems

        # Determine locality of given hamiltonian
        k = len(systems)

        # Already under max_locality? Return hamiltonian as is
        if k <= max_locality:
            return [(coef, tuple(operators), tuple(systems))]

        # Because the work of Bravyi et al. seems to assume that
        # the coupling is, by definition, positive, I here redefine
        # some operator K.ABC with K<0 as
        # J.A'BC with J=abs(K) and A'=sign(K).A
        # This implies the sign will vary with the squaring of A'
        sign = -1 if coef < 0 else 1
        coef = abs(coef)

        # Term is not k-local; we're going to need an ancilla
        ancilla_qubit = available_ancilla
        available_ancilla += 1

        # Choose type of decimation (3-to-2 or k-to-k/2)
        if k == 3:
            # 3-to-2 decimation
            a_operator, b_operator, c_operator = operators
            a_system, b_system, c_system = systems

            delta = coef/epsilon**3
            new_terms = [
                (delta, ('|1><1|',), (ancilla_qubit,)),  # H_0
                (-delta**(2/3)*coef**(1/3), (c_operator, '|1><1|'),
                    (c_system, ancilla_qubit)),  # V_d
                (-sign*delta**(2/3)*coef**(1/3)/np.sqrt(2), (a_operator, 'x'),
                    (a_system, ancilla_qubit)),  # V_od; term associated with -A
                (delta**(2/3)*coef**(1/3)/np.sqrt(2), (b_operator, 'x'),
                    (b_system, ancilla_qubit)),  # V_od; term associated with B
                # V_extra terms
                (-sign*delta**(1/3)*coef**(2/3),
                    (a_operator, b_operator), (a_system, b_system)),
                (coef, (c_operator,), (c_system,)),
                (delta**(1/3)*coef**(2/3), (), ())
            ]
        else:
            # Decimate from k to ceil(k/2)+1
            # Produce individual terms first
            # TODO: group them?
            # The partition is done by halving the hamiltonian
            division_index = int(ceil(len(systems)/2))

            low_hamiltonian = (
                operators[:division_index], systems[:division_index])
            high_hamiltonian = (
                operators[division_index:], systems[division_index:])

            delta = coef/epsilon**2
            new_terms = [
                # Projector term
                (delta, ('|1><1|',), (ancilla_qubit,)),
                # V; term associated to -A
                (-sign*sqrt(delta*coef/2),
                    (*low_hamiltonian[0], 'x'), (*low_hamiltonian[1], ancilla_qubit)),
                # V; term associated to B
                (sqrt(delta*coef/2),
                    (*high_hamiltonian[0], 'x'), (*high_hamiltonian[1], ancilla_qubit)),
                # V_extra; A² and B² are Pauli operators!
                # (Although projectors are introduced into the hamiltonian, they
                #  are never coupled to other bodies; the ancilla qubit is coupled
                #  to the existing bodies via an X Pauli operator)
                (coef, (), ())
            ]

        # Transform each new term into a decimated one; decimation will occur
        # recursively, so we can immediately return the array
        result = []
        for term in new_terms:
            result.extend(decimate(*term))
        return tuple(result)

    # Read the hamiltonian file;
    # Get from the file:
    #  - Number of interacting systems (to determine ancilla index)
    #  - Hamiltonian in [(coef, operators, systems)] form
    # We immediately decimate each of the terms and return all the
    # terms
    hamiltonian_terms = []

    with open(hamiltonian_file) as hamiltonian_fileio:
        for line in hamiltonian_fileio:
            if line.startswith('#!Ancilla Qubits Start At'):
                available_ancilla = int(
                    line[len('#!Ancilla Qubits Start At'):])
            if line.startswith('#'):
                continue
            hamiltonian_terms.extend(
                decimate(*parse_hamiltonian_line(line)))

    # Returns: Hamiltonian terms, number of systems involved
    return hamiltonian_terms, available_ancilla


def matrix_from_terms(hamiltonian, num_systems):
    total_hamiltonian = scipy.sparse.coo_matrix(
        (2**num_systems, 2**num_systems), dtype=complex)
    for term in hamiltonian:
        coef, operators, systems = term

        if len(systems) == 0:
            total_hamiltonian += coef * \
                scipy.sparse.eye(2**num_systems, dtype=complex)
            continue

        # Create a representation of the decomposed hamiltonian
        operator_matrix = scipy.sparse.eye(2**min(systems), dtype=complex)
        actual_op_index = 0
        for i in range(min(systems), max(systems)+1):
            if actual_op_index < len(systems) and systems[actual_op_index] == i:
                operator = operators[actual_op_index]
                if operator == 'x':
                    operator = np.matrix(((0, 1), (1, 0)), dtype=complex)
                elif operator == 'y':
                    operator = np.matrix(((0, -1j), (1j, 0)), dtype=complex)
                elif operator == 'z':
                    operator = np.matrix(((1, 0), (0, -1)), dtype=complex)
                elif operator == '|0><0|':
                    operator = np.matrix(((1, 0), (0, 0)), dtype=complex)
                elif operator == '|1><1|':
                    operator = np.matrix(((0, 0), (0, 1)), dtype=complex)
                else:
                    raise Exception(f"Unknown operator {operator}")
                operator_matrix = scipy.sparse.kron(
                    operator_matrix, scipy.sparse.coo_matrix(operator))
                actual_op_index += 1
            else:
                operator_matrix = scipy.sparse.kron(
                    operator_matrix, scipy.sparse.eye(2, dtype=complex))
        operator_matrix = scipy.sparse.kron(operator_matrix,
                                            scipy.sparse.eye(2**(num_systems - max(systems) - 1)))
        assert(operator_matrix.shape == (2**num_systems, 2**num_systems))
        total_hamiltonian += coef * operator_matrix
    return total_hamiltonian


def ground_state_from_terms(hamiltonian, num_systems):
    #return np.min(scipy.sparse.linalg.eigsh(matrix_from_terms(hamiltonian, num_systems),
    #                                        k=1, which='SA', return_eigenvectors=False))
    return np.min(np.linalg.eigvalsh(matrix_from_terms(hamiltonian, num_systems).todense()))


if __name__ == '__main__':
    arguments = os.sys.argv
    if len(arguments) == 1 or '-h' in arguments or '--help' in arguments:
        print(__doc__)
        exit(0)
    arguments = arguments[1:]

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
        with open(target_file, "w") as target_file:
            target_file.write("""\
{
    "output": {
        "directory": "",
        "file": ""
    },
    "molecule": {
        "name": "",
        "atoms": [
            "",
            ""
        ],
        "basis": [
            ""
        ],
        "multiplicity": ,
        "bond": {
            "npoints": ,
            "start": ,
            "end":
        }
    },
    "simulation": {
        "active_space": {
            "start": ,
            "stop":
        },
        "epsilon": ,
        "sfc": ,
        "mp2": ,
        "cisd": ,
        "ccsd": ,
        "fci":
    },
    "circuit": {
        "qubits": ,
        "layers":
    }
}
""")
        exit(0)

    do_force = '-f' in arguments or '--force' in arguments

    # Get the molecule file
    try:
        molecule_json = next(
            filter(lambda x: not x.startswith('-'), arguments))
        root_path = os.path.dirname(os.path.realpath(molecule_json))
    except StopIteration:
        print("Need a molecule file.")
        exit(1)

    if not os.path.exists(molecule_json):
        print(f"Could not find {molecule_json}")
        exit(1)

    with open(molecule_json, "r") as file:
        molecule_json = json.load(file)

    # Set files parameters
    output_directory = f"{root_path}/{molecule_json['output']['directory']}"
    output_file = molecule_json['output']['file']

    # Set molecule parameters.
    molecule_name = molecule_json['molecule']['name']

    def molecule_geometry(bond_length): return [
        (molecule_json['molecule']['atoms'][0], (0., 0., 0.)),
        (molecule_json['molecule']['atoms'][1], (0., 0., bond_length))]

    basis_set = tuple(molecule_json['molecule']['basis'])
    multiplicity = molecule_json['molecule']['multiplicity']
    n_points = molecule_json['molecule']['bond']['npoints']
    start_bond_length = molecule_json['molecule']['bond']['start']
    end_bond_length = molecule_json['molecule']['bond']['end']
    active_space_start = molecule_json['simulation']['active_space'].get(
        'start', None)
    active_space_stop = molecule_json['simulation']['active_space'].get(
        'stop', None)
    epsilon = molecule_json['simulation']['epsilon']

    # Set calculation parameters.
    run_scf = molecule_json['simulation']['sfc']
    run_mp2 = molecule_json['simulation']['mp2']
    run_cisd = molecule_json['simulation']['cisd']
    run_ccsd = molecule_json['simulation']['ccsd']
    run_fci = molecule_json['simulation']['fci']

    # Set quantum circuit parameters
    qubits = molecule_json['circuit']['qubits']
    layers = molecule_json['circuit']['layers']

    def get_filename(molecule_name, basis, *rest):
        return f"{output_directory}/{molecule_name}_{basis}_{rest}.molecule"

    # Create the output directory if needed
    if not do_force and os.path.exists(output_directory):
        print("WARNING: Output directory already exists.")
        answer = input("Proceed? [y/N] ")
        if answer != 'y':
            exit(1)
    else:
        os.makedirs(output_directory, exist_ok=True)

    # Create PSI4 molecule files
    skip = ('-so' in arguments or '--skip-orbital' in arguments)
    if not skip and not do_force and len(glob(f"{output_directory}/*.molecule.*")) != 0:
        print("WARNING: Directory already contains .molecule files.")
        print(glob(f"{output_directory}/*.molecule.*"))
        print(" These might be overridden.")
        answer = input("Override? [y/N] ")
        if answer != 'y':
            skip = True

    if not skip:
        for basis in basis_set:
            for bond_length in np.linspace(start_bond_length, end_bond_length, n_points):
                molecule = MolecularData(
                    molecule_geometry(bond_length), basis, multiplicity,
                    description=str(round(bond_length, 2)),
                    filename=get_filename(molecule_name, basis, round(bond_length, 3)))

                # Run Psi4.
                molecule = run_psi4(molecule,
                                    run_scf=run_scf,
                                    run_mp2=run_mp2,
                                    run_cisd=run_cisd,
                                    run_ccsd=run_ccsd,
                                    run_fci=run_fci)
                molecule.save()

                print(f"Saved {molecule.filename}")

    # Create the jordan-wigner hamiltonian
    # Load the molecules, and compute the Jordan-Wigner form of the fermionic
    # approximation to the hamiltonian

    skip = ('-sjw' in arguments or '--skip-jordan-wigner' in arguments)
    if not skip and not do_force and len(glob(f"{output_directory}/*.hamiltonian")) != 0:
        print("WARNING: Directory already contains .hamiltonian files.")
        print(" These might be overridden.")
        answer = input("Override? [y/N] ")
        if answer != 'y':
            skip = True

    if not skip:
        print("Calculting hamiltonians...")
        for basis in basis_set:
            for bond_length in np.linspace(start_bond_length, end_bond_length, n_points):
                geometry = molecule_geometry(bond_length)
                molecule = MolecularData(
                    geometry, basis, multiplicity,
                    description=str(round(bond_length, 2)),
                    filename=get_filename(molecule_name, basis, round(bond_length, 3)))
                molecule.load()

                # A list of spatial orbital indices indicating which
                # orbitals should be considered active.
                molecular_hamiltonian = molecule.get_molecular_hamiltonian(
                    occupied_indices=range(
                        active_space_start) if active_space_start is not None else None,
                    active_indices=range(active_space_start, active_space_stop) if active_space_stop is not None else None)

                # Map operator to fermions and qubits.
                fermion_hamiltonian = get_fermion_operator(
                    molecular_hamiltonian)
                qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
                qubit_hamiltonian.compress()

                qubit_hamiltonian *= STRETCH

                # Find out what the largest coupling is,
                # And the number of qubits involved
                largest_coupling = 0.
                max_q_index = 0
                for line in str(qubit_hamiltonian).splitlines():
                    coef, operators, systems = parse_hamiltonian_line(line)
                    if abs(coef) > largest_coupling:
                        largest_coupling = abs(coef)
                    local_max_index = max(systems) if len(systems) > 0 else -1
                    if local_max_index > max_q_index:
                        max_q_index = local_max_index + 1

                print(f"Finished computing JW for {molecule.filename}")

                # Write result to a file
                with open(molecule.filename + '.hamiltonian', 'w') as outfile:
                    outfile.write(
                        "# geometry basis multiplicity description filename\n")
                    outfile.write(
                        f"# {geometry} {basis} {multiplicity} {molecule.description} {molecule.filename}\n")
                    outfile.write(f"#!Largest Coupling {largest_coupling}\n")
                    outfile.write(f"#!Ancilla Qubits Start At {max_q_index}\n")
                    if run_scf:
                        outfile.write(
                            f"#!HF-energy {molecule.hf_energy * STRETCH}\n")
                    if run_ccsd:
                        outfile.write(
                            f"#!CCSD-energy {molecule.ccsd_energy * STRETCH}\n")
                    if run_cisd:
                        outfile.write(
                            f"#!CISD-energy {molecule.cisd_energy * STRETCH}\n")
                    if run_mp2:
                        outfile.write(
                            f"#!MP2-energy {molecule.mp2_energy * STRETCH}\n")
                    if run_fci:
                        outfile.write(
                            f"#!FCI-energy {molecule.fci_energy * STRETCH}\n")
                    outfile.write("# Jordan-Wigner hamiltonian follows\n")
                    outfile.write(f"{qubit_hamiltonian}")
                print(f"Finished writing {molecule.filename + '.hamiltonian'}")

    # Optionally exactly diagonalize both the k-local and JW hamiltonians
    # in order to check that their ground states match.
    if '--debug-decomposition' in arguments:
        print("====== DECOMPOSITION DEBUG FOLLOWS ======")
        with open(f"decomposition.debug", 'w') as debugio:
            debugio.write("# length exact decomposed\n")
            for file in tqdm(glob(f"{output_directory}/*.hamiltonian")):
                print(file)
                decomposed, system_num = schrieffer_wolff(file, qubits)
                approx = ground_state_from_terms(decomposed, system_num)
                print("Approx. ground state energy:")
                print(approx)

                with open(file) as fileio:
                    for line in fileio:
                        if line.startswith("#!HF-energy"):
                            exact = float(line[len("#!HF-energy"):])
                            print(line.strip())
                            break

                print(f"Error of {(exact - approx)/exact*100}%\n")
                length = re.search(
                    r"[^_]+_[^_]+_\(([\d\.]+),\).*", file).group(1)
                debugio.write(f"{length} {exact} {approx}\n")

    # Perform a Schrieffer-Wolfff transform on the hamiltonians, and aggregate
    # the terms; calculate a lower-bound.

    skip = ('-ssw' in arguments or '--skip-schrieffer-wolff' in arguments)
    if not skip and not do_force and len(glob(f"{output_directory}/*.sw")) > 0:
        print("WARNING: Directory already contains .sw files.")
        print(" These might be overridden.")
        answer = input("Override? [y/N] ")
        if answer != 'y':
            skip = True

    if not skip:
        print("Calculating Schrieffer-Wolff decomposition...", flush=True)
        for hamiltonian_file in tqdm(glob(f"{output_directory}/*.hamiltonian")):
            sw, system_num = schrieffer_wolff(hamiltonian_file, qubits)
            sw.sort(key=lambda x: x[2])

            with open(f"{hamiltonian_file}.sw", 'w') as outfile:
                outfile.write(f'# {hamiltonian_file}\n')
                outfile.write(f'#! Max Locality {qubits}\n')
                outfile.write(f'#! Number of Systems {system_num}\n')
                outfile.write(
                    '# Schrieffer-Wolff decomposition as JSON follows:\n')
                json.dump([
                    (sum(x[0] for x in g), *k) for k, g in itertools.groupby(sw, key=lambda x: x[1:])
                ], outfile, cls=NpEncoder)

    # Performing aggregation of terms:
    # There is only one possible partitioning of a `n`-qubit system at a max locality
    # of `q` when demanding the use of as least partitions as possible;
    # it is `n choose q`, because partitions smaller than `q` will be contained in
    # `q`-sized partitions. Therefore, the question is what partitions can be suppressed.
    # The technique here used is to assign a score to each partition, representing the
    # number of terms that it can contain.
    # Upon a new term, it is assigned to the partition that already contains most terms,
    # or a point is given to every tied-score partition.
    # By the end of the process, partitions with a 0 score can be discarded.
    # For this to work correctly, the terms must be iterated over in order from most
    # interacting systems to least interacting systems.

    skip = ('-sp' in arguments or '--skip-partitioning' in arguments)
    if not skip and not do_force and len(glob(f"{output_directory}/*.partitioned")) > 0:
        print("WARNING: Directory already contains .partitioned files.")
        print(" These might be overridden.")
        answer = input("Override? [y/N] ")
        if answer != 'y':
            skip = True

    if not skip:
        print("Calculating partitioning...", flush=True)
        for sw_file in tqdm(glob(f"{output_directory}/*.sw")):
            # Read the Schrieffer-Wolff hamiltonian
            with open(sw_file) as sw_file:
                while True:
                    rewind = sw_file.tell()
                    line = sw_file.readline()
                    if line.startswith('#!'):
                        if 'Max Locality' in line:
                            qubits = int(line[len('#! Max Locality'):])
                        elif 'Number of Systems' in line:
                            system_num = int(
                                line[len('#! Number of Systems'):])
                    if line.startswith('#'):
                        continue
                    else:
                        sw_file.seek(rewind)
                        break

                sw = json.loads(sw_file.read())

            # Split terms into partitions; if a term fits in more than one partition,
            # distribute it equally
            # To avoid having a double representation of `sw`, terms are instead
            # represented by (coef, index) where `index` indicates the (operator, system)
            # of the `index` element of `sw`

            """
            # Reference partitions; we'll use indexes in reference to these
            partitions = tuple(itertools.combinations(
                range(system_num), r=qubits))

            # How many terms has each partition absorved
            partition_score = [0 for _ in partitions]
            best_score = 0

            split_terms = [[] for partition in partitions]

            terms_larger_to_smaller = sorted(
                enumerate(sw), key=lambda i_term: len(i_term[1][2]), reverse=True)
            for term_index, term in tqdm(terms_larger_to_smaller):
                coef, operators, systems = term

                # Indexes of partitions that have the best score and fit
                # the current considered term
                best = []
                best_size = None
                for part_index in filter(lambda i: all(binary_search(partitions[i], sys)
                                                       != -1 for sys in systems),
                                         sorted(range(len(partitions)), key=lambda i: partition_score[i], reverse=True)):
                    if best_size is None:
                        best_size = len(partitions[part_index])
                    elif len(partitions[part_index]) < best_size:
                        break
                    best.append(part_index)

                if len(best) == 0:
                    raise Exception(
                        "Could not find a partition to host the term? This should not happen!")

                # Fit the current term into all of the "best" partitions
                # found
                for index in best:
                    partition_score[index] += 1
                    best_score += 1
                    split_terms[index].append((coef/len(best), term_index))

            # Exclude partitioning that is not used
            # Indices of partitions that are "in use":
            used_partitions = list(filter(
                lambda i: partition_score[i] != 0, range(len(partition_score))))

            # Split terms can be "culled" by considering only the terms
            # corresponding to non-empty partitions
            split_terms = list(map(lambda i: split_terms[i], used_partitions))
            """
            # partitions is a list of elements of structure
            #   (<partition>, (<coef, term> pairs))
            # where <partition> is a tuple indicating the terms to consider together,
            # <coef> is the actual coefficient to consider for each hamiltonian term and
            # <term> is the index to the corresponding hamiltonian term
            sw.sort(key=lambda i: len(i[2]), reverse=True)
            partitions = cextension.find_used_partitions(
                sw, system_num, qubits)

            used_partitions = [x[0] for x in partitions]
            split_terms = [x[1] for x in partitions]

            with open(f'{sw_file.name}.partitioned', 'w') as partitioned_out:
                json.dump({
                    'hamiltonian': sw,
                    'partitions': used_partitions,
                    'split': split_terms,
                }, partitioned_out)

    # Calculate the lower bound to the energy associated to the partition files.
    # This is relatively straightforward; consider each of the hamiltonians,
    # find the ground state, add the value of all partitions to obtain the lower bound.
    # Note that all results are written to a single file; however, to avoid loosing
    # intermediate results, each of the values is persistently held in a .hdf5 file.

    skip = ('-slb' in arguments or '--skip-lower-bound' in arguments)
    if not skip and not do_force and os.path.exists(f'{molecule_name}.txt'):
        print(f"WARNING: Results file {molecule_name}.txt already exists.")
        print(" This WILL be overritten.")
        answer = input("Override? [y/N] ")
        if answer != 'y':
            skip = True

    if not skip:
        with h5py.File(f"{output_directory}/{molecule_name}_intermediate.hdf5", 'w') as intermediate:
            partitioned_file_count = len(
                glob(f"{output_directory}/*.partitioned"))
            intermediate.create_dataset(
                "lower_bounds", (partitioned_file_count,), dtype=float)
            intermediate.create_dataset(
                "lower_bounds_nopart", (partitioned_file_count,), dtype=float)
            intermediate.create_dataset(
                "lower_bounds_discard", (partitioned_file_count,), dtype=float)
            intermediate.create_dataset(
                "filename_index", (partitioned_file_count,), dtype=h5py.string_dtype())

            print("Calculating lower bounds...")
            for i, partitioned in tqdm(enumerate(glob(f"{output_directory}/*.partitioned"))):
                intermediate["filename_index"][i] = partitioned

                with open(partitioned) as partitioned:
                    partitioned = json.load(partitioned)

                    hamiltonian = partitioned['hamiltonian']
                    partitions = partitioned['partitions']
                    split = partitioned['split']

                #pool = mp.Pool(2)

                def mp_lowerbound(args):
                    partition, terms = args
                    return ground_state_from_terms(
                        map(lambda x: (x[0], x[1], [partition.index(y) for y in x[2]]),
                            map(lambda x: (x[0], *hamiltonian[x[1]][1:]), terms)), len(partition))

                lower_bound = sum(map(mp_lowerbound, zip(partitions, split)))
                # lower_bound = sum(
                #    pool.map(mp_lowerbound, zip(partitions, split)))

                def mp_lowerbound_nopart(term):
                    if len(term[2]) == 0:
                        return term[0]
                    keyed_term = [term[0], term[1], [
                        i for i in range(len(term[2]))]]
                    matrix = matrix_from_terms(
                        [keyed_term], len(term[2])).todense()
                    return np.min(np.linalg.eigvalsh(matrix))

                lower_bound_nopart = sum(
                    map(mp_lowerbound_nopart, hamiltonian))
                # lower_bound_nopart = sum(
                #    pool.map(mp_lowerbound_nopart, hamiltonian))

                """
                for partition, terms in zip(partitions, split):
                    terms = [(x[0], *hamiltonian[x[1]][1:]) for x in terms]
                    terms = [(x[0], x[1], [partition.index(y)
                              for y in x[2]]) for x in terms]
                    local_ground_state = ground_state_from_terms(
                        terms, len(partition))
                    lower_bound += local_ground_state
                """

                intermediate["lower_bounds"][i] = lower_bound
                intermediate["lower_bounds_nopart"][i] = lower_bound_nopart

            with open(f"{molecule_name}.txt", 'w') as output:
                def get_hartree_fock(filename):
                    root = filename[:filename.rfind(
                        '.hamiltonian') + len('.hamiltonian')]
                    with open(root) as hamfile:
                        for line in hamfile:
                            if line.startswith('#!HF-energy'):
                                return float(line[len('#!HF-energy'):])

                filename_pat = re.compile(
                    r'[a-zA-Z0-9]+_[a-zA-Z0-9\-]+_\(([0-9\.]+),\).+')
                get_length = np.vectorize(lambda s: float(
                    filename_pat.search(s).group(1)))
                lengths = get_length(intermediate['filename_index'][:])
                energies = np.vectorize(get_hartree_fock)(
                    intermediate['filename_index'][:])
                sort_key = np.argsort(lengths)

                output.write(
                    "# <bound length>\t<exact>\t<lower bound>\t<lower bound no part>\n")
                for i in sort_key:
                    output.write(
                        f"{lengths[i]}\t{energies[i]}\t{intermediate['lower_bounds'][i]}\t{intermediate['lower_bounds_nopart'][i]}\n")

    # Determine energy minimum geometry and compare
    data = np.genfromtxt(fname=f"{molecule_name}.txt",
                         delimiter="\t", skip_header=1, filling_values=1)

    exact_geom = data[np.argmin(data[:, 1]), 0]
    lower_bound_geom = data[np.argmin(data[:, 2]), 0]

    print("Geometry from Hartree-Fock: ", exact_geom)
    print("Geometry from lower bound:  ", lower_bound_geom)
    if lower_bound_geom in (start_bond_length, end_bond_length):
        print("WARNING: geometry from lower bound matches edge of geometry range...")
