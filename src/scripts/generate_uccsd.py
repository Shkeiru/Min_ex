# ------------------------------------------------------------------------------
#     AUTHORING
# ------------------------------------------------------------------------------
"""
@file generate_uccsd.py
@author Rayan MALEK
@date 2026-02-19
@brief Script to generate UCCSD ansatz excitations using OpenFermion.
"""

# ------------------------------------------------------------------------------
#     IMPORTS
# ------------------------------------------------------------------------------

import argparse
import json
import sys
import itertools

# Try imports
try:
                from openfermion.ops import FermionOperator
                from openfermion.utils import hermitian_conjugated
                from openfermion.transforms import jordan_wigner, bravyi_kitaev
except ImportError as e:
                print(json.dumps({"error": f"Import failed: {e}. Please install openfermion."}))
                sys.exit(1)


# ------------------------------------------------------------------------------
#     HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def format_pauli_term(term):
                """
                Converts OpenFermion term tuple to string.

                Args:
                                term (tuple): ((0, 'X'), (1, 'Z'))

                Returns:
                                str: "X0 Z1"
                """
                if not term:
                                return "I"

                parts = []
                # OpenFermion terms are sorted by index
                for index, action in term:
                                parts.append(f"{action}{index}")
                return " ".join(parts)


# ------------------------------------------------------------------------------
#     ANSATZ GENERATION
# ------------------------------------------------------------------------------

def generate_singlet_uccsd(n_spatial_orbitals, n_electrons):
                """
                Generates Singlet UCCSD operators.

                Args:
                                n_spatial_orbitals (int): number of spatial orbitals (n_qubits / 2)
                                n_electrons (int): number of electrons

                Returns:
                                list: List of dictionary excitations [
                                                {"type": "single", "indices": [i, a], "operator": FermionOperator},
                                                ...
                                ]
                """
                n_occupied = n_electrons // 2
                n_virtual = n_spatial_orbitals - n_occupied

                occupied_indices = range(n_occupied)
                virtual_indices = range(n_occupied, n_spatial_orbitals)

                excitations = []

                # --------------------------------------------------------------------------
                # Singlet Single Excitations
                # --------------------------------------------------------------------------
                # G_ia = E_ai - E_ia = (a^dag_a alpha a_i alpha + a^dag_a beta a_i beta) - h.c.
                for i in occupied_indices:
                                for a in virtual_indices:
            
                                                i_alpha = 2 * i
                                                i_beta = 2 * i + 1
                                                a_alpha = 2 * a
                                                a_beta = 2 * a + 1

                                                # E_ai = a^dag_a_alpha a_i_alpha + a^dag_a_beta a_i_beta
                                                # Operator = E_ai - E_ia (anti-hermitian generator)

                                                op = FermionOperator(f"{a_alpha}^ {i_alpha}") + FermionOperator(
                                                                f"{a_beta}^ {i_beta}"
                                                )
                                                op -= FermionOperator(f"{i_alpha}^ {a_alpha}") + FermionOperator(
                                                                f"{i_beta}^ {a_beta}"
                                                )

                                                excitations.append({"type": "single", "indices": [i, a], "operator": op})

                # --------------------------------------------------------------------------
                # Singlet Double Excitations
                # --------------------------------------------------------------------------
    
                for i, j in itertools.combinations_with_replacement(occupied_indices, 2):
                                for a, b in itertools.combinations_with_replacement(virtual_indices, 2):
            
                                                i_a = 2 * i
                                                i_b = 2 * i + 1
                                                j_a = 2 * j
                                                j_b = 2 * j + 1
                                                a_a = 2 * a
                                                a_b = 2 * a + 1
                                                b_a = 2 * b
                                                b_b = 2 * b + 1

                                                E_ai = FermionOperator(f"{a_a}^ {i_a}") + FermionOperator(f"{a_b}^ {i_b}")
                                                E_bj = FermionOperator(f"{b_a}^ {j_a}") + FermionOperator(f"{b_b}^ {j_b}")

                                                # G_ijab = E_ai * E_bj - h.c.

                                                op = E_ai * E_bj
                                                op -= hermitian_conjugated(E_ai * E_bj)

                                                excitations.append(
                                                                {"type": "double", "indices": [i, j, a, b], "operator": op}
                                                )

                return excitations


# ------------------------------------------------------------------------------
#     MAIN EXECUTION
# ------------------------------------------------------------------------------

def main():
                """
                Main entry point for UCCSD generation.
                """
                parser = argparse.ArgumentParser(
                                description="Generate UCCSD Ansatz using OpenFermion"
                )
                parser.add_argument("--n_qubits", type=int, required=True, help="Number of qubits")
                parser.add_argument(
                                "--n_electrons", type=int, required=True, help="Number of electrons"
                )
                parser.add_argument(
                                "--mapping",
                                type=str,
                                default="jordan_wigner",
                                choices=["jordan_wigner", "bravyi_kitaev"],
                                help="Qubit mapping",
                )
                parser.add_argument(
                                "--output", type=str, default="uccsd.json", help="Output JSON file"
                )

                args = parser.parse_args()

                # Validate
                if args.n_qubits % 2 != 0:
                                print(json.dumps({"error": "n_qubits must be even."}))
                                sys.exit(1)

                n_spatial_orbitals = args.n_qubits // 2
                n_occupied = args.n_electrons // 2

                if n_occupied > n_spatial_orbitals:
                                print(json.dumps({"error": "More electrons than orbitals."}))
                                sys.exit(1)

                try:
                                # Generate Excitations
                                print(
                                                f"Generating Singlet UCCSD for {args.n_electrons}e- in {n_spatial_orbitals} orbitals...",
                                                flush=True,
                                )
                                excitations = generate_singlet_uccsd(n_spatial_orbitals, args.n_electrons)

                                # Output Structure
                                json_output = {
                                                "n_qubits": args.n_qubits,
                                                "n_electrons": args.n_electrons,
                                                "mapping": args.mapping,
                                                "excitations": [],
                                }

                                # Map to Qubits
                                print(f"Mapping to Qubits using {args.mapping}...", flush=True)
                                for exc in excitations:
                                                fermion_op = exc["operator"]

                                                if args.mapping == "bravyi_kitaev":
                                                                qubit_op = bravyi_kitaev(fermion_op)
                                                else:
                                                                qubit_op = jordan_wigner(fermion_op)

                                                # Convert to JSON format
                                                pauli_terms = []
                                                for term, coeff in qubit_op.terms.items():
                                                                pauli_str = format_pauli_term(term)

                                                                # Store coefficient string
                                                                c_val = coeff
                
                                                                pauli_terms.append(
                                                                                {"pauli": pauli_str, "coeff": str(c_val)}
                                                                )

                                                # Only add if not empty
                                                if pauli_terms:
                                                                json_output["excitations"].append(
                                                                                {
                                                                                                "type": exc["type"],
                                                                                                "indices": exc["indices"],
                                                                                                "pauli_terms": pauli_terms,
                                                                                }
                                                                )

                                # Write File
                                with open(args.output, "w") as f:
                                                json.dump(json_output, f, indent=2)

                                print(
                                                json.dumps(
                                                                {
                                                                                "status": "success",
                                                                                "file": args.output,
                                                                                "n_excitations": len(json_output["excitations"]),
                                                                }
                                                )
                                )

                except Exception as e:
                                print(json.dumps({"error": str(e)}))
                                sys.exit(1)


if __name__ == "__main__":
                main()
