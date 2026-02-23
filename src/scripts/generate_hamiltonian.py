# ------------------------------------------------------------------------------
#     AUTHORING
# ------------------------------------------------------------------------------
"""
@file generate_hamiltonian.py
@author Rayan MALEK
@date 2026-02-19
@brief Script to generate molecular Hamiltonian using OpenFermion and PySCF.
"""

# ------------------------------------------------------------------------------
#     IMPORTS
# ------------------------------------------------------------------------------

import argparse
import json
import os
import sys

# Try imports
try:
                from openfermion import MolecularData
                from openfermion.transforms import (
                                jordan_wigner,
                                bravyi_kitaev,
                                get_fermion_operator,
                )
                from openfermionpyscf import run_pyscf
                from pyscf import gto, scf, fci
except ImportError as e:
                print(
                                json.dumps(
                                                {
                                                                "error": f"Import failed: {e}. Please install openfermion, openfermionpyscf, pyscf."
                                                }
                                )
                )
                sys.exit(1)


# ------------------------------------------------------------------------------
#     HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def parse_geometry(geo_str):
                """
                Parses a geometry string into a list of atom tuples.

                Args:
                                geo_str (str): String format "H 0 0 0; H 0 0 0.74"

                Returns:
                                list: List of tuples [('H', (0.0, 0.0, 0.0)), ...]
                """
                atoms = []
                try:
                                parts = geo_str.split(";")
                                for part in parts:
                                                tokens = part.strip().split()
                                                if len(tokens) < 4:
                                                                continue
                                                symbol = tokens[0]
                                                coords = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
                                                atoms.append((symbol, coords))
                except Exception as e:
                                print(json.dumps({"error": f"Geometry parsing failed: {e}"}))
                                sys.exit(1)
                return atoms


def format_pauli_term(term):
                """
                Converts OpenFermion term tuple to a string representation.

                Args:
                                term (tuple): OpenFermion term structure e.g. ((0, 'X'), (1, 'Z'))

                Returns:
                                str: String format "X0 Z1"
                """
                if not term:
                                return "I"

                parts = []
                # OpenFermion terms are sorted by index
                for index, action in term:
                                parts.append(f"{action}{index}")
                return " ".join(parts)


# ------------------------------------------------------------------------------
#     BENCHMARKING
# ------------------------------------------------------------------------------

def get_god_values(atom_str, basis="sto-3g", charge=0, spin=0):
                """
                Calculates reference energy values (HF and FCI) for benchmarking.

                Args:
                                atom_str (str): Atom configuration string for PySCF.
                                basis (str): Basis set name.
                                charge (int): Total charge.
                                spin (int): Total spin (2S).
                """
                print(f"--- Benchmarking {atom_str} ---")

                # 1. Define Molecule
                mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin)
                mol.build()

                # 2. Nuclear Repulsion
                e_nuc = mol.energy_nuc()

                # 3. Hartree-Fock (Mean Field)
                mf = scf.RHF(mol)
                e_hf_tot = mf.kernel()

                # 4. Full Configuration Interaction (FCI) - Exact Solution
                cisolver = fci.FCI(mol, mf.mo_coeff)
                e_fci_tot, _ = cisolver.kernel()

                # 5. Electronic Energy Extraction
                e_fci_elec = e_fci_tot - e_nuc

                print(f"Nuclear Repulsion : {e_nuc:.8f} Ha")
                print(f"Total Energy (FCI): {e_fci_tot:.8f} Ha")
                print(f"-> TARGET ELECTRONIC ENERGY: {e_fci_elec:.8f} Ha")
                print("-----------------------------------")


# ------------------------------------------------------------------------------
#     MAIN EXECUTION
# ------------------------------------------------------------------------------

def main():
                """
                Main entry point for the script.
                Parses arguments, runs PySCF, generates Hamiltonian, and outputs JSON.
                """
                parser = argparse.ArgumentParser(
                                description="Generate Hamiltonian using OpenFermion/PySCF"
                )
                parser.add_argument(
                                "--atom",
                                type=str,
                                required=True,
                                help="Atom string (e.g. 'H 0 0 0; H 0 0 0.74')",
                )
                parser.add_argument("--basis", type=str, default="sto-3g", help="Basis set")
                parser.add_argument("--charge", type=int, default=0, help="Total charge")
                parser.add_argument("--spin", type=int, default=0, help="Total spin (2S)")
                parser.add_argument(
                                "--mapping",
                                type=str,
                                default="jordan_wigner",
                                choices=["jordan_wigner", "bravyi_kitaev"],
                                help="Qubit mapping",
                )
                parser.add_argument(
                                "--output", type=str, default="hamiltonian.json", help="Output JSON file"
                )

                args = parser.parse_args()

                # Geometry
                print("Parsing geometry...", flush=True)
                geometry = parse_geometry(args.atom)
                if not geometry:
                                print(json.dumps({"error": "Empty geometry parsed."}))
                                sys.exit(1)

                # Multiplicity = 2S + 1
                multiplicity = args.spin + 1

                # Run PySCF
                try:
                                print("Initializing molecule...", flush=True)
                                molecule = MolecularData(
                                                geometry, args.basis, multiplicity, args.charge, data_directory="."
                                )

                                print("Getting reference values...", flush=True)
                                get_god_values(args.atom, basis=args.basis, charge=args.charge, spin=args.spin)

                                print(f"Running PySCF ({args.basis})...", flush=True)
                                molecule = run_pyscf(molecule, run_scf=1)

                                # Get Hamiltonian
                                print("Extracting Hamiltonian...", flush=True)
                                molecular_hamiltonian = molecule.get_molecular_hamiltonian()
                                fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)

                                # Mapping
                                print(f"Applying {args.mapping} mapping...", flush=True)
                                if args.mapping == "bravyi_kitaev":
                                                qubit_hamiltonian = bravyi_kitaev(fermion_hamiltonian)
                                else:
                                                qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

                                # Format JSON
                                print("Formatting output...", flush=True)
                                json_output = {}
                                for i, term in enumerate(qubit_hamiltonian.terms):
                                                coeff = qubit_hamiltonian.terms[term]
                                                pauli_str = format_pauli_term(term)

                                                # Formatting as string for JSON consistency with the previous structure
                                                json_output[f"term_{i}"] = {
                                                                "coefficient": str(
                                                                                coeff.real if abs(coeff.imag) < 1e-10 else coeff
                                                                ),  # Handle complex? Usually real for molecules
                                                                "pauli_string": pauli_str,
                                                }

                                # Add metadata
                                json_output["n_qubits"] = molecule.n_orbitals * 2
                                json_output["n_orbitals"] = molecule.n_orbitals
                                json_output["multiplicity"] = multiplicity
                                json_output["charge"] = args.charge
                                json_output["basis"] = args.basis
                                json_output["n_electrons"] = molecule.n_electrons

                                # Write File
                                with open(args.output, "w") as f:
                                                json.dump(json_output, f, indent=2)

                                print(
                                                json.dumps(
                                                                {
                                                                                "status": "success",
                                                                                "file": args.output,
                                                                                "n_qubits": molecule.n_orbitals * 2,
                                                                                "n_terms": len(qubit_hamiltonian.terms),
                                                                }
                                                )
                                )

                except Exception as e:
                                print(json.dumps({"error": str(e)}))
                                sys.exit(1)


if __name__ == "__main__":
                main()
