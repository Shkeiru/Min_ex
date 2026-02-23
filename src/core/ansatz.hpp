//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file ansatz.hpp
 * @author Rayan MALEK
 * @date 2026-02-19
 * @brief Definition of Ansatz abstract base class and derived classes (HEA,
 * UCCSD).
 */

#pragma once

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include <cmath>
#include <complex>
#include <iostream>
#include <quest.h>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
//     BASE CLASS
//------------------------------------------------------------------------------

/**
 * @class Ansatz
 * @brief Abstract base class for variational ansatz circuits.
 *
 * Defines the interface for generating quantum circuits based on a specific
 * strategy (e.g., Hardware Efficient, UCCSD).
 */
class Ansatz {

public:
  /**
   * @brief Virtual destructor to ensure proper cleanup of derived classes.
   */
  virtual ~Ansatz() = default;

  /**
   * @brief Constructs the quantum circuit for the ansatz.
   *
   * @param qubits The quantum register to apply operations to.
   * @param params The variational parameters.
   * @param pauli_strings The list of Pauli strings (needed for some ansatzes).
   */
  virtual void
  construct_circuit(Qureg qubits, const std::vector<double> &params,
                    const std::vector<std::string> &pauli_strings) = 0;

  /**
   * @brief Gets the number of variational parameters required.
   * @return int Number of parameters.
   */
  virtual int get_num_params() const = 0;

  /**
   * @brief Gets the name and description of the ansatz.
   * @return std::string Name/Description.
   */
  virtual std::string get_name() const = 0;

  /**
   * @brief Indicates if the ansatz mathematically preserves the particle
   * number.
   * @return bool True if particle number is preserved, false otherwise.
   */
  virtual bool preserves_particle_number() const = 0;
};

//------------------------------------------------------------------------------
//     HARDWARE EFFICIENT ANSATZ (HEA)
//------------------------------------------------------------------------------

/**
 * @class HEA
 * @brief Hardware Efficient Ansatz implementation.
 *
 * Uses a layered structure of single-qubit rotations (Rx, Ry, Rz) followed by
 * entangling gates (CNOT chain). Designed to be suitable for NISQ devices.
 */
class HEA : public Ansatz {

private:
  int num_qubits; ///< Number of qubits.
  int depth;      ///< Number of layers.

public:
  /**
   * @brief Constructs a new HEA object.
   *
   * @param num_qubits Number of qubits.
   * @param depth Depth of the ansatz (number of layers).
   */
  HEA(int num_qubits, int depth) : num_qubits(num_qubits), depth(depth) {}

  void
  construct_circuit(Qureg qubits, const std::vector<double> &params,
                    const std::vector<std::string> &pauli_strings) override;

  int get_num_qubits() const;

  int get_depth() const;

  int get_num_params() const override;

  std::string get_name() const override;

  bool preserves_particle_number() const override;
};

//------------------------------------------------------------------------------
//     UCCSD ANSATZ
//------------------------------------------------------------------------------

/**
 * @struct UCCSDExcitation
 * @brief Represents a single UCCSD excitation term.
 */
struct UCCSDExcitation {

  struct Term {
    std::string pauli;          ///< Pauli string (e.g., "X0 Y1").
    std::complex<double> coeff; ///< Complex coefficient.
  };
  std::vector<Term> terms;
};
enum class GateType { Hadamard, RX_PI_2, RX_MINUS_PI_2, CNOT, RZ_PARAM };

struct Instruction {
  GateType type;
  int target;
  int control; // Only for CNOT
  int param_idx;
  double angle_multiplier;
};
/**
 * @class UCCSD
 * @brief Unitary Coupled Cluster Singles and Doubles Ansatz.
 *
 * Implements the UCCSD ansatz, chemically inspired, by generating excitations
 * based on electron number and mapping (Jordan-Wigner, Bravyi-Kitaev).
 * Relies on an external Python script to generate the excitation list.
 */
class UCCSD : public Ansatz {

private:
  int num_qubits;
  int num_electrons;
  std::vector<UCCSDExcitation> excitations;
  std::vector<Instruction> compiled_tape;

public:
  ~UCCSD() override;
  /**
   * @brief Constructs a new UCCSD object.
   *
   * Triggers the generation of the `uccsd.json` file via python script.
   *
   * @param num_qubits Number of qubits.
   * @param num_electrons Number of electrons.
   * @param mapping Mapping type (default: "jordan_wigner").
   */
  UCCSD(int num_qubits, int num_electrons,
        std::string mapping = "jordan_wigner");

  void
  construct_circuit(Qureg qubits, const std::vector<double> &params,
                    const std::vector<std::string> &pauli_strings) override;

  int get_num_qubits() const;
  int get_num_params() const override;
  std::string get_name() const override;
  bool preserves_particle_number() const override;
};
