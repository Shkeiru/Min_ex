#include "core/ansatz.hpp"
#include "core/physics.hpp"
#include "quest.h" //
#include <Eigen/Sparse> //
#include <complex> //
#include <iostream> //
#include <vector> //


// Example usage to verify compilation and basic logic
int main() {
  std::cout << ">>> QUANTUM BEAST - CLI VERIFICATION <<<" << std::endl;

  // ---------------------------------------------------------
  // 1. Verify Physics Class (Sparse Matrix)
  // ---------------------------------------------------------
  std::cout << "\n[1] Verifying Physics Class..." << std::endl;
  Physics physics;

  // Create a dummy 4x4 Sparse Matrix for 2 qubits (Identity)
  Eigen::SparseMatrix<std::complex<double>> hamiltonian(4, 4);
  hamiltonian.insert(0, 0) = std::complex<double>(1.0, 0.0);
  hamiltonian.insert(1, 1) = std::complex<double>(1.0, 0.0);
  hamiltonian.insert(2, 2) = std::complex<double>(1.0, 0.0);
  hamiltonian.insert(3, 3) = std::complex<double>(1.0, 0.0);
  hamiltonian.makeCompressed();

  physics.set_matrix(hamiltonian);

  std::cout << "Physics: Hamiltonian Matrix set (Size: "
            << physics.get_matrix().rows() << "x" << physics.get_matrix().cols()
            << ")" << std::endl;

  // ---------------------------------------------------------
  // 2. Verify Ansatz Class
  // ---------------------------------------------------------
  std::cout << "\n[2] Verifying Ansatz Class..." << std::endl;
  int num_qubits = 2;
  int depth = 1;
  Ansatz ansatz(num_qubits, depth);

  // Check parameter count for Hardware Efficient Ansatz
  int num_params = ansatz.get_num_params(AnsatzType::HardwareEfficient);
  std::cout << "Ansatz (HEA): Required parameters = " << num_params
            << std::endl;

  // Create dummy parameters
  std::vector<double> params(num_params, 0.5); // Fill with 0.5

  // ---------------------------------------------------------
  // 3. Verify QuEST Integration
  // ---------------------------------------------------------
  std::cout << "\n[3] Verifying QuEST Integration..." << std::endl;

  // Initialize QuEST environment
  QuESTEnv env = createQuESTEnv();

  // Create Quantum Register
  Qureg qubits = createQureg(num_qubits, env);
  initZeroState(qubits);

  // Apply Ansatz
  try {
    ansatz.construct_circuit(qubits, params, AnsatzType::HardwareEfficient);
    std::cout << "Ansatz applied successfully." << std::endl;

    // Measure probability of |00>
    qreal prob = calcProbOfQubitOutcome(qubits, 0, 0);
    std::cout << "Probability of qubit 0 being 0: " << prob << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error applying ansatz: " << e.what() << std::endl;
  }

  // Cleanup
  destroyQureg(qubits, env);
  destroyQuESTEnv(env);

  std::cout << "\n>>> VERIFICATION COMPLETE <<<" << std::endl;
  return 0;
}
