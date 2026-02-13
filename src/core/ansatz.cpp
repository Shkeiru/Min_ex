#include "ansatz.hpp"
#include <iostream> // Nécessaire ici pour std::cerr et std::endl

// ==========================================
// Implémentation de HardwareEfficientAnsatz
// ==========================================

// Nom
std::string HEA::get_name() const {
    return "HEA Ansatz, depth " + std::to_string(depth) 
    + ", qubits " + std::to_string(num_qubits)
    + ", Type : RX-RY-RZ + CNOT en ligne";
}

// Nombre de paramètres (3 par qubit par couche)
int HEA::get_num_params() const {
    return 3 * num_qubits * depth;
}
    
// La Construction du circuit (Le gros morceau)
void HEA::construct_circuit(Qureg qubits, const std::vector<double>& params, const std::vector<std::string>& pauli_strings) {
    // 1. Sécurité
    if (params.size() != get_num_params()) {
        std::cerr << "[Ansatz] ERREUR: Recu " << params.size() 
                  << " params, attendu " << get_num_params() << std::endl;
        return;
    }

    // 2. Construction du circuit HEA
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < num_qubits; ++j) {
            int param_index = 3 * (i * num_qubits + j);
            // Appliquer les rotations RX, RY, RZ
            applyRotateX(qubits, j, params[param_index]);
            applyRotateY(qubits, j, params[param_index + 1]);
            applyRotateZ(qubits, j, params[param_index + 2]);
        }
        // Appliquer les CNOT en ligne
        for (int j = 0; j < num_qubits - 1; ++j) {
            applyControlledPauliX(qubits, j, j + 1);
        }
    }
}

int HEA::get_num_qubits() const {
    return num_qubits;
}

int HEA::get_depth() const {
    return depth;
}