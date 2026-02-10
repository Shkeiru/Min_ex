#include "simulation.hpp"
#include <iostream>

Simulation::Simulation() {
    // Constructeur vide pour l'instant
}

Simulation::~Simulation() {
    if (initialized) {
        destroyQureg(qubits, env);
        destroyQuESTEnv(env);
    }
}

void Simulation::init(int num_qubits) {
    if (initialized) {
        destroyQureg(qubits, env);
        destroyQuESTEnv(env);
    }
    
    env = createQuESTEnv();
    qubits = createQureg(num_qubits, env);
    initZeroState(qubits);
    initialized = true;
    
    std::cout << "[CORE] Simulation initialisee avec " << num_qubits << " qubits." << std::endl;
}

void Simulation::run_circuit_test() {
    if (!initialized) return;
    
    std::cout << "[CORE] Application Hadamard sur q[0]..." << std::endl;
    hadamard(qubits, 0);
    
    if (qubits.numQubitsRepresented > 1) {
        std::cout << "[CORE] Application CNOT sur q[0], q[1]..." << std::endl;
        controlledNot(qubits, 0, 1);
    }
}

double Simulation::get_probability_zero() {
    if (!initialized) return 0.0;
    return getProbAmp(qubits, 0);
}