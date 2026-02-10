#pragma once

#include <vector>
#include <QuEST.h> // Core a le droit d'inclure ça, il compile bien.

class Simulation {
public:
    Simulation();
    ~Simulation();

    // Initialise l'environnement et les qubits
    void init(int num_qubits);

    // Applique un circuit de test (Hadamard + CNOT)
    void run_circuit_test();

    // Récupère la probabilité de l'état |0...0>
    double get_probability_zero();

private:
    QuESTEnv env;
    Qureg qubits;
    bool initialized = false;
};