#include <iostream>
#include <vector>
#include <string>

// On inclut QuEST en mode C pour éviter les embrouilles
extern "C" {
    #include <QuEST.h>
}

#include <CLI/CLI.hpp>

int main(int argc, char** argv) {
    CLI::App app{"QuantumBeast CLI - Le Mode Brutal"};

    int num_qubits = 10;
    app.add_option("-n,--qubits", num_qubits, "Nombre de qubits");

    CLI11_PARSE(app, argc, argv);

    std::cout << "[CLI] Démarrage sur " << num_qubits << " qubits..." << std::endl;

    // Initialisation QuEST
    QuESTEnv env = createQuESTEnv();
    
    std::cout << "[CLI] Environnement QuEST initialisé." << std::endl;
    if (env.isDistributed) {
        std::cout << "[CLI] Mode MPI: OUI (" << env.numRanks << " rangs)" << std::endl;
    } else {
        std::cout << "[CLI] Mode MPI: NON (OpenMP seul)" << std::endl;
    }

    Qureg qubits = createQureg(num_qubits, env);
    initZeroState(qubits);

    std::cout << "[CLI] Vecteur d'état alloué. Application de Hadamard..." << std::endl;
    for(int i=0; i<num_qubits; i++) {
        hadamard(qubits, i);
    }

    qreal prob = getProbAmp(qubits, 0);
    std::cout << "[CLI] Probabilité état 0 : " << prob << std::endl;

    destroyQureg(qubits, env);
    destroyQuESTEnv(env);

    std::cout << "[CLI] Terminé." << std::endl;
    return 0;
}