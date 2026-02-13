#pragma once 

#include <quest.h>
#include <vector>
#include <complex>
#include <string>
#include <iostream>
#include <cmath>


// Classe parente Ansatz
// Elle peut être spécialisée en HEA, UCCSD, etc. selon les besoins (Note, les ADAPT seront gere a part)
// Cahier des charges: 
// - Classe abstraite
// - Une méthode virtuelle pure pour générer le circuit à partir de l'Hamiltonien et de la profondeur
// - Un destructeur virtuel pour assurer une bonne gestion de la mémoire 
// - Des methodes pour accéder aux paramètres de l'ansatz (profondeur, nombre de qubits, etc.) si besoin

class Ansatz {


public:
    virtual ~Ansatz() = default; // Destructeur virtuel

    virtual void construct_circuit(Qureg qubits, const std::vector<double> &params, const std::vector<std::string> &pauli_strings) = 0; // Méthode virtuelle pure pour construire le circuit

    virtual int get_num_params() const = 0; // Méthode virtuelle pure pour obtenir le nombre de qubits
    virtual std::string get_name() const = 0; // Méthode virtuelle pure pour obtenir la profondeur

};

// Classe HEA (Hardware Efficient Ansatz)
// Spécialisée pour construire un HEA à partir d'un Hamiltonien donné
class HEA : public Ansatz {


private : 
    int num_qubits;
    int depth;

public : 
    HEA(int num_qubits, int depth) : num_qubits(num_qubits), depth(depth) {} // Constructeur

    void construct_circuit(Qureg qubits, const std::vector<double> &params, const std::vector<std::string> &pauli_strings) override;

    int get_num_qubits() const;

    int get_depth() const;

    int get_num_params() const override;

    std::string get_name() const override;

};
