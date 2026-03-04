//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file main_cli.cpp
 * @brief Command-Line Interface version of the VQE Simulator.
 */

#include "core/ansatz.hpp"
#include "core/compat.h"
#include "core/logger.hpp"
#include "core/physics.hpp"
#include "core/simulation.hpp"


#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


// Global flag for Ctrl-C termination
std::atomic<bool> keep_running{true};

void signal_handler(int signum) {
  spdlog::warn("Interruption (Ctrl-C) detectee! Arret en cours...");
  keep_running.store(false);
}

// Helper to map string optimizer to nlopt::algorithm
nlopt::algorithm get_nlopt_algorithm(const std::string &opt_str) {
  if (opt_str == "LN_NELDERMEAD" || opt_str == "Nelder-Mead")
    return nlopt::LN_NELDERMEAD;
  if (opt_str == "LN_COBYLA" || opt_str == "COBYLA")
    return nlopt::LN_COBYLA;
  if (opt_str == "LN_BOBYQA" || opt_str == "BOBYQA")
    return nlopt::LN_BOBYQA;
  if (opt_str == "LN_NEWUOA")
    return nlopt::LN_NEWUOA;
  if (opt_str == "LN_NEWUOA_BOUND")
    return nlopt::LN_NEWUOA_BOUND;
  if (opt_str == "LN_PRAXIS")
    return nlopt::LN_PRAXIS;
  if (opt_str == "LN_SBPLX")
    return nlopt::LN_SBPLX;
  if (opt_str == "GN_DIRECT")
    return nlopt::GN_DIRECT;
  if (opt_str == "GN_DIRECT_L")
    return nlopt::GN_DIRECT_L;
  if (opt_str == "GN_CRS2_LM")
    return nlopt::GN_CRS2_LM;
  if (opt_str == "GN_ISRES")
    return nlopt::GN_ISRES;
  if (opt_str == "GN_ESCH")
    return nlopt::GN_ESCH;
  if (opt_str == "LD_LBFGS" || opt_str == "L-BFGS")
    return nlopt::LD_LBFGS;
  if (opt_str == "LD_SLSQP" || opt_str == "SLSQP")
    return nlopt::LD_SLSQP;
  return nlopt::LN_NELDERMEAD; // Default
}

// Helper to replace hyphens with underscores and convert to lowercase
std::string format_mapping(std::string mapping) {
  std::transform(mapping.begin(), mapping.end(), mapping.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::replace(mapping.begin(), mapping.end(), '-', '_');
  return mapping;
}

int main(int argc, char **argv) {
  // Register signal handler
  std::signal(SIGINT, signal_handler);

  // Initialize logger
  qb_log::init_logger();

  // Initialize QuEST
  initQuESTEnv();

  CLI::App app{"VQE Simulator CLI"};

  // Molecule & Hamiltonian options
  std::string opt_atom;
  std::string opt_basis = "sto-3g";
  int opt_charge = 0;
  int opt_spin = 0;
  std::string opt_mapping = "jordan_wigner";

  app.add_option("--atom", opt_atom,
                 "Molecule string (e.g. \"H 0 0 0; H 0 0 0.735\")")
      ->required();
  app.add_option("--basis", opt_basis, "Basis set");
  app.add_option("--charge", opt_charge, "Molecule charge");
  app.add_option("--spin", opt_spin, "Molecule spin (2S)");
  app.add_option("--mapping", opt_mapping, "Mapping type");

  // VQE options
  std::string opt_optimizer = "LN_NELDERMEAD";
  int opt_max_iter = 1000;
  double opt_tolerance = 1e-6;
  int opt_shots = 0;
  std::string opt_ansatz = "HEA";
  int opt_hea_depth = 1;

  app.add_option("--optimizer", opt_optimizer, "NLopt algorithm");
  app.add_option("--max-iter", opt_max_iter, "Maximum number of evaluations");
  app.add_option("--tolerance", opt_tolerance, "Relative tolerance");
  app.add_option("--shots", opt_shots,
                 "Number of shots for noise (0 = statevector)");
  app.add_option("--ansatz", opt_ansatz, "Type d'ansatz (HEA ou UCCSD)");
  app.add_option("--hea-depth", opt_hea_depth, "Depth if Ansatz = HEA");

  // Diffraction Data options
  std::string opt_integrals = "";
  std::string opt_factors = "";
  double opt_lambda = 1.0;

  app.add_option("--integrals", opt_integrals, "Path to integrals file");
  app.add_option("--factors", opt_factors, "Path to experimental factors file");
  app.add_option("--lambda", opt_lambda, "Diffraction penalty lambda factor");

  // Output options
  std::string opt_out = "";
  app.add_option("--out", opt_out, "Explicit path for JSON output file");

  CLI11_PARSE(app, argc, argv);

  try {
    spdlog::info("Demarrage du VQE Simulator CLI...");

    std::string formatted_mapping = format_mapping(opt_mapping);

    // 1. Generate Hamiltonian via Python
    spdlog::info(">>> Generation Hamiltonien...");
    std::string command;
#ifdef _WIN32
    command = "wsl ";
#endif
    command += "python3 python/generate_hamiltonian.py"; // assumes available in
                                                         // current dir
    command += " --atom \"" + opt_atom + "\"";
    command += " --basis " + opt_basis;
    command += " --charge " + std::to_string(opt_charge);
    command += " --spin " + std::to_string(opt_spin);
    command += " --mapping " + formatted_mapping;

    spdlog::info("CMD: {}", command);

    FILE *pipe = _popen(command.c_str(), "r");
    if (!pipe) {
      spdlog::critical(
          "Impossible d'ouvrir le pipe pour la generation d'Hamiltonien.");
      finalizeQuESTEnv();
      return EXIT_FAILURE;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
      std::string msg(buffer);
      while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
        msg.pop_back();
      }
      if (!msg.empty()) {
        spdlog::info("PySCF: {}", msg);
      }
    }

    int ret = _pclose(pipe);
    if (ret != 0) {
      spdlog::critical("Erreur lors de l'execution du script Python. Code: {}",
                       ret);
      finalizeQuESTEnv();
      return EXIT_FAILURE;
    }

    // Parse hamiltonian.json
    std::ifstream f("hamiltonian.json");
    int num_paulis = 0;
    int num_qubits = 0;
    long long hilbert_space = 0;
    if (f.good()) {
      nlohmann::json hj;
      f >> hj;
      if (hj.contains("error")) {
        spdlog::critical("Erreur script Python: {}",
                         hj["error"].get<std::string>());
        finalizeQuESTEnv();
        return EXIT_FAILURE;
      }
      for (auto &[key, val] : hj.items()) {
        if (key.find("term_") == 0)
          num_paulis++;
      }
      if (hj.contains("n_qubits")) {
        num_qubits = hj["n_qubits"].get<int>();
        hilbert_space = (long long)std::pow(2, num_qubits);
      }
    } else {
      spdlog::critical("Fichier hamiltonian.json introuvable.");
      finalizeQuESTEnv();
      return EXIT_FAILURE;
    }

    spdlog::info("Hamiltonien charge. Qubits: {}, Termes: {}", num_qubits,
                 num_paulis);

    // 2. Load Physics
    Physics physics("hamiltonian.json");

    // 3. Setup Ansatz
    std::unique_ptr<Ansatz> ansatz;
    if (opt_ansatz == "HEA") {
      ansatz = std::make_unique<HEA>(physics.get_num_qubits(), opt_hea_depth);
    } else if (opt_ansatz == "UCCSD") {
      ansatz =
          std::make_unique<UCCSD>(physics.get_num_qubits(),
                                  physics.get_n_electrons(), formatted_mapping);
    } else {
      spdlog::critical("Ansatz inconnu: {}", opt_ansatz);
      finalizeQuESTEnv();
      return EXIT_FAILURE;
    }

    // 4. Create Simulation
    nlopt::algorithm algo = get_nlopt_algorithm(opt_optimizer);
    Simulation sim(physics, *ansatz, algo);
    sim.set_max_evals(opt_max_iter);
    sim.set_tolerance(opt_tolerance);
    sim.set_shots(opt_shots);
    sim.set_lambda(opt_lambda);

    // Required history arrays
    std::vector<double> iter_history;
    std::vector<double> energy_history;
    std::vector<std::vector<double>> probs_history;
    std::vector<std::vector<double>> params_history;

    iter_history.reserve(opt_max_iter);
    energy_history.reserve(opt_max_iter);
    probs_history.reserve(opt_max_iter);
    params_history.reserve(opt_max_iter);

    double best_energy = 1e9;
    std::vector<double> counts_values;

    // Callback
    auto callback = [&](int iter, double energy,
                        const std::vector<double> &probs,
                        const std::vector<double> &cb_params) {
      if (!keep_running.load()) {
        throw std::runtime_error("Interrupted by user");
      }
      iter_history.push_back((double)iter);
      energy_history.push_back(energy);
      probs_history.push_back(probs);
      params_history.push_back(cb_params);

      counts_values = probs;
      if (energy < best_energy)
        best_energy = energy;

      if (iter % 10 == 0 ||
          iter == 1) { // Log every 10 iterations to not flood terminal
        spdlog::info("Iter: {}, Energy: {:.6f}", iter, energy);
      }
    };

    spdlog::info(">>> Debut VQE... (Simulation)");
    std::vector<double> params(ansatz->get_num_params(), 0.1);

    double noisy_energy = 0.0;
    std::string status_message = "VQE Termine.";

    try {
      noisy_energy = sim.run(params, callback, opt_factors, opt_integrals);

      // Re-fetch final probabilities using the sim method for output
      counts_values = sim.get_probabilities(params);
    } catch (const std::exception &e) {
      spdlog::error("Simulation erreur: {}", e.what());
      status_message = "Erreur fatale VQE.";
    }

    // 5. Generate JSON strictly identical to GUI::SaveRun()
    nlohmann::json j;

    j["config"]["molecule"] = {
        {"atom_string", opt_atom},
        {"basis", opt_basis},
        {"charge", opt_charge},
        {"spin", opt_spin},
        {"mapping", opt_mapping} // GUI saves mappings[mapping_idx] which is e.g
                                 // "Jordan-Wigner", not the formatted one
    };

    j["config"]["vqe"] = {{"optimizer", opt_optimizer},
                          {"max_iterations", opt_max_iter},
                          {"shots", opt_shots},
                          {"hea_depth", opt_hea_depth},
                          {"ansatz", opt_ansatz}};

    j["results"] = {{"final_energy", noisy_energy},
                    {"best_exact_energy", best_energy},
                    {"status", status_message}};

    std::vector<nlohmann::json> history_arr;
    for (size_t i = 0; i < iter_history.size(); ++i) {
      nlohmann::json entry;
      entry["iteration"] = iter_history[i];
      entry["energy"] = energy_history[i];
      if (i < probs_history.size())
        entry["probabilities"] = probs_history[i];
      if (i < params_history.size())
        entry["parameters"] = params_history[i];
      history_arr.push_back(entry);
    }
    j["history"] = history_arr;

    j["state"]["probabilities"] = counts_values;

    std::vector<std::string> labels;
    int n_q = 0;
    if (counts_values.size() > 0)
      n_q = (int)std::log2(counts_values.size());
    for (size_t i = 0; i < counts_values.size(); ++i) {
      std::string bitstring = "";
      for (int b = 0; b < n_q; ++b) {
        bitstring += ((i >> (n_q - 1 - b)) & 1) ? "1" : "0";
      }
      labels.push_back(bitstring);
    }
    j["state"]["labels"] = labels;

    j["system"] = {{"simulator", "VQE Simulator C++ v1.0"},
                   {"num_qubits", num_qubits},
                   {"num_paulis", num_paulis}};

    std::string filename = opt_out;
    if (filename.empty()) {
      std::time_t now = std::time(nullptr);
      char buf[100];
      std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
      filename = "run_" + std::string(buf) + ".json";
    }

    std::ofstream o(filename);
    o << std::setw(4) << j << std::endl;

    spdlog::info("Run sauvegarde dans: {}", filename);

    finalizeQuESTEnv();

  } catch (const std::exception &e) {
    spdlog::critical("Erreur Critique: {}", e.what());
    finalizeQuESTEnv();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
