#define FMT_HEADER_ONLY 
#define _USE_MATH_DEFINES

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

// On inclut Eigen pour faire des maths de brute
#include <Eigen/Dense>
#include <Eigen/Sparse>

// On inclut NLopt 
#include <nlopt.hpp>

// On inclut Quest pour faire de la physique quantique
#include <quest.h>

// On inclut les classes de physique et d'ansatz
#include "core/ansatz.hpp"
#include "core/physics.hpp"

// On inclut les bibliothèques graphiques
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

// On inclut les bibliothèques standards
#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <stdio.h>
#include <vector>
#include <complex>
#include <cmath>
#include <bitset>


// -------------------------------------------------------
// Structure pour passer les données à la fonction de coût
// NLopt prend une fonction C-style, on encapsule tout dans
// une struct pour éviter les variables globales
// -------------------------------------------------------
struct VQEData {
    HEA& ansatz;
    Qureg& qubits;
    PauliStrSum& hamiltonian;
    std::vector<std::string>& paulis;
};

// -------------------------------------------------------
// Fonction de coût appelée par NLopt à chaque itération
// Signature imposée par NLopt :
//   - params : le vecteur de paramètres courant
//   - grad   : le gradient (nullptr si on utilise un algo sans gradient)
//   - data   : pointeur vers nos données custom (VQEData)
// -------------------------------------------------------
double cost_function(const std::vector<double>& params, std::vector<double>& grad, void* data) {
    VQEData* vqe = static_cast<VQEData*>(data);

    // Remet le registre quantique à |0000>
    // OBLIGATOIRE à chaque éval : sans ça on accumule l'état précédent
    initZeroState(vqe->qubits);
    applyPauliX(vqe->qubits, 0);
    applyPauliX(vqe->qubits, 1); //Etat hartree-fock |1100> pour commencer l'optimisation dans une zone plus prometteuse

    // Reconstruit le circuit avec les nouveaux paramètres
    vqe->ansatz.construct_circuit(vqe->qubits, params, vqe->paulis);

    // Calcule <ψ|H|ψ>, c'est notre énergie variationnelle
    qreal energy = calcExpecPauliStrSum(vqe->qubits, vqe->hamiltonian);

    return (double)energy;
}

void error_callback(int error, const char* description) {
        fprintf(stderr, "ERREUR GLFW (%d): %s\n", error, description);
    }


int main(int argc, char** argv) {
    spdlog::set_pattern("[%H:%M:%S] %v");
    spdlog::info(">>> QUANTUM BEAST - GUI : TEST INITIAL TERMINE <<<");
    spdlog::info(">>> LANCEMENT DE LA VERIFICATION BOUCLE VQE SIMPLE <<<");
    spdlog::info(">>> OUVERTURE DU JSON DE L'HAMILTONIEN <<<");

    // ------------------------------------------
    // Import de l'Hamiltonien depuis un fichier JSON
    // ------------------------------------------
    spdlog::info(">>> Chargement du fichier hamiltonian.json (Esperons qu'il existe) <<<");

    std::vector<std::string> pauli_strings;
    std::vector<qcomp> coefficients;

    std::ifstream file("hamiltonian.json");
    nlohmann::json j = nlohmann::json::parse(file);

    for (auto& [key, term] : j.items()) {
        std::string pauli = term["pauli_string"].get<std::string>();
        //reverse the string because we want qubit 0 on the right
        //std::reverse(pauli.begin(), pauli.end());
        std::string coeff_str = term["coefficient"].get<std::string>();

        // Parse "(real+imagj)"
        double real = 0.0, imag = 0.0;
        std::sscanf(coeff_str.c_str(), "(%lf%lfj)", &real, &imag);

        pauli_strings.push_back(pauli);
        coefficients.push_back(getQcomp(real, imag));
    }

    for (size_t i = 0; i < pauli_strings.size(); ++i) {
        spdlog::info("term {}: ({:.16f}+{:.16f}j) {}", i, coefficients[i].real(), coefficients[i].imag(), pauli_strings[i]);
    }

    // ------------------------------------------
    // Generation de l'Ansatz HEA (Hardware Efficient Ansatz)
    // ------------------------------------------
    
    initQuESTEnv();
    //On va utiliser une classe qui sera dans ansatz.hpp pour générer un HEA à partir de l'Hamiltonien.
    //On lui passe le nombre de qubits (dérivé du H) et la profondeur

    //On va utiliser une classe qui sera dans ansatz.hpp pour générer un HEA à partir de l'Hamiltonien.
    //On lui passe le nombre de qubits (dérivé du H) et la profondeur
    int num_qubits = 4;
    int depth = 3;
    std::vector<std::string> paulis; //Dummy, pas besoin dans un HEA
    HEA ansatz(num_qubits, depth);
    // On génère le circuit à partir de l'ansatz et de l'Hamiltonien
    int num_params = ansatz.get_num_params();
    std::vector<double> params(num_params, 0.1); // Paramètres init
    // On crée un registre quantique avec Quest
    Qureg qubits = createQureg(num_qubits);
    initZeroState(qubits);

    std::vector<PauliStr> hamiltonian_terms(pauli_strings.size());
    std::vector<int> count = {0, 1, 2, 3};
    for (size_t i = 0; i < pauli_strings.size(); ++i) {
        hamiltonian_terms[i] = getPauliStr(pauli_strings[i]);
        spdlog::info("term {}: {}", i, pauli_strings[i]);
        reportPauliStr(hamiltonian_terms[i]);
    }
    PauliStrSum hamiltonian = createPauliStrSum(hamiltonian_terms, coefficients);

    // -------------------------------------------------------
    // Initialisation de NLopt
    // On choisit COBYLA : algo sans gradient, robuste pour
    // les fonctions bruitées comme les circuits quantiques
    // LN = Local, No-gradient. Alternatives : LN_NELDERMEAD, LN_SBPLX
    // -------------------------------------------------------
    nlopt::opt optimizer(nlopt::LN_NELDERMEAD, num_params);

    // Borne les paramètres entre -2π et 2π
    // Physiquement les angles de rotation sont périodiques,
    // inutile de chercher au delà
    optimizer.set_lower_bounds(std::vector<double>(num_params, -2 * M_PI));
    optimizer.set_upper_bounds(std::vector<double>(num_params,  2 * M_PI));

    // La fonction à minimiser + pointeur vers nos données
    VQEData vqe_data{ansatz, qubits, hamiltonian, paulis};
    optimizer.set_min_objective(cost_function, &vqe_data);

    // Critères d'arrêt :
    // - ftol_rel : arrêt si amélioration relative < 1e-8 entre deux itérations
    // - maxeval  : arrêt au bout de 1000 appels à la fonction de coût quoi qu'il arrive
    optimizer.set_ftol_rel(1e-8);
    optimizer.set_maxeval(1000);

    // -------------------------------------------------------
    // Lancement de l'optimisation
    // params est modifié in-place : il contiendra les paramètres optimaux
    // minval reçoit l'énergie minimale trouvée
    // -------------------------------------------------------
    double minval;
    try {
        nlopt::result result = optimizer.optimize(params, minval);
        spdlog::info("Optimisation terminée. Code retour : {}", (int)result);
        spdlog::info("Energie minimale trouvée : {:.10f}", minval);
        spdlog::info("Energie FCI attendue     : -1.8572750302");
        spdlog::info("Erreur                   : {:.2e}", std::abs(minval - (-1.8572750302)));
    } catch (const std::exception& e) {
        spdlog::error("NLopt a planté : {}", e.what());
    }

    // ------------------------------------------
    //  GRAPHIQUE (La foire au pixels)
    // ------------------------------------------

    
    spdlog::info("Initialisation de la fenetre graphique avec GLFW, GLAD, ImGui et ImPlot... Accrochez-vous.");
    
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) {

        std::cerr << "Failed to initialize GLFW" << std::endl;
        spdlog::critical("GLFW a refuse de demarrer. Il a sans doute mieux a faire.");
        return -1;
    }
    
   glfwDefaultWindowHints();

    // 2. On arrête de rêver : OpenGL 4.6 c'est NON sous WSL souvent.
    // On demande la version 3.3. C'est le "Gold Standard". Tout tourne en 3.3.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    
    // 3. On reste propre (Core Profile)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Pour faire plaisir à MacOS si jamais

    // 4. LE COUPABLE : L'Antialiasing (MSAA)
    // GLXBadFBConfig arrive souvent parce que tu demandes implicitement du x4 ou x8
    // et le driver virtuel ne sait pas le gérer. On le tue.
    glfwWindowHint(GLFW_SAMPLES, 0); 

    // 5. On explicite les bits de couleur pour être sûr qu'il ne cherche pas du 10-bits HDR de l'espace
    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);

    // --- CRÉATION ---
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "QuantumBeast - The Omnibus Test", nullptr, nullptr);
    
    if (!window) { 
        spdlog::critical("GLXBadFBConfig a encore frappe. La creation de fenetre a echoue.");
        glfwTerminate(); 
        return -1; 
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // V-Sync activée, on n'est pas des sauvages.

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        spdlog::critical("GLAD n'a pas pu charger OpenGL. RIP.");
        return -1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext(); // <<-- INDISPENSABLE POUR IMPLOT
    
    ImGui::StyleColorsDark(); // Le thème clair est pour les psychopathes.
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Données pour notre graph custom
    std::vector<float> x_data(100);
    std::vector<float> y_data(100);
    for(int i=0; i<100; ++i) {
        x_data[i] = i * 0.1f;
        y_data[i] = std::sin(x_data[i]);
    }

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // --- FENETRE DE CONTROLE ---
        ImGui::Begin("Tableau de Bord Omniscient");
        
        ImGui::TextColored(ImVec4(0,1,0,1), "Systeme: EN LIGNE");
        ImGui::Separator();
        
        ImGui::Text("Tests Initiaux :");
        ImGui::BulletText("JSON: OK");
        ImGui::BulletText("GLFW: OK");
        ImGui::BulletText("GLAD: OK");
        ImGui::BulletText("ImGui: OK");
        ImGui::BulletText("ImPlot: OK");
        ImGui::BulletText("NLopt: OK");
        ImGui::BulletText("Eigen: OK");
        ImGui::BulletText("Quest: OK");


        ImGui::Separator();
        ImGui::Text("Donnees Temps Reel :");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        
        // --- TEST IMPLOT CUSTOM ---
        // On dessine un graphe directement ici pour prouver qu'on maîtrise la bête
        if (ImPlot::BeginPlot("Onde de Probabilite (Fake)")) {
            ImPlot::PlotLine("Sin(x)", x_data.data(), y_data.data(), 100);
            ImPlot::EndPlot();
        }

        ImGui::End();

        // --- DEMOS OFFICIELLES ---
        // On garde ça pour que tu puisses frimer avec les features que tu n'as pas codées
        ImGui::ShowDemoWindow();
        ImPlot::ShowDemoWindow();

        // Rendu
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f); // Noir presque total
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Nettoyage (si on arrive jusque là sans crash)
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();

    spdlog::info("Fermeture propre. Incroyable mais vrai.");
    return 0;
}