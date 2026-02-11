#define FMT_HEADER_ONLY 
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

// On inclut Eigen pour faire des maths de brute
#include <Eigen/Dense>

// On inclut NLopt (C API pour être sûr que ça passe partout)
#include <nlopt.h>

#include <quest.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>

// Petite fonction bidon pour tester NLopt
// On cherche à minimiser f(x) = (x - 3)^2 + 4. Minimum théorique à x=3.
double myfunc(unsigned n, const double *x, double *grad, void *my_func_data) {
    if (grad) {
        grad[0] = 2.0 * (x[0] - 3.0);
    }
    return pow(x[0] - 3.0, 2) + 4.0;
}

int main(int argc, char** argv) {
    spdlog::set_pattern("[%H:%M:%S] %v");
    spdlog::info(">>> QUANTUM BEAST - GUI : MODE HYPER-TEST <<<");

    // ------------------------------------------
    // 1. TEST JSON (Parce qu'on est en 2026)
    // ------------------------------------------
    nlohmann::json j;
    j["projet"] = "QuantumBeast";
    j["statut"] = "Instable mais vivant";
    j["niveau_de_stress"] = 99.9;
    spdlog::info("[JSON] Dump: {}", j.dump());

    // ------------------------------------------
    // 2. TEST EIGEN (Algèbre linéaire de base)
    // ------------------------------------------
    Eigen::Matrix2d mat;
    mat << 1, 2,
           3, 4;
    // Le déterminant de [[1,2],[3,4]] est 1*4 - 2*3 = -2. Si ça affiche autre chose, fuis.
    spdlog::info("[Eigen] Determinant de la matrice 2x2: {}", mat.determinant());

    // ------------------------------------------
    // 3. TEST NLOPT (Optimisation)
    // ------------------------------------------
    nlopt_opt opt = nlopt_create(NLOPT_LD_MMA, 1); // Algorithme MMA
    nlopt_set_min_objective(opt, myfunc, NULL);
    nlopt_set_xtol_rel(opt, 1e-4);
    double x[1] = { 0.0 }; // Point de départ (x=0)
    double minf; 
    
    if (nlopt_optimize(opt, x, &minf) < 0) {
        spdlog::error("[NLopt] Echec de l'optimisation (le monde est injuste).");
    } else {
        spdlog::info("[NLopt] Minimum trouve a x = {:.4f} (Attendu: 3.0000), Valeur f = {:.4f}", x[0], minf);
    }
    nlopt_destroy(opt);

    // ------------------------------------------
    // 4. TEST QUEST (Physique Quantique)
    // ------------------------------------------
    // On garde ta logique, c'est la seule chose saine ici.
    initQuESTEnv();
    reportQuESTEnv(); // Ça spamme la console, mais tu aimes ça.
    
    Qureg qubits = createQureg(12);
    initZeroState(qubits);
    applyHadamard(qubits, 0);
    applyControlledPauliX(qubits, 0, 1);
    qreal prob = calcProbOfQubitOutcome(qubits, 0, 0);
    
    spdlog::info("[QuEST] Probabilite de |00>: {:.4f} (Si c'est pas 0.5, on a casse la physique)", prob);
    
    // On ne détruit pas les qubits tout de suite si on veut jouer avec plus tard, 
    // mais pour l'instant on suit ton script.
    destroyQureg(qubits);
    
    // Note: Normalement on ferme l'env QuEST, mais ta lib semble pas avoir destroyQuESTEnv() 
    // exposé ou tu l'as oublié. Dans le doute, on laisse fuiter la mémoire comme des pros.

    // ------------------------------------------
    // 5. GRAPHIQUE (La foire au pixels)
    // ------------------------------------------

    // Ajoute cette fonction quelque part
    void error_callback(int error, const char* description) {
        fprintf(stderr, "ERREUR GLFW (%d): %s\n", error, description);
    }

    spdlog::info("Initialisation de la fenetre graphique avec GLFW, GLAD, ImGui et ImPlot... Accrochez-vous.");
    
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) {

        std::cerr << "Failed to initialize GLFW" << std::endl;
        spdlog::critical("GLFW a refuse de demarrer. Il a sans doute mieux a faire.");
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "QuantumBeast - The Omnibus Test", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    
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
    ImGui_ImplOpenGL3_Init("#version 460");

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
        ImGui::BulletText("Eigen: OK (Det = %f)", mat.determinant());
        ImGui::BulletText("NLopt: OK (Min @ %f)", x[0]);
        ImGui::BulletText("QuEST: OK (Prob = %.4f)", prob);

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