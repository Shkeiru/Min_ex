// ==========================================
// 1. LA PHYSIQUE D'ABORD (CRUCIAL)
// ==========================================

// QuEST doit être le PREMIER include pour éviter les conflits
// avec std::complex ou les macros d'Eigen/Spdlog.
#include <QuEST.h>

// ==========================================
// 2. LES MATHS
// ==========================================
#include <Eigen/Dense>
#include <nlopt.hpp>

// ==========================================
// 3. LES UTILITAIRES
// ==========================================
#define FMT_HEADER_ONLY 
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

// ==========================================
// 4. LE GRAPHIQUE
// ==========================================
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <iostream>

// ==========================================
// MAIN
// ==========================================
int main(int argc, char** argv) {
    
    spdlog::set_pattern("[%H:%M:%S] %v");
    spdlog::info(">>> SYSTEME VQE HYBRIDE - DEMARRAGE <<<");

    // ------------------------------------------
    // TEST DES LIBS
    // ------------------------------------------
    
    // Test QuEST
    try {
        QuESTEnv env = createQuESTEnv();
        spdlog::info("QuEST    : OK (Distributed={})", env.isDistributed);
        
        // On reste léger : 2 qubits
        Qureg qubits = createQureg(2, env);
        initZeroState(qubits);
        
        // Un petit circuit
        hadamard(qubits, 0);
        controlledNot(qubits, 0, 1);
        
        qreal prob = getProbAmp(qubits, 0);
        spdlog::info("QuEST    : Bell State Prob |00> = {:.4f}", prob);

        destroyQureg(qubits, env);
        destroyQuESTEnv(env);
    } catch (...) {
        spdlog::error("QuEST    : CRITIQUE - Echec initialisation");
    }

    // ------------------------------------------
    // INITIALISATION FENETRE
    // ------------------------------------------
    if (!glfwInit()) return -1;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "QuantumBeast", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    // ImGui Init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    spdlog::info("GUI      : Operationnel. Boucle de rendu active.");

    // ------------------------------------------
    // BOUCLE DE RENDU
    // ------------------------------------------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Tes fenêtres ici
        ImGui::ShowDemoWindow();
        ImPlot::ShowDemoWindow();

        // Rendu
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }

    // Nettoyage
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}