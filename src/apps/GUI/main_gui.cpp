#define FMT_HEADER_ONLY 
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

// ON N'INCLUT PLUS QUEST.H ICI DIRECTEMENT
// On passe par notre wrapper propre
#include "simulation.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <iostream>

int main(int argc, char** argv) {
    spdlog::set_pattern("[%H:%M:%S] %v");
    spdlog::info(">>> QUANTUM BEAST - GUI <<<");

    // ------------------------------------------
    // TEST PHYSIQUE VIA LE COEUR
    // ------------------------------------------
    // C'est propre, ça ne crashera pas le compilateur
    initQuESTEnv();
    reportQuESTEnv();
    Qureg qubits = createQureg(12);
    reportQureg(qubits);
    initZeroState(qubits);
    applyHadamard(qubits, 0);
    applyControlledPauliX(qubits, 0, 1);
    qreal prob = calcProbOfQubitOutcome(qubits, 0, 0);
    spdlog::info("Probabilité de |00>: {:.4f}", prob);
    destroyQureg(qubits);
    



    // ------------------------------------------
    // GRAPHIQUE (Classique)
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

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Une petite fenêtre de contrôle
        ImGui::Begin("Controle Quantique");
        ImGui::Text("Etat: Operationnel");
        ImGui::End();

        // Démo pour vérifier que ImPlot marche
        ImPlot::ShowDemoWindow();

        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}