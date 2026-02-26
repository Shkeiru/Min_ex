//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file main_gui.cpp
 * @author Rayan MALEK
 * @date 2026-02-19
 * @brief Main entry point for the VQE Simulator GUI application.
 */

//------------------------------------------------------------------------------
//     DEFINES
//------------------------------------------------------------------------------

#define FMT_HEADER_ONLY
#define _USE_MATH_DEFINES

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

// compat
#include "core/compat.h"

// GLAD must be included before any other OpenGL header (and sometimes before
// windows.h)
#include <glad/glad.h>

#include "core/logger.hpp"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

// NLopt
#include <nlopt.hpp>

// Quest
#include <quest.h>

// Project Core Headers
#include "core/ansatz.hpp"
#include "core/physics.hpp"

// Project GUI Headers
#include "gui.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <GLFW/glfw3.h>

// Standard Library
#include <bitset>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <stdio.h>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
//     CALLBACKS
//------------------------------------------------------------------------------

/**
 * @brief GLFW error callback function.
 *
 * @param error The error code.
 * @param description A string description of the error.
 */
void error_callback(int error, const char *description) {
  fprintf(stderr, "ERREUR GLFW (%d): %s\n", error, description);
}

//------------------------------------------------------------------------------
//     MAIN FUNCTION
//------------------------------------------------------------------------------

/**
 * @brief Main application entry point.
 *
 * Initializes GLFW, OpenGL (via GLAD), ImGui, and the application GUI state.
 * runs the main event loop.
 *
 * @param argc Command line argument count.
 * @param argv Command line argument values.
 * @return int Exit status code.
 */
int main(int argc, char **argv) {
  // 1. Logging Initialization
  qb_log::init_logger();
  spdlog::info(">>> QUANTUM BEAST - GUI : TEST INITIAL TERMINE <<<");

  // 2. Graphics (GLFW) Initialization
  // ------------------------------------------
  //  GRAPHIQUE (La foire au pixels)
  // ------------------------------------------

  spdlog::info("Initialisation de la fenetre graphique avec GLFW, GLAD, ImGui "
               "et ImPlot... Accrochez-vous.");

  glfwSetErrorCallback(error_callback);
  if (!glfwInit()) {

    std::cerr << "Failed to initialize GLFW" << std::endl;
    spdlog::critical(
        "GLFW a refuse de demarrer. Il a sans doute mieux a faire.");
    return -1;
  }

  glfwDefaultWindowHints();

  // On demande la version 3.3. C'est le "Gold Standard". Tout tourne en 3.3.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

  // On reste propre (Core Profile)
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,
                 GL_TRUE); // Pour faire plaisir à MacOS si jamais

  // L'Antialiasing (MSAA) OFF pour eviter GLXBadFBConfig sous WSL
  glfwWindowHint(GLFW_SAMPLES, 0);

  // Bits de couleur explicites
  glfwWindowHint(GLFW_RED_BITS, 8);
  glfwWindowHint(GLFW_GREEN_BITS, 8);
  glfwWindowHint(GLFW_BLUE_BITS, 8);
  glfwWindowHint(GLFW_ALPHA_BITS, 8);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);
  glfwWindowHint(GLFW_STENCIL_BITS, 8);

  // --- CREATION DE LA FENETRE ---

  GLFWwindow *window = glfwCreateWindow(
      1280, 720, "QuantumBeast - The Omnibus Test", nullptr, nullptr);

  if (!window) {
    spdlog::critical(
        "GLXBadFBConfig a encore frappe. La creation de fenetre a echoue.");
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // V-Sync activée, on n'est pas des sauvages.

  // 3. GLAD Initialization
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    spdlog::critical("GLAD n'a pas pu charger OpenGL. RIP.");
    return -1;
  }

  // 4. ImGui Initialization
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext(); // <<-- INDISPENSABLE POUR IMPLOT

  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  ImGui::StyleColorsDark(); // Le thème clair est pour les psychopathes.

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  // 5. GUI Instantiation
  GUI gui;

  // 6. Main Loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Create a DockSpace that covers the entire viewport
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    // RENDER OUR NEW GUI
    gui.Render();

    ImGui::Render();
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.05f, 0.05f, 0.05f, 1.0f); // Noir presque total
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  // 7. Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  spdlog::info("Fermeture propre. Incroyable mais vrai.");
  return 0;
}