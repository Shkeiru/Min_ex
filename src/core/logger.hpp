#pragma once

#include "compat.h"
#include <memory>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <vector>


namespace qb_log {

// Expose the ringbuffer sink so the GUI can read from it
extern std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> gui_sink;

/**
 * @brief Initialize the global spdlog logger with Dual-Sink (Stdout +
 * Ringbuffer).
 */
inline void init_logger() {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [Thread %t] %v");

  // Keep the last 1000 messages for the GUI
  gui_sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(1000);
  gui_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [Thread %t] %v");

  std::vector<spdlog::sink_ptr> sinks{console_sink, gui_sink};
  auto logger =
      std::make_shared<spdlog::logger>("qb_logger", sinks.begin(), sinks.end());

  // Set the global format and default logger
  spdlog::register_logger(logger);
  spdlog::set_default_logger(logger);

  // We want detailed logs (TRACE and DEBUG) for the library
  spdlog::set_level(spdlog::level::trace);
  spdlog::flush_on(spdlog::level::info);
}

// Global variable definition
inline std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> gui_sink = nullptr;

} // namespace qb_log
