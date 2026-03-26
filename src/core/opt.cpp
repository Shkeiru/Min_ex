//------------------------------------------------------------------------------
//     AUTHORING
//------------------------------------------------------------------------------
/**
 * @file opt.cpp
 * @author Rayan MALEK
 * @date 2026-03-16
 * @brief SPSA optimizer implementation.
 */

//------------------------------------------------------------------------------
//     INCLUDES
//------------------------------------------------------------------------------

#include "opt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

//------------------------------------------------------------------------------
//     HELPERS
//------------------------------------------------------------------------------

namespace {

/**
 * @brief Clip a value into [lo, hi].  hi < lo is treated as unbounded.
 */
inline double clip(double v, double lo, double hi)
{
    if (lo <= hi) {
        v = std::max(v, lo);
        v = std::min(v, hi);
    }
    return v;
}

/**
 * @brief Apply per-component box constraints to x.
 */
void apply_bounds(std::vector<double>       &x,
                  const std::vector<double> &lb,
                  const std::vector<double> &ub)
{
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double lo = lb.empty() ? -std::numeric_limits<double>::max() : lb[i];
        const double hi = ub.empty() ?  std::numeric_limits<double>::max() : ub[i];
        x[i] = clip(x[i], lo, hi);
    }
}

/**
 * @brief Gain sequences (Spall 1998).
 *   aₖ = a / (A + k)^alpha
 *   cₖ = c / k^gamma
 *
 * k is 1-based (first iteration k=1).
 */
inline double gain_a(const SPSAParams &p, int k)
{
    return p.a / std::pow(p.A + static_cast<double>(k), p.alpha);
}

inline double gain_c(const SPSAParams &p, int k)
{
    return p.c / std::pow(static_cast<double>(k), p.gamma);
}

} // anonymous namespace

//------------------------------------------------------------------------------
//     IMPLEMENTATION
//------------------------------------------------------------------------------

SPSAResult spsa_optimize(
    std::function<double(const std::vector<double> &)> f,
    std::vector<double>                               &x,
    const std::vector<double>                         &lb,
    const std::vector<double>                         &ub,
    const SPSAParams                                  &p,
    int                                                max_evals,
    double                                             ftol_rel,
    unsigned int                                       seed)
{
    // ------------------------------------------------------------------ setup
    const std::size_t n = x.size();

    if (!lb.empty() && lb.size() != n)
        throw std::invalid_argument("spsa_optimize: lb size mismatch");
    if (!ub.empty() && ub.size() != n)
        throw std::invalid_argument("spsa_optimize: ub size mismatch");
    if (max_evals < 2)
        throw std::invalid_argument("spsa_optimize: max_evals must be >= 2");

    // RNG — Bernoulli ±1
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::bernoulli_distribution coin(0.5);

    // Working vectors
    std::vector<double> x_plus(n), x_minus(n), delta(n);
    std::vector<double> x_best = x;

    // Clip starting point into bounds
    apply_bounds(x, lb, ub);

    // Evaluate at initial point (counts toward budget)
    double f_curr  = f(x);
    double f_best  = f_curr;
    int    n_evals = 1;
    int    n_iters = 0;

    // ------------------------------------------------------------------ loop
    while (n_evals + 2 <= max_evals) {

        ++n_iters;
        const int    k  = n_iters;           // 1-based iteration index
        const double ak = gain_a(p, k);
        const double ck = gain_c(p, k);

        // --- Draw Bernoulli ±1 perturbation vector
        for (std::size_t i = 0; i < n; ++i)
            delta[i] = coin(rng) ? 1.0 : -1.0;

        // --- Build x± and apply bounds
        for (std::size_t i = 0; i < n; ++i) {
            x_plus[i]  = x[i] + ck * delta[i];
            x_minus[i] = x[i] - ck * delta[i];
        }
        apply_bounds(x_plus,  lb, ub);
        apply_bounds(x_minus, lb, ub);

        // --- Two function evaluations  (this is the entire cost of one step)
        const double f_plus  = f(x_plus);
        const double f_minus = f(x_minus);
        n_evals += 2;

        // --- Simultaneous gradient approximation
        //     ĝᵢ = (f+ − f−) / (2 cₖ Δᵢ)
        //     Update: xᵢ ← xᵢ − aₖ ĝᵢ
        const double diff = f_plus - f_minus;
        for (std::size_t i = 0; i < n; ++i)
            x[i] -= ak * diff / (2.0 * ck * delta[i]);

        apply_bounds(x, lb, ub);

        // --- Track best point seen
        //     We re-use f_plus / f_minus to avoid an extra evaluation.
        //     The updated x is not yet evaluated — record best among ±.
        if (f_plus < f_best) { f_best = f_plus;  x_best = x_plus;  }
        if (f_minus < f_best){ f_best = f_minus; x_best = x_minus; }

        // --- ftol_rel stopping criterion
        //     Compare the average of the two bracket values to f_curr.
        const double f_avg = 0.5 * (f_plus + f_minus);
        if (ftol_rel > 0.0 && std::abs(f_curr) > 0.0) {
            if (std::abs(f_avg - f_curr) / std::abs(f_curr) < ftol_rel)
                break;
        }
        f_curr = f_avg;
    }

    // Return best known point (not necessarily the last iterate)
    x = x_best;

    SPSAResult result;
    result.minval  = f_best;
    result.n_evals = n_evals;
    result.n_iters = n_iters;
    result.status  = (n_evals >= max_evals)
                         ? nlopt::MAXEVAL_REACHED
                         : nlopt::FTOL_REACHED;
    return result;
}

//------------------------------------------------------------------------------
//     CLASS SPSA_Optimizer IMPLEMENTATION
//------------------------------------------------------------------------------

SPSA_Optimizer::SPSA_Optimizer(int dim) : dim_(dim) {}

void SPSA_Optimizer::set_min_objective(vfunc f, void *f_data) {
  f_ = f;
  f_data_ = f_data;
}

void SPSA_Optimizer::set_maxeval(int m) {
  max_evals_ = m;
}

int SPSA_Optimizer::get_maxeval() const {
  return max_evals_;
}

void SPSA_Optimizer::set_ftol_rel(double tol) {
  ftol_rel_ = tol;
}

void SPSA_Optimizer::set_spsa_params(const SPSAParams &p) {
  params_ = p;
}

nlopt::result SPSA_Optimizer::optimize(std::vector<double> &x, double &minf) {
  if (!f_) {
    throw std::runtime_error("SPSA_Optimizer: objective function not set.");
  }

  // Wrap the vfunc into a std::function for the existing spsa_optimize
  auto f_wrapper = [this](const std::vector<double> &params) -> double {
    std::vector<double> dummy_grad;
    return this->f_(params, dummy_grad, this->f_data_);
  };

  SPSAResult res = spsa_optimize(f_wrapper, x, {}, {}, params_, max_evals_, ftol_rel_, 0);
  minf = res.minval;
  return res.status;
}