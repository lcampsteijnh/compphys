#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <chrono>

// Define struct to hold results from Ising simulation
struct IsingResult {
    double eps_mean_per_spin;
    double m_abs_mean_per_spin;
    double Cv_per_spin;
    double chi_per_spin;
    std::vector<double> E_vals;
};

// Periodic boundary conditions helpers
inline int ip(int i, int L) {
    return (i + 1 == L) ? 0 : i + 1;
}

inline int im(int i, int L) {
    return (i == 0) ? (L - 1) : (i - 1);
}

// Function to run Ising model simulation
IsingResult run_ising(int L, double T, int n_cycles, bool rnd_init, double seed) {

    // Defining constants and parameters
    const int N = L *  L;
    const double J = 1.0; // Interaction strength [J]
    const double kb = 1.0; // Boltzmann constant [J/K]
    const double beta = 1.0 / (kb * T);

    // Defining storage for observables
    double E_tot = 0.0; double E2_tot = 0.0;
    double  m_abs_tot = 0; double M2_tot = 0;
    std::vector<double> E_vals;

    // RNG
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<int> pick_spin(0, L - 1);

    // Define lattice and initialise spins
    std::vector<int> s(N);

    if (rnd_init) {
        // Random initial spin configuration
        for (int i = 0; i < N; ++i) {
            s[i] = (uniform(rng) < 0.5) ? 1 : -1;
        }        
    }
    else {
        // Ordered initial spin configuration
        for (int i = 0; i < N; ++i) {
            s[i] = 1;
        }
    }
    
    // 2D --> 1D lattice indexing for efficiency
    auto idx = [L](int i, int j) {
        return i * L + j;
    };

    // Precompute neighbour indices to avoid conditional statements in main loop (per problem description)
    std::vector<int> n1(N), n2(N), n3(N), n4(N);
    // For each site, compute indices of its four neighbours
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            int k = idx(i, j); // current site
            n1[k] = idx(ip(i, L), j); // right neighbour
            n2[k] = idx(im(i, L), j); // left neighbour
            n3[k] = idx(i, ip(j, L)); // down neighbour
            n4[k] = idx(i, im(j, L)); // up neighbour
        }
    }

    // Compute initial observables
    double E = 0.0;
    int M = 0;
    
    // Iterate over full grid
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            int k = idx(i, j);
            int sij = s[k];
            
            // Compute energy contribution from right and down neighbours only to avoid double counting
            E += -J * sij * (s[n1[k]] + s[n3[k]]);

            // Compute magnetisation
            M += sij;
        }
    }
    
    // Store initial energy
    E_vals.push_back(E);

    // Precomputing all possible Boltzmann factors (indexed by dE + 8 to avoid negative indices)
    std::vector<double> exp_dE(17);
    for (int dE = -8; dE <= 8; dE += 4) {
        exp_dE[dE + 8] = std::exp(-beta * dE);
    }

    // Main Monte Carlo loop
    for (int cycle = 0; cycle < n_cycles; cycle++) {
        // Monte Carlo cycles
        for (int n = 0; n < N; n++) {
            // Pick random spin to flip
            int i = pick_spin(rng);
            int j = pick_spin(rng);
            int k = idx(i, j);
            int sij = s[k];

            // Compute energy change for flipping spin
            int s_sum = s[n1[k]] + s[n2[k]] + s[n3[k]] + s[n4[k]];
            int dE_prime = 2 * J * sij * s_sum;

            // Metropolis criterion
            if (dE_prime <= 0 || uniform(rng) < exp_dE[dE_prime + 8]) {
                // If accepted, update spin, energy and magnetisation
                s[k] *= -1;
                E += dE_prime;
                M += - 2 * sij;
            }
        }

        // Accumulate observables
        E_vals.push_back(E);
        E_tot += E; E2_tot += E * E;
        m_abs_tot += std::abs(M); M2_tot += M * M;
    }

    // Normalise results and calculate specific heat and susceptibility
    double norm = 1.0 / double(n_cycles);
    E_tot *= norm; E2_tot *= norm; m_abs_tot *= norm; M2_tot *= norm;
    double Cv_mean = (E2_tot - E_tot * E_tot) * beta / (N * T);
    double chi_mean = (M2_tot - m_abs_tot * m_abs_tot) * beta / N;

    // Return final (mean) results in struct
    IsingResult result;
        result.eps_mean_per_spin = E_tot / N;
        result.m_abs_mean_per_spin = m_abs_tot / N;
        result.Cv_per_spin = Cv_mean;
        result.chi_per_spin = chi_mean;
        result.E_vals = E_vals;
    return result;
}

int main() {
    // Defining parameters
    int n_cycles = 1e7;
    int n_walkers = 8;
    bool rnd_init = true;

    // Iteratiing over L and T values
    for (int L : {40, 60, 80, 100}) {
        std::ofstream file("results_L" + std::to_string(L) + ".txt");
        file << std::setprecision(17) << std::scientific;
        file << "#T eps_mean_per_spin m_abs_mean_per_spin Cv_per_spin chi_per_spin\n";
        for (double T : {2.1, 2.13, 2.16, 2.19, 2.22, 2.25, 2.26, 2.27, 2.28, 2.285, 2.29, 2.295, 2.30, 2.31, 2.32, 2.33, 2.34, 2.35, 2.37, 2.4}) {
            // Run multiple walkers in parallel
            std::vector<IsingResult> results(n_walkers);
            #pragma omp parallel for
            for (int w = 0; w < n_walkers; w++) {
                // Unique seed per walker to ensure independent (but reproducible) streams
                int seed = 1234 + w * 100;
                results[w] = run_ising(L, T, n_cycles/n_walkers, rnd_init, seed);
            }

            // Combine results from all walkers as simple averages
            double eps_mean_per_spin = 0.0; double m_abs_mean_per_spin = 0.0; double Cv_per_spin = 0.0; double chi_per_spin = 0.0;

            for (int w = 0; w < n_walkers; w++) {
                eps_mean_per_spin += results[w].eps_mean_per_spin;
                m_abs_mean_per_spin += results[w].m_abs_mean_per_spin;
                Cv_per_spin += results[w].Cv_per_spin;
                chi_per_spin += results[w].chi_per_spin;
            }

            eps_mean_per_spin /= n_walkers;
            m_abs_mean_per_spin /= n_walkers;
            Cv_per_spin /= n_walkers;
            chi_per_spin /= n_walkers;

            // Write results to file
            file << T << " " << eps_mean_per_spin << " " << m_abs_mean_per_spin << " " << Cv_per_spin << " " << chi_per_spin << "\n";
            std::cout << "L = " << L << ", T=" << T << " done.\n";
        };
        file.close();  
        std::cout << "Results for L = " << L << " written to file.\n"; 
    };
    

    // OpenMP Speedup testing
    /*
    int L_test = 20;
    double T_test = 2.4;
    int total_cycles = 1e6;
    int N_runs = 10;

    // Output file for speedup results
    std::ofstream file("speedup_results.txt");
    file << "n_threads  avg_runtime" << std::endl;

    // Iterating over different number of threads
    for (int p : {1, 2, 4, 6, 8, 10}) {
        // Set number of threads
        omp_set_num_threads(p);

        // Distribute cycles among walkers
        int n_walkers = p;
        int cycles_per_walker = total_cycles / n_walkers;

        // Iterate over multiple identical runs for averaging
        for (int r = 0; r < N_runs; r++) {

            // Data structures for results and seeds
            std::vector<IsingResult> results(n_walkers);
            std::vector<int> seeds(n_walkers);

            // Unique seeds per walker
            for (int w = 0; w < n_walkers; w++) {
                seeds[w] = 1234 + 123 * w + 12 * r;
            }

            // Set start time
            auto t0 = std::chrono::high_resolution_clock::now();

            // Parallel execution of walkers
            #pragma omp parallel for
            for (int w = 0; w < n_walkers; w++) {
                results[w] = run_ising(L_test, T_test, cycles_per_walker, true, seeds[w]);
            }

            // Set end time and compute runtime
            auto t1 = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(t1 - t0).count();
            
            // Log individual runtimes
            file << p << " " << runtime << std::endl;
        }
    }
    file.close();
    */

    return 0;
}