#include <armadillo>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>

using arma::sp_cx_mat;
using arma::cx_double;
using arma::cx_vec;
using arma::cx_mat;

// Index helper function
inline int idx(int i, int j, int M) {
    int m = M - 2;
    int k = (j - 1) * m + (i - 1);
    return k;
}

// Matrix setup function
void matrix_setup(sp_cx_mat& A, sp_cx_mat& B, cx_vec& a, cx_vec& b, cx_double r) {
    // Defining the size of the matrices
    int N = a.n_elem;
    int m = static_cast<int>(std::sqrt(static_cast<double>(N))); // since N = m * m
    A.zeros(N, N);
    B.zeros(N, N);

    // Fill matrices A and B with appropriate values
    for (int j = 1; j <= m; j++) {
        for (int i = 1; i <= m; i++) {
            // Flat index calculation
            int k = idx(i, j, m + 2);

            // Diagonal elements
            A(k, k) = a(k);
            B(k, k) = b(k);

            // Left neighbour
            if (i > 1) {
                int kl = k - 1;
                A(k, kl) = -r;
                B(k, kl) = r;
            }

            // Right neighbour
            if (i < m) {
                int kr = k + 1;
                A(k, kr) = -r;
                B(k, kr) = r;
            }

            // Bottom neighbour
            if (j > 1) {
                int kb = k - m;
                A(k, kb) = -r;
                B(k, kb) = r;
            }

            // Top neighbour
            if (j < m) {
                int kt = k + m;
                A(k, kt) = -r;
                B(k, kt) = r;
            }
        }
    }
}

// Function to build matrices A and B
void build_matrices(sp_cx_mat& A, sp_cx_mat& B, double dt, double h, int M, arma::mat& V) {
    // Defining sizes
    int m = M - 2;
    int N = m * m;
    cx_double I(0.0, 1.0);

    // Initialize coefficient vectors
    cx_vec a(N), b(N);
    cx_double r = I * dt / (2.0 * h * h); // as given in the paper

    // Fill coefficient vectors
    for (int j = 1; j <= m; j++) {
        for (int i = 1; i <=m; i++) {
            // Flat index calculation
            int k = idx(i, j, M);
            
            // Potential at grid point
            double vij = V(i, j);

            // Diagonal elements
            a(k) = 1.0 + 4.0 * r + I * dt * vij / 2.0;
            b(k) = 1.0 - 4.0 * r - I * dt * vij / 2.0;
        } 
    }
    // Setup matrices A and B using the helper function
    matrix_setup(A, B, a, b, r);
}

// Function to initialize the wavepacket
void init_wavepacket(cx_mat& u, int M, double h, double x_c, double y_c, double sigma_x, double sigma_y, double p_x, double p_y, arma::mat& V) {
    
    // Initialize wavefunction matrix
    u.zeros(M, M);

    // Constants
    cx_double I(0.0, 1.0);
    double wall_cutoff = 0.5 * V.max(); // Define wall cutoff based on potential

    // Fill the wavefunction matrix
    for (int j = 1; j <= M - 2; j++) {
        double y = j * h;
        for (int i = 1; i <= M - 2; i++) {
            double x = i * h;

            // Check for walls in potential
            if (V(i, j) > wall_cutoff) {
                u(i, j) = cx_double(0.0, 0.0);
                continue;
            }

            // Gaussian wavepacket formula
            double dx = x - x_c; double dy = y - y_c;

            // Calculate Gaussian and phase factors
            double gauss = std::exp( - (dx * dx) / (2.0 * sigma_x * sigma_x) - (dy * dy) / (2.0 * sigma_y * sigma_y));
            cx_double phase = std::exp(I * (p_x * x + p_y * y));

            u(i, j) = gauss * phase;
        }
    }

    // Normalize the wavefunction
    double norm_squared = 0.0;

    // Calculate norm squared
    for (int j = 1; j <= M - 2; j++) {
        for (int i = 1; i <= M - 2; i++) {
            norm_squared += std::norm(u(i, j));
        }
    }

    // Divide by norm to normalize
    double norm = std::sqrt(norm_squared);
    u /= norm;
}

// Function to calculate total probability
double total_prob(const arma::cx_mat& u) {
    double P = 0.0;

    // Sum over all grid points
    for (arma::uword j = 0; j < u.n_cols; j++) {
        for (arma::uword i = 0; i < u.n_rows; i++) {
            P += std::norm(u(i, j));
        }
    }

    // Return total probability
    return P;
}

// Read parameters from file
bool read_parameters(const std::string& filename, double& h, double& dt, double& T, double& x_c, double& y_c, double& sigma_x, double& sigma_y, double& p_x, double& p_y, double& v0, std::string& pot_file, std::string& out_base) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening parameter file: " << filename << std::endl;
        return false;
    }

    // Expect parameters in specific order
    infile >> h >> dt >> T >> x_c >> y_c >> sigma_x >> sigma_y >> p_x >> p_y >> v0 >> pot_file >> out_base;

    infile.close();
    return true;
}

int main() {
    using namespace arma;

    // Read parameters from file
    double h, dt, T;
    double x_c, y_c;
    double sigma_x, sigma_y;
    double p_x, p_y;
    double v0;
    std::string pot_file, out_base;

    const std::string param_file = "parameters.txt";
    if (!read_parameters(param_file, h, dt, T, x_c, y_c, sigma_x, sigma_y, p_x, p_y, v0, pot_file, out_base)) {
        return 1; // Exit if reading parameters failed
    }

    // Derive other quantities
    int M = int(1.0 / h) + 1;
    int n_steps = static_cast<int>(T / dt);
    int m = M - 2;
    int N = m * m;

    // Potential
    mat V = zeros<arma::mat>(M, M);
    if (!V.load(pot_file)) {
        std::cerr << "Error loading potential file: " << pot_file << std::endl;
        return 1;
    }

    // Scale potential
    V *= v0;

    // Initialize wavepacket
    cx_mat u(M, M);
    init_wavepacket(u, M, h, x_c, y_c, sigma_x, sigma_y, p_x, p_y, V);

    // Build matrices A and B
    sp_cx_mat A, B;
    build_matrices(A, B, dt, h, M, V);

    // Time evolution
    cx_cube u_tot(M, M, n_steps +1); // Store wavefunction at each time step
    u_tot.slice(0) = u;

    // Vectors for linear system
    cx_vec u_vec(N), b_vec(N), u_next(N);

    std::string prob_file = out_base + "_total_prob.raw";
    std::ofstream prob(prob_file);
    prob << std::setprecision(18) << std::scientific;

    for (int n = 0; n <n_steps; n++) {
        // Flatten current wavefunction into vector
        int k = 0;
        for (int j = 1; j <= M - 2; j++) {
            for (int i = 1; i <= M - 2; i++) {
                u_vec(k++) = u(i, j);
            }
        }

        // Compute right-hand side
        b_vec = B * u_vec;

        // Solve linear system using sparse solver
        spsolve(u_next, A, b_vec); 
        // NOTE: In the paper I write that we avoid refactorisation of A at each timestep. I see now (too late to edit the paper)
        // that this implementation clearly does not do that. It is however very possible to implement (using eg. SuperLU), so I should
        // have put it in the "Future Work" section instead :)

        // Reshape solution back into matrix
        k = 0;
        for (int j = 1; j <= M - 2; j++) {
            for (int i = 1; i <= M - 2; i++) {
                cx_double val = u_next(k++);
                u(i, j) = val;
                u_tot(i, j, n + 1) = val;
            }
        }

        // Calculate and record total probability
        double t = (n + 1) * dt;
        double P = total_prob(u);
        prob << t << " " << P << "\n";
    }
    prob.close();

    // Save final wavefunction data
    std::string raw_file = out_base + "wavefunction.raw";
    u_tot.save(raw_file, arma::raw_binary);

    std::cout << "Simulation complete. Final wavefunction saved to " << raw_file << ".\n" << "Total probability data saved to " << prob_file << ".\n";

    return 0;
}
