#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

void write_potential(const std::string& filename, int M, double h, int n_slits) {

    // Define wall parameters as per paper specifications
    double x_wall_center = 0.5;
    double wall_thickness = 0.02;
    double x_min_wall = x_wall_center - wall_thickness / 2.0;  // 0.49
    double x_max_wall = x_wall_center + wall_thickness / 2.0;  // 0.51

    double aperture = 0.05;
    double sep = 0.05;
    double y_center = 0.5;

    // Build slit intervals in y, symmetric around y=0.5
    struct opening {double y_min, y_max;};
    std::vector<opening> slits; 

    if (n_slits ==1) {

        // One slit centered at y=0.5
        double half = aperture / 2.0;
        slits.push_back({y_center - half, y_center + half});
    } else if (n_slits == 2) {

        // Two slits centered at +- (sep/2 + aperture/2)
        double offset = sep / 2.0 + aperture / 2.0;
        double c1 = y_center - offset;
        double c2 = y_center + offset;

        slits.push_back({c1 - aperture / 2.0, c1 + aperture / 2.0});
        slits.push_back({c2 - aperture / 2.0, c2 + aperture / 2.0});
    } else if (n_slits == 3) {

        // Three slits centered at y=0.5, and +- (sep + aperture)
        double half = aperture / 2.0;
        slits.push_back({y_center - half, y_center + half});

        // distance from center to upper/lower center = aperture + sep
        double offset = aperture + sep;
        double c1 = y_center - offset;
        double c2 = y_center + offset;

        slits.push_back({c1 - half, c1 + half});
        slits.push_back({c2 - half, c2 + half});
    }

    std::ofstream out(filename);

    // Loop over grid points and write potential values
    for (int i = 0; i < M; i++) {
        double x = i * h;
        for (int j = 0; j < M; j++) {
            double y = j * h;

            double vij = 0.0;

            // Check if inside wall region
            if (x >= x_min_wall && x <= x_max_wall) {

                // Check if inside any slit opening
                bool in_slit = false;
                for (const auto& slit : slits) {
                    if (y >= slit.y_min && y <= slit.y_max) {
                        in_slit = true;
                        break;
                    }
                }
                if (!in_slit) {
                    vij = 1.0;  // scaled by factor v0 in solver.cpp
                }
            }

            // Write potential value
            out << vij;
            if (j < M - 1) { out << " ";  // space between values in a row
            }
        }

        out << "\n";  // new line after each row
    }

    // Confirmation message
    std::cout << "Wrote " << filename << " with " << n_slits << " slits.\n";
}

int main() {

    // Read parameter from file
    double h = 0.0;

    std::ifstream param_file("parameters.txt");
    param_file >> h;
    std::cout << "Using h = " << h << "\n";

    // Determine grid size
    int M = int(1.0 / h) + 1;  // grid size

    // Generate potentials with different slit configurations
    write_potential("single_slit.txt", M, h, 1);
    write_potential("double_slit.txt", M, h, 2);
    write_potential("triple_slit.txt", M, h, 3);

    return 0;
}