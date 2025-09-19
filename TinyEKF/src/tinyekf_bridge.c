#define EKF_N 4
#define EKF_M 2
#define TINYEKF_IMPLEMENTATION
#include "tiny_ekf.h"

// Global EKF instance
static ekf_t filter;

/**
 * Initializes the TinyEKF filter with diagonal covariance values.
 */
void tinyekf_init() {
    float pdiag[EKF_N] = {1.0f, 1.0f, 1.0f, 1.0f};
    ekf_initialize(&filter, pdiag);
}

/**
 * Runs the EKF update step.
 *
 * @param z  Measurement vector of size EKF_M
 * @param hx Expected measurement vector of size EKF_N
 * @param H  Jacobian matrix (EKF_M x EKF_N) as row-major array
 * @param R  Measurement noise covariance (EKF_M x EKF_M) as row-major array
 *
 * @return 1 if update succeeds, 0 if inversion fails
 */
int tinyekf_update(const float* z, const float* hx, const float* H, const float* R) {
    return ekf_update(&filter, z, hx, H, R) ? 1 : 0;
}

/**
 * Access a single element from the state vector.
 */
float tinyekf_get_state(int i) {
    return filter.x[i];
}

/**
 * Writes the full state vector to the output array.
 */
void tinyekf_get_state_vector(float* out) {
    for (int i = 0; i < EKF_N; i++) {
        out[i] = filter.x[i];
    }
}
