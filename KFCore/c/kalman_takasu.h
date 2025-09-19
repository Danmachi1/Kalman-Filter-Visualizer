#ifndef KALMAN_TAKASU_H
#define KALMAN_TAKASU_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Kalman Filter Update (Takasu Formulation).
 *
 * @param[in,out] x     State vector (n x 1)
 * @param[in,out] P     Covariance matrix (n x n) – upper triangle used
 * @param[in]     dz    Measurement residual (m x 1)
 * @param[in]     R     Measurement noise covariance (m x m)
 * @param[in]     Ht    Transposed measurement matrix H' (n x m)
 * @param[in]     n     State dimension
 * @param[in]     m     Measurement dimension
 * @param[in]     chi2_threshold  Outlier rejection threshold (0 to disable)
 * @param[out]    chi2 Optional output for chi-squared statistic (nullable)
 *
 * @return 0 on success, -1 on error, -2 if rejected as outlier.
 */
int kalman_takasu(float* x, float* P, const float* dz, const float* R,
                  const float* Ht, int n, int m,
                  float chi2_threshold, float* chi2);

/**
 * @brief Kalman Filter Prediction Step.
 *
 * @param[in,out] x   (optional) State vector (n x 1)
 * @param[in,out] P   Covariance matrix (n x n)
 * @param[in]     Phi State transition matrix (n x n)
 * @param[in]     G   Process noise gain matrix (n x r)
 * @param[in]     Q   Process noise covariance diagonal (r x 1)
 * @param[in]     n   State dimension
 * @param[in]     r   Noise dimension
 */
void kalman_predict(float* x, float* P, const float* Phi,
                    const float* G, const float* Q, int n, int r);

#ifdef __cplusplus
}
#endif

#endif // KALMAN_TAKASU_H
