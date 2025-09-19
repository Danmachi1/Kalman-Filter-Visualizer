// online_noise_reestimator.hpp
#ifndef SKYFILTER_ONLINE_NOISE_REESTIMATOR_HPP_
#define SKYFILTER_ONLINE_NOISE_REESTIMATOR_HPP_

#include <vector>
#include <deque>     // For O(1) front erasures, replacing std::vector for history buffers
#include <Eigen/Dense>
#include <algorithm> // For std::min, std::max
#include <cmath>     // For std::sqrt

// Boost.Math for chi-squared distribution (optional, for proper consistency check)
#include <boost/math/distributions/chi_squared.hpp> // Uncommented for proper chi-squared test

/**
 * @brief The OnlineNoiseReEstimator class dynamically adjusts the process noise (Q)
 * and measurement noise (R) matrices based on observed filter behavior (innovation
 * sequence and prediction errors). It also performs covariance consistency checks.
 *
 * @note This class is not intrinsically thread-safe. If used in a multi-threaded
 * environment, external synchronization mechanisms (e.g., mutexes) must be applied
 * around calls to its methods that modify or access the history buffers.
 */
class OnlineNoiseReEstimator {
public:
    // Buffers for innovations and prediction errors over a sliding window
    // Using std::deque for O(1) push_back and pop_front operations.
    std::deque<Eigen::VectorXd> innovation_history;
    std::deque<Eigen::MatrixXd> innovation_covariance_history; // Store P_y
    std::deque<Eigen::VectorXd> prediction_error_history; // x_k|k - x_k|k-1

    int window_size_; // Window size for history
    double chi_squared_alpha_lower_; // Lower percentile for consistency test
    double chi_squared_alpha_upper_; // Upper percentile for consistency test

    // Parameters for Q/R adjustment heuristics
    double q_adjust_threshold_factor_; // e.g., 1.5 for 1.5x observed error variance
    double q_increase_factor_;         // e.g., 1.1 for 10% increase
    double q_decrease_factor_;         // e.g., 0.9 for 10% decrease

    double r_adjust_threshold_factor_; // e.g., 1.5 for 1.5x observed innovation variance
    double r_increase_factor_;         // e.g., 1.1 for 10% increase
    double r_decrease_factor_;         // e.g., 0.9 for 10% decrease

    /**
     * @brief Constructor for OnlineNoiseReEstimator.
     * @param window_size The size of the sliding window for history.
     * @param alpha_lower Lower percentile for the chi-squared consistency test (e.g., 0.05).
     * @param alpha_upper Upper percentile for the chi-squared consistency test (e.g., 0.95).
     * @param q_threshold_factor Threshold factor for Q adjustment (e.g., 1.5).
     * @param q_inc_factor Scaling factor to increase Q (e.g., 1.1).
     * @param q_dec_factor Scaling factor to decrease Q (e.g., 0.9).
     * @param r_threshold_factor Threshold factor for R adjustment (e.g., 1.5).
     * @param r_inc_factor Scaling factor to increase R (e.g., 1.1).
     * @param r_dec_factor Scaling factor to decrease R (e.g., 0.9).
     */
    OnlineNoiseReEstimator(int window_size = 30,
                           double alpha_lower = 0.05,
                           double alpha_upper = 0.95,
                           double q_threshold_factor = 1.5,
                           double q_inc_factor = 1.1,
                           double q_dec_factor = 0.9,
                           double r_threshold_factor = 1.5,
                           double r_inc_factor = 1.1,
                           double r_dec_factor = 0.9)
        : window_size_(window_size),
          chi_squared_alpha_lower_(alpha_lower),
          chi_squared_alpha_upper_(alpha_upper),
          q_adjust_threshold_factor_(q_threshold_factor),
          q_increase_factor_(q_inc_factor),
          q_decrease_factor_(q_dec_factor),
          r_adjust_threshold_factor_(r_threshold_factor),
          r_increase_factor_(r_inc_factor),
          r_decrease_factor_(r_dec_factor) {}

    /**
     * @brief Adds an innovation and its covariance to the history.
     * @param innovation The innovation vector (z_k - y_k|k-1).
     * @param innovation_covariance The innovation covariance matrix (P_y).
     */
    void addInnovation(const Eigen::VectorXd& innovation, const Eigen::MatrixXd& innovation_covariance) {
        innovation_history.push_back(innovation);
        innovation_covariance_history.push_back(innovation_covariance);
        if (innovation_history.size() > window_size_) {
            innovation_history.pop_front(); // O(1) for deque
            innovation_covariance_history.pop_front(); // O(1) for deque
        }
    }

    /**
     * @brief Adds a prediction error to the history.
     * @param prediction_error The prediction error vector (x_k|k - x_k|k-1).
     */
    void addPredictionError(const Eigen::VectorXd& prediction_error) {
        prediction_error_history.push_back(prediction_error);
        if (prediction_error_history.size() > window_size_) {
            prediction_error_history.pop_front(); // O(1) for deque
        }
    }

    /**
     * @brief Estimates and adjusts the process noise covariance (Q) based on prediction error drift.
     * @param current_Q The current process noise covariance matrix.
     * @param P_predicted The predicted state covariance matrix (P_k|k-1).
     * @param P_updated The updated state covariance matrix (P_k|k).
     * @return The adjusted Q matrix.
     */
    Eigen::MatrixXd estimateQ(const Eigen::MatrixXd& current_Q, const Eigen::MatrixXd& P_predicted, const Eigen::MatrixXd& P_updated) const {
        if (prediction_error_history.size() < window_size_ / 2) return current_Q; // Not enough data

        Eigen::MatrixXd sample_cov_error = Eigen::MatrixXd::Zero(current_Q.rows(), current_Q.cols());
        for (const auto& error : prediction_error_history) {
            sample_cov_error += error * error.transpose();
        }
        sample_cov_error /= prediction_error_history.size();

        // Simplified heuristic for Q adjustment.
        // The term (P_predicted - P_updated) represents the reduction in uncertainty
        // due to the measurement update. If the observed prediction error variance
        // (sample_cov_error) is consistently larger than this expected reduction,
        // it suggests that the process noise Q is underestimated.
        Eigen::MatrixXd expected_error_cov_reduction = P_predicted - P_updated;
        
        Eigen::MatrixXd adjusted_Q = current_Q;
        for (int i = 0; i < current_Q.rows(); ++i) {
            // Avoid division by zero or negative values for expected_error_cov_reduction_diag
            double expected_error_cov_reduction_diag = expected_error_cov_reduction(i,i);
            if (expected_error_cov_reduction_diag < 1e-12) expected_error_cov_reduction_diag = 1e-12; // Clamp

            if (sample_cov_error(i,i) > expected_error_cov_reduction_diag * q_adjust_threshold_factor_) { // If observed error variance is higher than threshold
                adjusted_Q(i,i) *= q_increase_factor_; // Increase Q
            } else if (sample_cov_error(i,i) < expected_error_cov_reduction_diag / q_adjust_threshold_factor_) { // If observed error variance is lower than threshold
                adjusted_Q(i,i) *= q_decrease_factor_; // Decrease Q
            }
        }
        // Enforce symmetry after adjustment to guard against floating-point drift
        adjusted_Q = (adjusted_Q + adjusted_Q.transpose()) * 0.5;
        return adjusted_Q;
    }

    /**
     * @brief Estimates and adjusts the measurement noise covariance (R) based on innovation sequence.
     * @param current_R The current measurement noise covariance matrix.
     * @param measurement_dim The dimension of the measurement vector.
     * @return The adjusted R matrix.
     */
    Eigen::MatrixXd estimateR(const Eigen::MatrixXd& current_R, int measurement_dim) const {
        if (innovation_history.size() < window_size_ / 2) return current_R; // Not enough data

        Eigen::MatrixXd sample_cov_innovation = Eigen::MatrixXd::Zero(measurement_dim, measurement_dim);
        int actual_innovation_count = 0; // Count of innovations with correct dimension
        for (const auto& innovation : innovation_history) {
            // Ensure innovation has the correct dimension before multiplying
            if (innovation.size() == measurement_dim) { // Check dimension safety
                sample_cov_innovation += innovation * innovation.transpose();
                actual_innovation_count++;
            }
        }
        // Divide by the count of *valid* innovations, not total history size
        if (actual_innovation_count > 0) {
            sample_cov_innovation /= actual_innovation_count;
        } else {
            return current_R; // No valid innovations to compute sample covariance
        }

        // Simplified heuristic for R adjustment
        // Compare sample_cov_innovation to average theoretical innovation_covariance_history
        Eigen::MatrixXd avg_theoretical_innovation_cov = Eigen::MatrixXd::Zero(measurement_dim, measurement_dim);
        int valid_cov_count = 0;
        for(const auto& cov : innovation_covariance_history) {
            if (cov.rows() == measurement_dim && cov.cols() == measurement_dim) {
                avg_theoretical_innovation_cov += cov;
                valid_cov_count++;
            }
        }
        if (valid_cov_count > 0) {
            avg_theoretical_innovation_cov /= valid_cov_count;
        } else {
            return current_R; // No valid history to compare
        }

        Eigen::MatrixXd adjusted_R = current_R;
        for (int i = 0; i < current_R.rows(); ++i) {
            // Avoid division by zero or negative values for avg_theoretical_innovation_cov_diag
            double avg_theoretical_innovation_cov_diag = avg_theoretical_innovation_cov(i,i);
            if (avg_theoretical_innovation_cov_diag < 1e-12) avg_theoretical_innovation_cov_diag = 1e-12; // Clamp

            if (sample_cov_innovation(i,i) > avg_theoretical_innovation_cov_diag * r_adjust_threshold_factor_) {
                adjusted_R(i,i) *= r_increase_factor_;
            } else if (sample_cov_innovation(i,i) < avg_theoretical_innovation_cov_diag / r_adjust_threshold_factor_) {
                adjusted_R(i,i) *= r_decrease_factor_;
            }
        }
        // Enforce symmetry after adjustment to guard against floating-point drift
        adjusted_R = (adjusted_R + adjusted_R.transpose()) * 0.5;
        return adjusted_R;
    }

    /**
     * @brief Performs a consistency check (NIS test) on the NIS value.
     * @param nis_value The Normalized Innovation Squared value.
     * @param measurement_dim The dimension of the measurement vector (degrees of freedom for chi-squared).
     * @return True if the NIS is within the expected statistical bounds, false otherwise.
     * @note This function uses Boost.Math for a proper chi-squared distribution test.
     */
    bool checkConsistency(double nis_value, int measurement_dim) const {
        if (measurement_dim <= 0) {
            return true; // Cannot perform chi-squared test for non-positive dimensions
        }
        try {
            boost::math::chi_squared_distribution<> chi_sq_dist(measurement_dim);
            double lower_bound = boost::math::quantile(chi_sq_dist, chi_squared_alpha_lower_);
            double upper_bound = boost::math::quantile(chi_sq_dist, chi_squared_alpha_upper_);
            return (nis_value >= lower_bound && nis_value <= upper_bound);
        } catch (const std::exception& e) {
            // Log error if Boost.Math fails (e.g., invalid degrees of freedom)
            // For now, return true to avoid crashing the filter.
            // In a real system, you'd log this: std::cerr << "Chi-squared distribution error: " << e.what() << std::endl;
            return true;
        }
    }
};

#endif // SKYFILTER_ONLINE_NOISE_REESTIMATOR_HPP_
