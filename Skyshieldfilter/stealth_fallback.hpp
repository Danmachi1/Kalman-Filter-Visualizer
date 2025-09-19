// stealth_fallback.hpp
#ifndef SKYFILTER_STEALTH_FALLBACK_HPP_
#define SKYFILTER_STEALTH_FALLBACK_HPP_

#include <vector>
#include <memory>     // For std::shared_ptr
#include <random>     // For std::default_random_engine, std::normal_distribution, std::uniform_real_distribution
#include <cmath>      // For std::sqrt, std::exp, std::lgamma
// #include <numeric>    // For std::accumulate - REMOVED: Not used
#include <stdexcept>  // For std::runtime_error
#include <limits>     // For std::numeric_limits
#include <cassert>    // For assert

#include <Eigen/Dense>
#include <Eigen/Cholesky> // For LLT

// Kalman Library Includes
#include "SquareRootUnscentedKalmanFilter.hpp" // For StateType, CovarianceMatrixType definitions
#include "Kalman/Vector.hpp"
#include "Kalman/Matrix.hpp"
#include "Kalman/CovarianceSquareRoot.hpp"
#include "Kalman/UnscentedKalmanFilterBase.hpp" // For SystemModelBase, MeasurementModelBase

// Project Specific Includes
#include "common_types.hpp"       // For AdaptedMeasurement
#include "motion_models.hpp"      // For Kalman::SystemModelBase
#include "measurement_models.hpp" // For Kalman::MeasurementModelBase

/**
 * @brief The StealthFallback class implements a Particle Filter (PF)
 * for handling stealth mode (sparse data) and divergence recovery.
 * It predicts the target's state using a set of weighted particles.
 *
 * @tparam StateDim The dimension of the state vector.
 * @tparam ControlDim The dimension of the control input vector (usually 0).
 */
template<int StateDim, int ControlDim = 0>
class StealthFallback {
public:
    // Using Kalman::Vector for state and measurement types
    using StateType = Kalman::Vector<double, StateDim>;
    using ControlType = Kalman::Vector<double, ControlDim>;
    using CovarianceMatrixType = Kalman::Matrix<double, StateDim, StateDim>;
    using MeasurementType = Kalman::Vector<double, Eigen::Dynamic>; // Dynamic for measurement models

    // Use constexpr for PI to avoid macro collisions
    static constexpr double kPI = 3.14159265358979323846;

    /**
     * @brief Represents a single particle in the particle filter.
     */
    struct Particle {
        StateType state;
        double weight;
    };

    std::vector<Particle> particles_;
    bool is_active_;         // Flag if this instance is active (either stealth or recovery)
    // is_recovery_mode_ is used to adjust convergence thresholds.
    bool is_recovery_mode_;  // True if this is the tiny embedded recovery bank

    std::default_random_engine generator_;
    std::normal_distribution<double> normal_dist_;         // For sampling from Gaussian
    std::uniform_real_distribution<double> uniform_dist_;  // For resampling

    std::shared_ptr<Kalman::SystemModelBase<StateDim, ControlDim>> system_model_;
    std::shared_ptr<Kalman::MeasurementModelBase<Eigen::Dynamic, StateDim>> measurement_model_; // For re-weighting

    double student_t_degrees_of_freedom_; // For likelihood calculation during re-weighting
    int num_particles_;
    double min_effective_particles_ratio_; // Threshold for resampling (e.g., 0.5)

    /**
     * @brief Constructor for StealthFallback.
     * @param num_particles The number of particles to use in the filter.
     * @param sys_model Shared pointer to the system model for particle propagation.
     * @param meas_model Shared pointer to a generic measurement model for likelihood calculation.
     * @param student_t_dof Degrees of freedom for Student's t-distribution likelihood.
     * @param recovery_mode True if this is the recovery particle bank, false for stealth mode.
     * @param min_eff_particles_ratio Minimum ratio of effective particles to total particles to trigger resampling.
     * @param seed Optional seed for the random number generator. If 0, a non-deterministic seed is used.
     * @throws std::runtime_error if num_particles is non-positive, or sys_model or meas_model are nullptr.
     */
    StealthFallback(int num_particles,
                    std::shared_ptr<Kalman::SystemModelBase<StateDim, ControlDim>> sys_model,
                    std::shared_ptr<Kalman::MeasurementModelBase<Eigen::Dynamic, StateDim>> meas_model,
                    double student_t_dof = 5.0,
                    bool recovery_mode = false,
                    double min_eff_particles_ratio = 0.5,
                    unsigned int seed = 0) // Added optional seed parameter
        : num_particles_(num_particles),
          system_model_(std::move(sys_model)),
          measurement_model_(std::move(meas_model)),
          student_t_degrees_of_freedom_(student_t_dof),
          is_active_(false),
          is_recovery_mode_(recovery_mode),
          normal_dist_(0.0, 1.0),
          uniform_dist_(0.0, 1.0),
          min_effective_particles_ratio_(min_eff_particles_ratio) {

        if (num_particles_ <= 0) {
            throw std::runtime_error("StealthFallback: Number of particles must be positive.");
        }
        if (!system_model_) {
            throw std::runtime_error("StealthFallback: System model cannot be nullptr.");
        }
        if (!measurement_model_) {
            throw std::runtime_error("StealthFallback: Measurement model cannot be nullptr.");
        }
        
        // Resize particles vector. Kalman::Vector (Eigen::Vector) is typically zero-initialized.
        particles_.resize(num_particles_); 

        // Seed the random number generator
        if (seed != 0) {
            generator_.seed(seed); // Use fixed seed for testing/reproducibility
        } else {
            generator_.seed(std::random_device{}()); // Use non-deterministic seed for production
        }
    }

    /**
     * @brief Initializes particles by sampling from a Gaussian distribution.
     * This is typically called when transitioning into stealth or recovery mode.
     * @param mean The mean of the Gaussian distribution (e.g., last SRUKF state).
     * @param covariance The covariance of the Gaussian distribution (e.g., last SRUKF covariance).
     * @throws std::runtime_error if covariance is not positive definite.
     */
    void initialize(const StateType& mean, const CovarianceMatrixType& covariance) {
        // Use Cholesky decomposition of covariance to sample particles
        Eigen::LLT<CovarianceMatrixType> llt_of_cov(covariance);
        if (llt_of_cov.info() != Eigen::Success) {
            throw std::runtime_error("StealthFallback: Input covariance matrix for particle initialization is not positive definite.");
        }
        CovarianceMatrixType L = llt_of_cov.matrixL(); // Lower triangular Cholesky factor

        for (auto& p : particles_) {
            // Generate a random sample from a standard normal distribution (mean 0, variance 1)
            // and transform it using L to get a sample from the desired Gaussian.
            StateType random_sample_std_normal = StateType::NullaryExpr(StateDim, [&](int i){ return normal_dist_(generator_); });
            p.state = mean + L * random_sample_std_normal;
            p.weight = 1.0 / num_particles_; // Initialize with uniform weights
        }
        is_active_ = true;
    }

    /**
     * @brief Performs the prediction step for each particle.
     * Particles are propagated through the system model and process noise is added.
     * @param delta_time The time step for prediction.
     */
    void predict(double delta_time) {
        if (!is_active_) return;

        // Update system model's process noise if it's time-dependent (e.g., Singer model)
        // This relies on SystemModelBase having a virtual updateProcessNoise method,
        // with concrete models implementing it (or a no-op default).
        system_model_->updateProcessNoise(delta_time);
        
        // Get the square root of the process noise covariance (S_Q)
        // Kalman::CovarianceSquareRoot::getMatrix() is assumed to return the lower triangular Cholesky factor.
        CovarianceMatrixType L_Q = system_model_->getCovarianceSquareRoot().getMatrix();

        ControlType u_zero = ControlType::Zero(); // Control input is 0-dimensional for current models

        for (auto& p : particles_) {
            // Propagate particle state through the system model
            p.state = system_model_->f(p.state, u_zero, delta_time);

            // Add process noise by sampling from Q
            StateType process_noise_sample = StateType::NullaryExpr(StateDim, [&](int i){ return normal_dist_(generator_); });
            p.state += L_Q * process_noise_sample;
        }
    }

    /**
     * @brief Performs the update (re-weighting and resampling) step for particles.
     * This is called when a new measurement becomes available.
     * @param adapted_measurement The incoming adapted measurement.
     */
    void update(const AdaptedMeasurement& adapted_measurement) {
        if (!is_active_) return;

        double sum_weights = 0.0;
        
        // This conversion to full R and then LLT decomposition is done once outside the particle loop for efficiency.
        CovarianceMatrixType R_matrix = adapted_measurement.R_covariance_sqrt.toCovarianceMatrix();
        
        double R_determinant = 0.0;
        Eigen::LLT<Eigen::MatrixXd> llt_R; // Declare outside if-block to use its solve method later

        // Robustly check for positive definiteness and compute determinant once using Cholesky
        if (R_matrix.rows() == R_matrix.cols() && R_matrix.rows() > 0) { // Check for square and non-empty matrix
            llt_R.compute(R_matrix); // Compute Cholesky decomposition
            if (llt_R.info() == Eigen::Success) {
                const auto& L_R = llt_R.matrixL();
                double log_det_R = 2.0 * L_R.diagonal().array().log().sum();
                R_determinant = std::exp(log_det_R);
            }
        }
        
        // Set the full R matrix to the measurement model for its internal use (e.g., h() or other methods)
        // This is done after computing its properties for the current update cycle.
        measurement_model_->setFullMeasurementNoiseCovariance(R_matrix);


        for (auto& p : particles_) {
            // Calculate predicted measurement from particle state
            MeasurementType predicted_measurement = measurement_model_->h(p.state);

            // Assert that measurement dimensions match
            assert(adapted_measurement.measurement_vector.size() == predicted_measurement.size() &&
                   "Measurement dimension mismatch between adapted_measurement and predicted_measurement.");

            // Calculate innovation
            MeasurementType innovation = adapted_measurement.measurement_vector - predicted_measurement;

            // Calculate NIS for this particle
            double nis_particle = std::numeric_limits<double>::max();
            if (R_matrix.rows() > 0 && R_determinant > 1e-18) {
                // Use Cholesky factor for solving, more stable than inverse()
                nis_particle = innovation.transpose() * llt_R.solve(innovation);
            }

            // Calculate likelihood using Student's t-distribution
            double likelihood = calculateStudentTLikelihood(
                nis_particle,
                adapted_measurement.measurement_vector.size(),
                student_t_degrees_of_freedom_,
                R_determinant
            );

            p.weight *= likelihood;
            sum_weights += p.weight;
        }

        // Normalize weights
        if (sum_weights < 1e-30) { // All weights are very low, likely divergence or bad measurement
            // Reset weights uniformly to prevent collapse, or signal error
            for (auto& p : particles_) p.weight = 1.0 / num_particles_;
            // Optionally, throw an error or log a warning
            // throw std::runtime_error("StealthFallback: All particle weights are extremely low, likely divergence.");
        } else {
            for (auto& p : particles_) p.weight /= sum_weights;
        }

        // Resample if effective number of particles is too low
        if (calculateEffectiveParticles() < num_particles_ * min_effective_particles_ratio_) {
            resampleParticles();
        }
    }

    /**
     * @brief Checks if the particle distribution is sufficiently "tight" for SRUKF handover.
     * This can be based on the estimated covariance determinant or effective number of particles.
     * @return True if the filter has converged enough to hand over to SRUKF, false otherwise.
     */
    bool hasConvergedForSRUKFHandover() const {
        if (!is_active_) return false;

        // Example convergence criteria:
        // 1. Particle covariance determinant is below a threshold
        // 2. Effective number of particles is high
        
        CovarianceMatrixType estimated_cov = getEstimatedCovariance();
        // Note: for very high dimensions or extreme values, consider working with log-determinant
        // for numerical stability to avoid underflow/overflow of the determinant itself.
        double cov_determinant = estimated_cov.determinant(); 

        // Thresholds need careful tuning based on application and state space
        const double CONVERGENCE_COV_DET_THRESHOLD = is_recovery_mode_ ? 1e-3 : 1e-5; // More lenient for recovery
        const double CONVERGENCE_NEFF_RATIO = 0.8; // 80% of particles are effective

        return (cov_determinant < CONVERGENCE_COV_DET_THRESHOLD &&
                calculateEffectiveParticles() >= num_particles_ * CONVERGENCE_NEFF_RATIO);
    }

    /**
     * @brief Returns the weighted mean state estimate from the particles.
     * @return The estimated state vector.
     */
    StateType getEstimatedState() const {
        if (!is_active_ || particles_.empty()) {
            return StateType::Zero(); // Or throw, depending on desired behavior
        }
        StateType mean = StateType::Zero();
        for (const auto& p : particles_) {
            mean += p.weight * p.state;
        }
        return mean;
    }

    /**
     * @brief Returns the weighted covariance matrix estimate from the particles.
     * @return The estimated covariance matrix.
     */
    CovarianceMatrixType getEstimatedCovariance() const {
        if (!is_active_ || particles_.empty()) {
            return CovarianceMatrixType::Identity(); // Or throw
        }
        CovarianceMatrixType cov = CovarianceMatrixType::Zero();
        StateType mean = getEstimatedState();
        for (const auto& p : particles_) {
            StateType diff = p.state - mean;
            cov += p.weight * diff * diff.transpose();
        }
        return cov;
    }

    /**
     * @brief Deactivates the particle filter instance.
     * This can be called when handing over control back to the IMM.
     */
    void deactivate() {
        is_active_ = false;
        // Optionally reset particles or clear them to save memory
    }

private:
    /**
     * @brief Resamples particles using systematic resampling to mitigate particle degeneracy.
     */
    void resampleParticles() {
        std::vector<Particle> new_particles(num_particles_);
        std::vector<double> cumulative_weights(num_particles_);

        // Compute cumulative weights
        cumulative_weights[0] = particles_[0].weight;
        for (int i = 1; i < num_particles_; ++i) {
            cumulative_weights[i] = cumulative_weights[i-1] + particles_[i].weight;
        }

        // Generate starting point for systematic resampling
        double r = uniform_dist_(generator_) / num_particles_;

        int current_particle_idx = 0;
        for (int i = 0; i < num_particles_; ++i) {
            double u = r + (double)i / num_particles_;
            // Ensure current_particle_idx does not go out of bounds
            while (current_particle_idx < num_particles_ - 1 && u > cumulative_weights[current_particle_idx]) {
                current_particle_idx++;
            }
            new_particles[i] = particles_[current_particle_idx];
            new_particles[i].weight = 1.0 / num_particles_; // Reset weights to uniform
        }
        particles_ = new_particles;
    }

    /**
     * @brief Calculates the effective number of particles (N_eff).
     * A low N_eff indicates particle degeneracy.
     * @return The effective number of particles.
     */
    double calculateEffectiveParticles() const {
        double sum_of_squared_weights = 0.0;
        for (const auto& p : particles_) {
            sum_of_squared_weights += p.weight * p.weight;
        }
        if (sum_of_squared_weights < 1e-30) { // Avoid division by zero
            return 0.0;
        }
        return 1.0 / sum_of_squared_weights;
    }

    /**
     * @brief Calculates the likelihood of a measurement using the Student's t-distribution PDF.
     * @param nis_value The Normalized Innovation Squared (NIS) value.
     * @param measurement_dim The dimension of the measurement.
     * @param degrees_of_freedom The degrees of freedom for the Student's t-distribution.
     * @param innovation_covariance_determinant The determinant of the innovation covariance matrix (P_y).
     * @return The likelihood value.
     */
    double calculateStudentTLikelihood(double nis_value, int measurement_dim, double degrees_of_freedom, double innovation_covariance_determinant) const {
        if (measurement_dim <= 0 || degrees_of_freedom <= 0 || innovation_covariance_determinant < 1e-18) {
            return 1e-30; // Return a very small likelihood for invalid inputs
        }

        double t_pdf_term = 1.0 + (1.0 / degrees_of_freedom) * nis_value;
        
        // Using std::lgamma for log-gamma to avoid overflow with large arguments to std::tgamma
        double log_gamma_numerator = std::lgamma((degrees_of_freedom + measurement_dim) / 2.0);
        double log_gamma_denominator = std::lgamma(degrees_of_freedom / 2.0);

        double log_constant_part = log_gamma_numerator - log_gamma_denominator -
                                   0.5 * measurement_dim * std::log(degrees_of_freedom * kPI) -
                                   0.5 * std::log(innovation_covariance_determinant);

        double log_power_term = -(degrees_of_freedom + measurement_dim) / 2.0 * std::log(t_pdf_term);

        return std::exp(log_constant_part + log_power_term);
    }
};

#endif // SKYFILTER_STEALTH_FALLBACK_HPP_
