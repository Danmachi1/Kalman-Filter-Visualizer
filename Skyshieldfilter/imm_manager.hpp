// imm_manager.hpp
#ifndef SKYFILTER_IMM_MANAGER_HPP_
#define SKYFILTER_IMM_MANAGER_HPP_

#include <vector>
#include <memory> // For std::shared_ptr, std::dynamic_pointer_cast
// #include <numeric> // For std::accumulate - REMOVED: Not used
#include <map>     // For map of measurement models
#include <cmath>   // For std::exp, std::sqrt, std::lgamma
#include <limits>  // For std::numeric_limits
#include <stdexcept> // For std::runtime_error

#include <Eigen/Dense>
#include <Eigen/Cholesky> // For LLT

// Kalman Library Includes
#include "SquareRootUnscentedKalmanFilter.hpp"
#include "Kalman/Vector.hpp"
#include "Kalman/Matrix.hpp"
#include "Kalman/CovarianceSquareRoot.hpp"
#include "Kalman/UnscentedKalmanFilterBase.hpp" // For predictMeasurement, computeInnovationCovariance

// Project Specific Includes
#include "common_types.hpp"
#include "motion_models.hpp"      // For Kalman::SystemModelBase and concrete models
#include "measurement_models.hpp" // For Kalman::MeasurementModelBase and concrete models
#include "online_noise_reestimator.hpp" // For OnlineNoiseReEstimator

/**
 * @brief Wrapper for each SRUKF instance and its associated models.
 * This holds the SRUKF, its system model, and a collection of measurement models
 * that it can use for updates.
 *
 * @tparam StateDim The dimension of the state vector (e.g., 6 for CV, 7 for CT, 9 for CA/Singer).
 * @tparam ControlDim The dimension of the control input vector (0 for most models here).
 */
template<int StateDim, int ControlDim>
class ModelInstanceWrapper {
public:
    MotionModelType type;
    Kalman::SquareRootUnscentedKalmanFilter<Kalman::Vector<double, StateDim>> srukf;

    // The actual system model (CV, CT, CA, Singer)
    std::shared_ptr<Kalman::SystemModelBase<StateDim, ControlDim>> system_model;

    // Map of measurement models for different sensor types.
    // We use Eigen::Dynamic for MeasurementDim in the base class pointer,
    // assuming concrete models will have fixed dimensions.
    std::map<SensorType, std::shared_ptr<Kalman::MeasurementModelBase<Eigen::Dynamic, StateDim>>> measurement_models;

    // Process noise square roots for low and high maneuver states
    Kalman::CovarianceSquareRoot<Kalman::Vector<double, StateDim>> Q_lowManeuver_sqrt;
    Kalman::CovarianceSquareRoot<Kalman::Vector<double, StateDim>> Q_highManeuver_sqrt;
    
    // Flag to indicate if this model is currently in a "high maneuver" state
    bool is_maneuvering_state;

    /**
     * @brief Constructor for ModelInstanceWrapper.
     * @param model_type The type of motion model (CV, CT, CA, Singer).
     * @param initial_state The initial state vector for the SRUKF.
     * @param initial_covariance The initial covariance matrix for the SRUKF.
     * @param sys_model A shared pointer to the concrete system model.
     * @param meas_models A map of shared pointers to concrete measurement models.
     * @param q_low_maneuver_sqrt Initial Q_lowManeuver_sqrt.
     * @param q_high_maneuver_sqrt Initial Q_highManeuver_sqrt.
     */
    ModelInstanceWrapper(MotionModelType model_type,
                         const Kalman::Vector<double, StateDim>& initial_state,
                         const Kalman::Matrix<double, StateDim, StateDim>& initial_covariance,
                         std::shared_ptr<Kalman::SystemModelBase<StateDim, ControlDim>> sys_model,
                         const std::map<SensorType, std::shared_ptr<Kalman::MeasurementModelBase<Eigen::Dynamic, StateDim>>>& meas_models,
                         const Kalman::CovarianceSquareRoot<Kalman::Vector<double, StateDim>>& q_low_maneuver_sqrt,
                         const Kalman::CovarianceSquareRoot<Kalman::Vector<double, StateDim>>& q_high_maneuver_sqrt)
        : type(model_type), system_model(std::move(sys_model)), measurement_models(meas_models),
          Q_lowManeuver_sqrt(q_low_maneuver_sqrt), Q_highManeuver_sqrt(q_high_maneuver_sqrt),
          is_maneuvering_state(false) { // Start in non-maneuvering state
        
        // Initialize system model with the low maneuver Q initially
        // This must happen BEFORE srukf.init() so the filter's initial internal Q is correct.
        if (system_model) { // Ensure system_model is not nullptr
            system_model->setFullProcessNoiseCovariance(Q_lowManeuver_sqrt.toCovarianceMatrix());
        }
        srukf.init(initial_state, initial_covariance);
    }

    /**
     * @brief Default constructor for ModelInstanceWrapper.
     * Initializes members to prevent undefined behavior if an instance is default-constructed
     * (e.g., when resizing a vector).
     */
    ModelInstanceWrapper()
        : type(MotionModelType::UNKNOWN), // Initialize type to a safe default
          srukf(Kalman::Vector<double, StateDim>::Zero(), Kalman::Matrix<double, StateDim, StateDim>::Identity()), // Initialize SRUKF with safe defaults
          system_model(nullptr), // Initialize shared_ptr to nullptr
          is_maneuvering_state(false) {
        // measurement_models, Q_lowManeuver_sqrt, Q_highManeuver_sqrt
        // will be default constructed (empty map, zero matrices for CovarianceSquareRoot).
    }

    /**
     * @brief Sets the maneuvering state for this model.
     * @param is_maneuvering True if the model should use high maneuver Q, false for low maneuver Q.
     */
    void setManeuveringState(bool maneuvering) {
        if (is_maneuvering_state != maneuvering) {
            is_maneuvering_state = maneuvering;
            if (system_model) { // Guard against nullptr dereference
                if (is_maneuvering_state) {
                    system_model->setFullProcessNoiseCovariance(Q_highManeuver_sqrt.toCovarianceMatrix());
                } else {
                    system_model->setFullProcessNoiseCovariance(Q_lowManeuver_sqrt.toCovarianceMatrix());
                }
            } else {
                // Log an error or throw if system_model is expected to be valid here
                // std::cerr << "Warning: system_model is nullptr in setManeuveringState." << std::endl;
            }
        }
    }
};


/**
 * @brief The IMMManager class is responsible for running multiple filters in parallel,
 * blending their estimates, and updating their probabilities based on how well each model
 * predicts the incoming measurements.
 *
 * @tparam StateDim The maximum dimension of the state vector across all models.
 * @tparam ControlDim The dimension of the control input vector (usually 0).
 */
template<int StateDim, int ControlDim = 0>
class IMMManager {
public:
    // Using Kalman::Vector for state and measurement types
    using StateType = Kalman::Vector<double, StateDim>;
    using ControlType = Kalman::Vector<double, ControlDim>;
    using CovarianceMatrixType = Kalman::Matrix<double, StateDim, StateDim>;

    // Use constexpr for PI to avoid macro collisions
    static constexpr double kPI = 3.14159265358979323846;

    std::vector<ModelInstanceWrapper<StateDim, ControlDim>> models_;
    Eigen::VectorXd model_probabilities_; // mu_k_j
    Eigen::MatrixXd transition_matrix_;   // M_ij = P(M_k=j | M_k-1=i)

    // Overall combined state and covariance
    StateType combined_state_;
    CovarianceMatrixType combined_covariance_;

    // JPDA-related parameters
    double clutter_density_; // Lambda_clutter: Expected number of clutter measurements per unit volume
    /**
     * @brief Chi-squared threshold for validation gating (e.g., 99.7% confidence).
     * This value should correspond to the desired confidence level for a chi-squared
     * distribution with degrees of freedom equal to the measurement dimension.
     * For example, for 3D measurements (3 DOF) at 99.7% confidence, a common value is 9.21.
     */
    double validation_gate_chi2_threshold_; 

    // Student's t-distribution degrees of freedom for likelihood calculation
    double student_t_degrees_of_freedom_;

    // Maneuver detection threshold (NIS)
    double maneuver_nis_threshold_;
    int maneuver_nis_window_size_;
    std::vector<int> consistent_high_nis_count_; // Counter for consecutive high NIS for each model

    /**
     * @brief Constructor for the IMMManager.
     * @param initial_models A vector of ModelInstanceWrapper objects, each pre-configured.
     * @param initial_transition_matrix The initial transition probability matrix.
     * @param clutter_density Expected number of clutter measurements per unit volume.
     * @param validation_gate_chi2_threshold Chi-squared threshold for validation gating (e.9., 99.7% confidence).
     * @param student_t_dof Degrees of freedom for Student's t-distribution likelihood.
     * @param maneuver_nis_threshold NIS threshold for detecting maneuver.
     * @param maneuver_nis_window_size Number of consecutive high NIS values to trigger maneuver state.
     * @throws std::runtime_error if initial_models is empty or transition_matrix dimensions are incorrect.
     */
    IMMManager(std::vector<ModelInstanceWrapper<StateDim, ControlDim>> initial_models,
               const Eigen::MatrixXd& initial_transition_matrix,
               double clutter_density = 1e-4, // Default value for clutter density
               double validation_gate_chi2_threshold = 9.21, // Chi2 for 3 DOF, 99.7% (approx)
               double student_t_dof = 5.0, // Default degrees of freedom
               double maneuver_nis_threshold = 10.0, // Example threshold, needs tuning
               int maneuver_nis_window_size = 3) // Example window, needs tuning
        : models_(std::move(initial_models)),
          transition_matrix_(initial_transition_matrix),
          clutter_density_(clutter_density),
          validation_gate_chi2_threshold_(validation_gate_chi2_threshold),
          student_t_degrees_of_freedom_(student_t_dof),
          maneuver_nis_threshold_(maneuver_nis_threshold),
          maneuver_nis_window_size_(maneuver_nis_window_size) {

        if (models_.empty()) {
            throw std::runtime_error("IMMManager: Initial models list cannot be empty.");
        }
        if (transition_matrix_.rows() != models_.size() || transition_matrix_.cols() != models_.size()) {
            throw std::runtime_error("IMMManager: Transition matrix dimensions must match the number of models.");
        }

        model_probabilities_ = Eigen::VectorXd::Constant(models_.size(), 1.0 / models_.size()); // Initialize uniformly
        combined_state_.setZero();
        combined_covariance_.setIdentity();
        consistent_high_nis_count_.resize(models_.size(), 0); // Initialize NIS counters
    }

    /**
     * @brief Performs the IMM prediction step: mixing and model-conditional prediction.
     * Also applies adaptive Q adjustments based on feedback from the noise re-estimator.
     * @param delta_time The time step for prediction.
     * @param env_context Environmental context for adaptive motion models.
     * @param noise_re_estimator The online noise re-estimator (for Q adjustment feedback).
     */
    void predict(double delta_time, const EnvironmentalContext& env_context, OnlineNoiseReEstimator& noise_re_estimator) {
        // 1. Calculate normalization constants for mixed probabilities (bar_c_j)
        Eigen::VectorXd c_bar_j(models_.size()); // Sum_i (M_ij * mu_i_k-1)
        for (int j = 0; j < models_.size(); ++j) { // Target model j
            c_bar_j(j) = 0.0;
            for (int i = 0; i < models_.size(); ++i) { // From model i
                c_bar_j(j) += transition_matrix_(i, j) * model_probabilities_(i);
            }
        }

        // 2. Calculate mixed probabilities (mu_k-1_given_j) and perform mixing
        std::vector<Eigen::VectorXd> mu_k_minus_1_given_j(models_.size()); // mu_i_given_j
        for (int j = 0; j < models_.size(); ++j) { // Target model j
            if (c_bar_j(j) < 1e-12) { // Avoid division by zero for very low probabilities
                mu_k_minus_1_given_j[j] = Eigen::VectorXd::Zero(models_.size());
                // If a model has zero incoming probability, it won't be mixed.
                // Re-initialize its SRUKF with a default or last known state to prevent issues.
                models_[j].srukf.init(combined_state_, combined_covariance_); // Use combined as fallback
                continue;
            }
            mu_k_minus_1_given_j[j].resize(models_.size());
            for (int i = 0; i < models_.size(); ++i) { // From model i
                mu_k_minus_1_given_j[j](i) = (transition_matrix_(i, j) * model_probabilities_(i)) / c_bar_j(j);
            }

            // Compute mixed state x_0j and covariance P_0j for model j
            StateType x_0j = StateType::Zero();
            CovarianceMatrixType P_0j = CovarianceMatrixType::Zero();

            for (int i = 0; i < models_.size(); ++i) {
                x_0j += models_[i].srukf.getState() * mu_k_minus_1_given_j[j](i);
            }

            for (int i = 0; i < models_.size(); ++i) {
                StateType diff = models_[i].srukf.getState() - x_0j;
                P_0j += (models_[i].srukf.getCovariance() + diff * diff.transpose()) * mu_k_minus_1_given_j[j](i);
            }

            // Initialize SRUKF for model j with mixed state and covariance
            models_[j].srukf.init(x_0j, P_0j);
        }

        // 3. Model-Conditional Prediction for each SRUKF
        for (auto& model_wrapper : models_) {
            // Apply Environment-Adaptive Motion Model logic:
            // Use virtual method to update process noise if the model supports it (e.g., Singer)
            if (model_wrapper.system_model) { // Guard against nullptr
                model_wrapper.system_model->updateProcessNoise(delta_time);
            }
            
            // Perform SRUKF prediction
            // Control input 'u' is 0-dimensional for the current motion models.
            ControlType u_zero = ControlType::Zero();
            model_wrapper.srukf.predict(*model_wrapper.system_model, u_zero);
        }

        // 4. Apply Adaptive Q: Adjust Q for each model based on prediction error drift feedback
        // This is done after prediction, but the adjusted Q will be used for the *next* prediction cycle.
        for (auto& model_wrapper : models_) {
            // Get the current Q matrix from the system model (convert S_Q to Q)
            // This `current_Q` will be the one chosen by `setManeuveringState` or the continuously adapted one.
            CovarianceMatrixType current_Q = model_wrapper.system_model->getCovarianceSquareRoot().toCovarianceMatrix();
            
            // Estimate adjusted Q using the noise re-estimator
            // The estimateQ method refines the Q based on historical performance.
            // P_k|k-1 is model_wrapper.srukf.getPredictedCovariance() (covariance before update in current cycle)
            // P_k|k is model_wrapper.srukf.getCovariance() (covariance after update in previous cycle)
            CovarianceMatrixType adjusted_Q = noise_re_estimator.estimateQ(
                current_Q,
                model_wrapper.srukf.getPredictedCovariance(),
                model_wrapper.srukf.getCovariance()
            );
            
            // Apply the adjusted Q back to the system model for the *next* prediction cycle
            model_wrapper.system_model->setFullProcessNoiseCovariance(adjusted_Q);
        }
    }

    /**
     * @brief Performs the IMM update step: model-conditional update, probability update, and combination.
     * Incorporates JPDA-like logic to handle multiple gated measurements and clutter.
     * Also applies adaptive R adjustments based on feedback from the noise re-estimator.
     * Includes Q-matrix switching based on maneuver detection (NIS).
     *
     * @param gated_measurements A vector of AdaptedMeasurement objects that fall within the validation gate.
     * @param noise_re_estimator The online noise re-estimator.
     * @return True if at least one model was updated, false otherwise.
     * @throws std::runtime_error if a measurement model for a sensor type is not found for a model.
     */
    bool update(const std::vector<AdaptedMeasurement>& gated_measurements, OnlineNoiseReEstimator& noise_re_estimator) {
        Eigen::VectorXd likelihoods(models_.size()); // Lambda_k_j
        bool any_model_updated = false;

        // 1. Model-Conditional Update for each SRUKF
        for (int j = 0; j < models_.size(); ++j) {
            auto& model_wrapper = models_[j];
            
            double current_model_likelihood_sum = 0.0; // Sum of beta_k * L_k
            double prob_of_no_detection_product = 1.0; // Product of (1 - beta_k)
            
            int best_measurement_idx = -1;
            double max_beta = -1.0;
            
            // Variables to pass to noise re-estimator and consistency check from the "best" update
            Eigen::VectorXd best_innovation_for_reestimator;
            Eigen::MatrixXd best_innovation_covariance_for_reestimator;
            double best_nis_for_reestimator = 0.0;
            int best_measurement_dim_for_reestimator = 0;

            // Flag to track if any measurement was considered for this model
            bool measurement_considered_for_model = false;

            // Iterate through all gated measurements to calculate association probabilities and individual likelihoods
            for (size_t k = 0; k < gated_measurements.size(); ++k) {
                const auto& current_gated_measurement = gated_measurements[k];

                auto it = model_wrapper.measurement_models.find(current_gated_measurement.sensor_type);
                if (it == model_wrapper.measurement_models.end()) {
                    // This model instance does not support this sensor type. Skip this measurement for this model.
                    continue;
                }
                std::shared_ptr<Kalman::MeasurementModelBase<Eigen::Dynamic, StateDim>> measurement_model_k = it->second;
                measurement_considered_for_model = true;

                // Set the adapted S_R from the MeasurementAdapter to the measurement model
                // This is the initial R from the MeasurementAdapter, possibly adapted by sensor context.
                measurement_model_k->setMeasurementNoiseCovarianceSqrt(current_gated_measurement.R_covariance_sqrt);

                // Get predicted measurement and its covariance for this specific measurement and model
                Kalman::Vector<double, Eigen::Dynamic> predicted_y_k;
                Kalman::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> P_y_k;
                
                // Use the SRUKF's internal methods to get predicted measurement and innovation covariance
                predicted_y_k = model_wrapper.srukf.predictMeasurement(*measurement_model_k);
                P_y_k = model_wrapper.srukf.computeInnovationCovariance(*measurement_model_k);

                // Calculate innovation
                Kalman::Vector<double, Eigen::Dynamic> innovation_k = current_gated_measurement.measurement_vector - predicted_y_k;
                
                // Calculate NIS
                double nis_k = std::numeric_limits<double>::max(); // Default to max if covariance is invalid
                
                // Robustly check for positive definiteness and compute determinant once
                double P_y_k_determinant = 0.0;
                if (P_y_k.rows() > 0) { // Check dimensions
                    Eigen::LLT<Eigen::MatrixXd> llt_P_y_k(P_y_k);
                    if (llt_P_y_k.info() == Eigen::Success) {
                        // Compute determinant from Cholesky factor for numerical stability
                        const auto& L = llt_P_y_k.matrixL();
                        double log_det = 2.0 * L.diagonal().array().log().sum();
                        P_y_k_determinant = std::exp(log_det);

                        if (P_y_k_determinant > 1e-18) { // Check if determinant is large enough
                            nis_k = innovation_k.transpose() * llt_P_y_k.solve(innovation_k); // Use Cholesky factor for solve
                        }
                    }
                }

                // --- Validation Gating ---
                // If NIS is above the threshold, this measurement is considered an outlier for this model.
                // Skip it for the current model's update and likelihood calculation.
                if (nis_k > validation_gate_chi2_threshold_) {
                    continue; // Skip this measurement for this model
                }

                // Calculate likelihood using Student's t-distribution (passing P_y_k for determinant)
                double L_k = calculateStudentTLikelihood(nis_k, current_gated_measurement.measurement_vector.size(), student_t_degrees_of_freedom_, P_y_k_determinant);

                // Calculate Validation Gate Volume (for clutter probability)
                double V_G_k = 0.0;
                if (P_y_k.rows() > 0 && P_y_k_determinant > 1e-18) {
                    V_G_k = std::sqrt(std::pow(2 * kPI, current_gated_measurement.measurement_vector.size()) * P_y_k_determinant);
                } else {
                    V_G_k = 1.0; // Default to a small volume if covariance is degenerate
                }

                // Calculate probability of measurement being clutter
                double P_clutter_k = clutter_density_ / V_G_k;

                // Calculate association probability beta_k
                double beta_k = L_k / (L_k + P_clutter_k);
                
                // Accumulate terms for JPDA likelihood for this model
                current_model_likelihood_sum += beta_k * L_k;
                prob_of_no_detection_product *= (1.0 - beta_k);

                // Track the "best" measurement for the actual SRUKF update and noise re-estimator
                // NOTE: The SRUKF update method does not directly support weighted innovations.
                // Therefore, we use the single "best" associated measurement for the SRUKF update,
                // while the IMM model probability calculation uses the full JPDA likelihood.
                if (beta_k > max_beta) {
                    max_beta = beta_k;
                    best_measurement_idx = k;
                    best_innovation_for_reestimator = innovation_k;
                    best_innovation_covariance_for_reestimator = P_y_k;
                    best_nis_for_reestimator = nis_k;
                    best_measurement_dim_for_reestimator = current_gated_measurement.measurement_vector.size();
                }
            } // End of loop over gated_measurements

            // Final model likelihood for IMM: sum of weighted likelihoods + probability of no detection (clutter)
            likelihoods(j) = current_model_likelihood_sum + prob_of_no_detection_product * clutter_density_;

            if (std::isnan(likelihoods(j)) || likelihoods(j) < 1e-30) {
                likelihoods(j) = 1e-30; // Clamp very small or NaN likelihoods
            }

            // --- Q-Matrix Switching based on Maneuver Detection (NIS) ---
            // This logic is applied *after* calculating NIS for the best measurement.
            if (measurement_considered_for_model && best_measurement_idx != -1) {
                if (best_nis_for_reestimator > maneuver_nis_threshold_) {
                    consistent_high_nis_count_[j]++;
                    if (consistent_high_nis_count_[j] >= maneuver_nis_window_size_) {
                        model_wrapper.setManeuveringState(true); // Switch to high maneuver Q
                    }
                } else {
                    consistent_high_nis_count_[j] = 0; // Reset counter
                    model_wrapper.setManeuveringState(false); // Switch to low maneuver Q
                }
            } else {
                 // If no measurement was considered or no best measurement, assume non-maneuvering
                 consistent_high_nis_count_[j] = 0;
                 model_wrapper.setManeuveringState(false);
            }


            // Perform SRUKF update for model j using the "best" associated measurement
            // This is a practical compromise due to the SRUKF API.
            if (best_measurement_idx != -1) {
                const auto& best_measurement = gated_measurements[best_measurement_idx];
                auto best_measurement_model = model_wrapper.measurement_models.at(best_measurement.sensor_type);
                best_measurement_model->setMeasurementNoiseCovarianceSqrt(best_measurement.R_covariance_sqrt); // Ensure R is set for update

                model_wrapper.srukf.update(*best_measurement_model, best_measurement.measurement_vector);
                any_model_updated = true;

                // Add prediction error to noise re-estimator
                noise_re_estimator.addPredictionError(model_wrapper.srukf.getState() - model_wrapper.srukf.getPredictedState());

                // Add innovation (from the best measurement) to noise re-estimator
                noise_re_estimator.addInnovation(best_innovation_for_reestimator, best_innovation_covariance_for_reestimator);

                // Perform Covariance Consistency Checks (χ² tests)
                if (!noise_re_estimator.checkConsistency(best_nis_for_reestimator, best_measurement_dim_for_reestimator)) {
                    // This indicates an inconsistency. The noise_re_estimator will internally
                    // adjust Q/R if configured. SkyFilterDetectX might trigger recovery here.
                }

                // Apply Adaptive R: Adjust R for this measurement model based on innovation sequence feedback
                // This will affect the R used for *future* updates for this specific measurement model.
                // Get the current R matrix from the measurement model (convert S_R to R)
                // NOTE: Converting from S_R to full R matrix can be computationally intensive.
                // If OnlineNoiseReEstimator could work directly with S_R, it would be more efficient.
                CovarianceMatrixType current_R = best_measurement_model->getCovarianceSquareRoot().toCovarianceMatrix();
                
                // Estimate adjusted R using the noise re-estimator
                CovarianceMatrixType adjusted_R = noise_re_estimator.estimateR(
                    current_R,
                    best_measurement_dim_for_reestimator
                );
                
                // Apply the adjusted R back to the measurement model for the *next* update cycle
                best_measurement_model->setFullMeasurementNoiseCovariance(adjusted_R);

            } else {
                // If no measurements associated, the SRUKF update is skipped for this model.
                // Its state and covariance remain at their predicted values.
                // The likelihood calculation above still allows it to contribute to IMM probabilities.
            }
        } // End of loop over models

        // 2. Update IMM model probabilities
        Eigen::VectorXd new_model_probabilities(models_.size());
        double sum_new_probabilities = 0.0;

        for (int j = 0; j < models_.size(); ++j) {
            // Recalculate c_bar_j from predict step (or reuse if stored)
            double c_bar_j_recalc = 0.0;
            for (int i = 0; i < models_.size(); ++i) {
                c_bar_j_recalc += transition_matrix_(i, j) * model_probabilities_(i);
            }
            new_model_probabilities(j) = likelihoods(j) * c_bar_j_recalc;
            sum_new_probabilities += new_model_probabilities(j);
        }

        if (sum_new_probabilities < 1e-30) { // All models have very low likelihood
            // This indicates a severe mismatch. Reset probabilities or handle divergence.
            // For now, re-uniform and throw.
            model_probabilities_ = Eigen::VectorXd::Constant(models_.size(), 1.0 / models_.size());
            throw std::runtime_error("IMMManager: All model likelihoods are extremely low, likely divergence.");
        } else {
            model_probabilities_ = new_model_probabilities / sum_new_probabilities; // Normalize
        }

        // 3. Combine individual SRUKF states/covariances into overall track estimate
        combined_state_.setZero();
        combined_covariance_.setZero();

        for (int j = 0; j < models_.size(); ++j) {
            combined_state_ += models_[j].srukf.getState() * model_probabilities_(j);
        }

        for (int j = 0; j < models_.size(); ++j) {
            StateType diff = models_[j].srukf.getState() - combined_state_;
            combined_covariance_ += (models_[j].srukf.getCovariance() + diff * diff.transpose()) * model_probabilities_(j);
        }
        return any_model_updated;
    }

    /**
     * @brief Adjusts the IMM transition matrix based on environmental context.
     * @param context The environmental context.
     *
     * @note This is a heuristic example. Real-world adaptation logic would be
     * more sophisticated and likely derived from extensive data analysis.
     * The goal is to bias model transitions towards behaviors expected in
     * specific environments (e.g., more turns in urban canyons).
     */
    void adaptTransitionMatrix(const EnvironmentalContext& context) {
        // Get indices for common motion models
        int cv_idx = -1, ct_idx = -1, ca_idx = -1, singer_idx = -1;
        for (size_t i = 0; i < models_.size(); ++i) {
            if (models_[i].type == MotionModelType::CONSTANT_VELOCITY) cv_idx = i;
            else if (models_[i].type == MotionModelType::CONSTANT_TURN) ct_idx = i;
            else if (models_[i].type == MotionModelType::CONSTANT_ACCELERATION) ca_idx = i;
            else if (models_[i].type == MotionModelType::SINGER) singer_idx = i;
        }

        // Apply adaptation based on environmental context
        if (context.building_density > 0.7) { // Dense urban environment
            // Increase probability of staying in CT or transitioning to CT
            if (ct_idx >= 0 && static_cast<size_t>(ct_idx) < models_.size()) { // Robust index check
                transition_matrix_(ct_idx, ct_idx) = std::min(0.95, transition_matrix_(ct_idx, ct_idx) * 1.1); // Increase CT->CT
            }
            if (cv_idx >= 0 && static_cast<size_t>(cv_idx) < models_.size() && ct_idx >= 0 && static_cast<size_t>(ct_idx) < models_.size()) { // Robust index check
                transition_matrix_(cv_idx, ct_idx) = std::min(0.2, transition_matrix_(cv_idx, ct_idx) * 1.2); // Increase CV->CT
            }
            // Decrease probability of staying in CV if a CT model exists
            if (cv_idx >= 0 && static_cast<size_t>(cv_idx) < models_.size() && ct_idx >= 0 && static_cast<size_t>(ct_idx) < models_.size()) { // Robust index check
                transition_matrix_(cv_idx, cv_idx) = std::max(0.05, transition_matrix_(cv_idx, cv_idx) * 0.9);
            }
        } else if (context.nearest_obstacle_distance > 100.0) { // Open area
            // Increase probability of staying in CV
            if (cv_idx >= 0 && static_cast<size_t>(cv_idx) < models_.size()) { // Robust index check
                transition_matrix_(cv_idx, cv_idx) = std::min(0.95, transition_matrix_(cv_idx, cv_idx) * 1.1); // Increase CV->CV
            }
            // Decrease probability of staying in CT if a CV model exists
            if (ct_idx >= 0 && static_cast<size_t>(ct_idx) < models_.size() && cv_idx >= 0 && static_cast<size_t>(cv_idx) < models_.size()) { // Robust index check
                transition_matrix_(ct_idx, ct_idx) = std::max(0.05, transition_matrix_(ct_idx, ct_idx) * 0.9);
            }
        }
        // Add more conditions for other environments (e.g., near_water, jamming_level)
        // Ensure that after any adaptation, the rows of transition_matrix_ sum to 1.0.
        // This is crucial for maintaining valid probabilities.
        for (int i = 0; i < models_.size(); ++i) {
            double row_sum = transition_matrix_.row(i).sum();
            if (row_sum > 1e-12) { // Avoid division by zero
                transition_matrix_.row(i) /= row_sum;
            } else {
                // If a row sums to zero, it means no transitions are possible from this model.
                // This might indicate an issue or a model that should be pruned.
                // For robustness, re-uniform the row or set a default transition.
                transition_matrix_.row(i).setConstant(1.0 / models_.size());
            }
        }
    }

    /**
     * @brief Prunes inactive models by setting their probabilities to a very small value.
     * This can save computation by avoiding prediction/update for low-probability models.
     * @param threshold The probability threshold below which a model is considered inactive.
     *
     * @note This method primarily adjusts model probabilities. For true computational savings,
     * the `predict` and `update` loops would need to explicitly check `model_probabilities_(i)`
     * against a threshold and skip computations for models below it. The current implementation
     * still iterates through all models, but their low probabilities mean they contribute
     * negligibly to the combined state.
     */
    void pruneInactiveModels(double threshold = 1e-3) {
        for (int i = 0; i < models_.size(); ++i) {
            if (model_probabilities_(i) < threshold) {
                model_probabilities_(i) = 1e-10; // Set to a very small non-zero value
            }
        }
        // Re-normalize probabilities after pruning, guarding against zero sum
        double sum_probabilities = model_probabilities_.sum();
        if (sum_probabilities < 1e-30) {
            // If sum is near zero after pruning, re-uniform to prevent division by zero
            model_probabilities_.setConstant(1.0 / models_.size());
        } else {
            model_probabilities_ /= sum_probabilities;
        }
    }

    /**
     * @brief Returns the overall combined state estimate.
     * @return The combined state vector.
     */
    const StateType& getCombinedState() const {
        return combined_state_;
    }

    /**
     * @brief Returns the overall combined covariance matrix.
     * @return The combined covariance matrix.
     */
    const CovarianceMatrixType& getCombinedCovariance() const {
        return combined_covariance_;
    }

private:
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
                                   0.5 * std::log(innovation_covariance_determinant); // Use cached determinant

        double log_power_term = -(degrees_of_freedom + measurement_dim) / 2.0 * std::log(t_pdf_term);

        return std::exp(log_constant_part + log_power_term);
    }
};
