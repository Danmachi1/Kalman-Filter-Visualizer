// SystemModels.hpp
#ifndef SKYFILTER_SYSTEM_MODELS_HPP_
#define SKYFILTER_SYSTEM_MODELS_HPP_

#include <Eigen/Dense> // For Eigen types
#include <cmath>       // For std::sin, std::cos, std::abs
#include <stdexcept>   // For std::runtime_error

// Include necessary Kalman library headers for base classes and types
// These should be from the same Kalman library as your SRUKF.
#include "Kalman/Vector.hpp"
#include "Kalman/Matrix.hpp"
#include "Kalman/CovarianceSquareRoot.hpp" // For Kalman::CovarianceSquareRoot
#include "Kalman/UnscentedKalmanFilterBase.hpp" // For Kalman::SystemModelType

namespace Kalman {
    // Re-defining SystemModelBase to ensure it matches the expected interface of
    // Kalman::UnscentedKalmanFilterBase's SystemModelType template parameter.
    // This template parameter is typically expected to be a class that provides:
    // 1. An 'f' method for state prediction.
    // 2. A 'getCovarianceSquareRoot' method for process noise.
    // We use Kalman::Vector and Kalman::Matrix for consistency with the Kalman library.
    template<int StateDim>
    class SystemModelBase {
    public:
        using StateType = Kalman::Vector<double, StateDim>;
        using ControlType = Kalman::Vector<double, 0>; // Default to no control input
        using CovarianceMatrix = Kalman::Matrix<double, StateDim, StateDim>;
        using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<StateType>;

        // Pure virtual function for the state prediction function f(x, u, w)
        // Here, 'u' is control input (optional), 'w' is process noise (implicitly handled by SRUKF)
        virtual StateType f(const StateType& x, const ControlType& u, double dt) const = 0;

        // Pure virtual function to get the square root of the process noise covariance (S_Q)
        // This version is typically used by SRUKF, which expects a constant S_Q.
        virtual CovarianceSquareRootType getCovarianceSquareRoot() const = 0;

        // Optional: Virtual function to get the square root of the process noise covariance (S_Q)
        // that might be time-step dependent. This is more common for process noise (Q).
        // Derived classes should override this for time-varying Q.
        virtual CovarianceSquareRootType getCovarianceSquareRoot(double dt) const {
            // By default, process noise Q might be time-step dependent if not explicitly modeled.
            // However, for simplicity, we return the constant version.
            // Derived classes should implement time-varying Q if needed.
            (void)dt; // Suppress unused parameter warning
            return getCovarianceSquareRoot();
        }

        // Virtual method to set the full process noise covariance matrix (Q)
        // This method will compute and store the Cholesky decomposition (S_Q).
        // Default implementation throws an error if not overridden.
        virtual void setFullProcessNoiseCovariance(const CovarianceMatrix& full_Q_matrix) {
            throw std::runtime_error("setFullProcessNoiseCovariance not implemented for this model.");
        }

        // Virtual method to dynamically adjust process noise based on environmental context.
        // This allows for inflating Q near obstacles or in dense clutter.
        // Derived classes should override this to implement specific adaptation logic.
        virtual void setProcessNoiseFromContext(const struct EnvironmentalContext& ctx, const StateType& x) {
            // Default implementation does nothing or throws, derived classes will implement.
            (void)ctx; // Suppress unused parameter warning
            (void)x; // Suppress unused parameter warning
            throw std::runtime_error("setProcessNoiseFromContext not implemented for this model.");
        }

        // Pure virtual methods for dimension introspection
        virtual int getStateDim() const = 0;

        // Virtual method to get the model name for debugging/logging
        virtual const char* getModelName() const = 0;

        // Virtual destructor for proper polymorphic cleanup
        virtual ~SystemModelBase() = default;
    };
} // namespace Kalman

// Forward declaration of EnvironmentalContext (defined in MeasurementModels.hpp or a common header)
// Assuming it's defined in MeasurementModels.hpp for now, or moved to a common header.
struct EnvironmentalContext; // This needs to be a full definition if used here.
// For now, let's assume it's included from a common utility header.
// For a complete build, you might need a common `Contexts.hpp` file.

/**
 * @brief Constant Velocity (CV) System Model for 3D motion.
 *
 * This model assumes constant velocity in 3D (x, y, z, vx, vy, vz).
 * The state vector is [x, y, z, vx, vy, vz]^T.
 *
 * @tparam StateDim The dimension of the state vector (must be 6 for this model).
 */
template<int StateDim>
class ConstantVelocitySystemModel : public Kalman::SystemModelBase<StateDim> {
    // Compile-time check to ensure StateDim is 6
    static_assert(StateDim == 6, "ConstantVelocitySystemModel requires a 6-dimensional state vector [x, y, z, vx, vy, vz].");

public:
    using StateType = Kalman::Vector<double, StateDim>;
    using ControlType = Kalman::Vector<double, 0>; // No control input for basic CV
    using CovarianceMatrix = Kalman::Matrix<double, StateDim, StateDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<StateType>;

    /**
     * @brief Constructor for the Constant Velocity System Model.
     * @param process_noise_std_dev_initial Initial standard deviation for the process noise
     * (acceleration components) for X, Y, Z axes. This is a baseline.
     * @param damping_factor_initial Initial damping factor for velocities, useful in cluttered environments.
     */
    explicit ConstantVelocitySystemModel(const Eigen::Vector3d& process_noise_std_dev_initial,
                                         double damping_factor_initial = 0.0) // Default to no damping
        : process_noise_std_dev_base_(process_noise_std_dev_initial),
          initial_process_noise_std_dev_base_(process_noise_std_dev_initial),
          damping_factor_(damping_factor_initial)
    {
        // Initialize the square root of the process noise covariance matrix (S_Q).
        // S_Q will be dynamically computed in getCovarianceSquareRoot(dt).
        // This initial s_q_ is primarily for setFullProcessNoiseCovariance if called.
        s_q_.setIdentity();
        // A nominal initialization for s_q_ (not dt-dependent here)
        s_q_(0,0) = process_noise_std_dev_base_(0);
        s_q_(1,1) = process_noise_std_dev_base_(1);
        s_q_(2,2) = process_noise_std_dev_base_(2);
        s_q_(3,3) = process_noise_std_dev_base_(0); // Assuming velocity noise related to position noise
        s_q_(4,4) = process_noise_std_dev_base_(1);
        s_q_(5,5) = process_noise_std_dev_base_(2);
    }

    /**
     * @brief Implements the state prediction function for the Constant Velocity model.
     *
     * This function predicts the state at time `t + dt` given the state at `t`.
     * It assumes constant velocity over the time step `dt`, with optional velocity damping.
     *
     * @param x The current state vector [x, y, z, vx, vy, vz]^T.
     * @param u The control input (unused for basic CV model).
     * @param dt The time step.
     * @return The predicted state vector.
     */
    StateType f(const StateType& x, const ControlType& u, double dt) const override {
        (void)u; // Suppress unused parameter warning

        StateType predicted_x = x; // Start with current state

        // Predict position based on current velocity and time step
        predicted_x(0) += x(3) * dt; // x = x + vx * dt
        predicted_x(1) += x(4) * dt; // y = y + vy * dt
        predicted_x(2) += x(5) * dt; // z = z + vz * dt

        // Apply velocity damping for more realism in cluttered environments
        // This reduces the velocity component over time, simulating drag or obstacle interaction.
        predicted_x(3) *= (1.0 - damping_factor_ * dt); // vx damping
        predicted_x(4) *= (1.0 - damping_factor_ * dt); // vy damping
        predicted_x(5) *= (1.0 - damping_factor_ * dt); // vz damping

        return predicted_x;
    }

    /**
     * @brief Returns the square root of the process noise covariance matrix (S_Q).
     * This version is typically used by SRUKF, which expects a constant S_Q.
     *
     * @note This version throws an error, as Q for CV is inherently time-dependent.
     * @return The S_Q matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        throw std::runtime_error("ConstantVelocitySystemModel::getCovarianceSquareRoot() without dt is not recommended. Use getCovarianceSquareRoot(dt).");
    }

    /**
     * @brief Returns the square root of the process noise covariance matrix (S_Q)
     * that is time-step dependent.
     *
     * Q for a CV model is derived from a continuous-time white noise acceleration model.
     * For a 3D CV model with noise on acceleration (q_a_x, q_a_y, q_a_z), the Q matrix
     * components related to position and velocity are fully correlated.
     *
     * @param dt The time step.
     * @return The S_Q matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        CovarianceMatrix Q = CovarianceMatrix::Zero(); // Initialize full Q matrix

        // Get per-axis acceleration variances
        double q_a_x = process_noise_std_dev_base_(0) * process_noise_std_dev_base_(0);
        double q_a_y = process_noise_std_dev_base_(1) * process_noise_std_dev_base_(1);
        double q_a_z = process_noise_std_dev_base_(2) * process_noise_std_dev_base_(2);

        // Construct the full Q matrix with correlated position-velocity noise
        // This is a block diagonal matrix where each block is 2x2 for (pos, vel) for each axis.
        // For X-axis (px, vx)
        Q(0,0) = q_a_x * dt * dt * dt / 3.0; // px-px
        Q(0,3) = q_a_x * dt * dt / 2.0;      // px-vx
        Q(3,0) = q_a_x * dt * dt / 2.0;      // vx-px
        Q(3,3) = q_a_x * dt;                 // vx-vx

        // Block for Y-axis (py, vy)
        Q(1,1) = q_a_y * dt * dt * dt / 3.0; // py-py
        Q(1,4) = q_a_y * dt * dt / 2.0;      // py-vy
        Q(4,1) = q_a_y * dt * dt / 2.0;      // vy-py
        Q(4,4) = q_a_y * dt;                 // vy-vy

        // Block for Z-axis (pz, vz)
        Q(2,2) = q_a_z * dt * dt * dt / 3.0; // pz-pz
        Q(2,5) = q_a_z * dt * dt / 2.0;      // pz-vz
        Q(5,2) = q_a_z * dt * dt / 2.0;      // vz-pz
        Q(5,5) = q_a_z * dt;                 // vz-vz

        // Ensure the matrix is symmetric before Cholesky decomposition for numerical stability
        // (Though constructed this way, it should already be symmetric)
        CovarianceMatrix sym_Q = (Q + Q.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_Q);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("ConstantVelocitySystemModel::getCovarianceSquareRoot(dt): Computed Q matrix is not positive definite or symmetric.");
        }
        return llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Sets the full process noise covariance matrix (Q) and computes its Cholesky decomposition (S_Q).
     * @param full_Q_matrix The full, positive semi-definite Q matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     *
     * @note This method is used by higher-level adaptive logic (e.g., in OnlineNoiseReEstimator)
     * to dynamically adjust the process uncertainty.
     */
    void setFullProcessNoiseCovariance(const CovarianceMatrix& full_Q_matrix) override {
        // Ensure the matrix is symmetric before Cholesky decomposition for numerical stability
        CovarianceMatrix sym_Q = (full_Q_matrix + full_Q_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_Q);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("ConstantVelocitySystemModel: Input Q matrix is not positive definite or symmetric.");
        }
        s_q_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Dynamically adjusts the process noise (S_Q) based on environmental context.
     * @param ctx The EnvironmentalContext containing dynamic environmental information.
     * @param x The current state vector.
     *
     * @note This is a heuristic example. Real-world adaptation functions can be complex.
     * This method would typically modify `process_noise_std_dev_base_` which then affects
     * `getCovarianceSquareRoot(dt)`.
     */
    void setProcessNoiseFromContext(const struct EnvironmentalContext& ctx, const StateType& x) override {
        // Example adaptation logic for CV model noise:
        // Higher building density or closer to obstacles -> higher process noise (more likely to maneuver)
        double environment_factor = 1.0;
        // Assuming ctx.building_density is 0.0 (sparse) to 1.0 (dense)
        environment_factor += ctx.building_density * 0.5; // More dense -> higher noise

        // Assuming ctx.nearest_obstacle_distance is in meters
        if (ctx.nearest_obstacle_distance < 100.0) { // Closer than 100m to an obstacle
            environment_factor += (100.0 - ctx.nearest_obstacle_distance) * 0.01; // Closer -> higher noise
        }
        environment_factor = std::max(1.0, environment_factor); // Ensure factor is at least 1.0

        // Apply factor to the base standard deviation for each axis
        process_noise_std_dev_base_ = initial_process_noise_std_dev_base_.array() * environment_factor;

        // You could also adapt damping_factor_ here if desired
        // damping_factor_ = initial_damping_factor_base_ * environment_factor; // If you had an initial_damping_factor_base_
    }

    /**
     * @brief Returns the dimension of the state vector.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the system model.
     */
    const char* getModelName() const override {
        return "ConstantVelocity";
    }

private:
    Eigen::Vector3d process_noise_std_dev_base_; // Baseline standard deviation for process noise (acceleration) per axis
    Eigen::Vector3d initial_process_noise_std_dev_base_; // Store initial value for adaptation
    double damping_factor_; // Damping factor for velocity
    CovarianceSquareRootType s_q_; // Square root of the process noise covariance matrix (S_Q)
};

/**
 * @brief Constant Turn (CT) System Model for 3D motion with a 2D turn in XY plane.
 *
 * This model assumes constant velocity in Z and a constant turn rate in the XY plane.
 * The state vector is [x, y, z, vx, vy, vz, omega_z]^T.
 *
 * @tparam StateDim The dimension of the state vector (must be 7 for this model).
 */
template<int StateDim>
class ConstantTurnSystemModel : public Kalman::SystemModelBase<StateDim> {
    // Compile-time check to ensure StateDim is 7
    static_assert(StateDim == 7, "ConstantTurnSystemModel requires a 7-dimensional state vector [x, y, z, vx, vy, vz, omega_z].");

public:
    using StateType = Kalman::Vector<double, StateDim>;
    using ControlType = Kalman::Vector<double, 0>; // No control input for basic CT
    using CovarianceMatrix = Kalman::Matrix<double, StateDim, StateDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<StateType>;

    /**
     * @brief Constructor for the Constant Turn System Model.
     * @param process_noise_accel_std_dev_initial Initial standard deviation for the process noise
     * (acceleration components). This is a baseline for kinematic noise.
     * @param process_noise_turn_rate_std_dev_initial Initial standard deviation for the process noise
     * on the turn rate (omega_z).
     */
    ConstantTurnSystemModel(double process_noise_accel_std_dev_initial,
                            double process_noise_turn_rate_std_dev_initial)
        : process_noise_accel_std_dev_base_(process_noise_accel_std_dev_initial),
          initial_process_noise_accel_std_dev_base_(process_noise_accel_std_dev_initial),
          process_noise_turn_rate_std_dev_base_(process_noise_turn_rate_std_dev_initial),
          initial_process_noise_turn_rate_std_dev_base_(process_noise_turn_rate_std_dev_initial)
    {
        // Initialize the square root of the process noise covariance matrix (S_Q).
        // S_Q will be dynamically computed in getCovarianceSquareRoot(dt).
        // This initial s_q_ is primarily for setFullProcessNoiseCovariance if called.
        s_q_.setIdentity();
        // A nominal initialization for s_q_ (not dt-dependent here)
        s_q_(0,0) = process_noise_accel_std_dev_base_;
        s_q_(1,1) = process_noise_accel_std_dev_base_;
        s_q_(2,2) = process_noise_accel_std_dev_base_;
        s_q_(3,3) = process_noise_accel_std_dev_base_;
        s_q_(4,4) = process_noise_accel_std_dev_base_;
        s_q_(5,5) = process_noise_accel_std_dev_base_;
        s_q_(6,6) = process_noise_turn_rate_std_dev_base_;
    }

    /**
     * @brief Implements the state prediction function for the Constant Turn model.
     *
     * This function predicts the state at time `t + dt` given the state at `t`.
     * It assumes a constant turn rate in the XY plane and constant Z velocity.
     * Handles the special case where turn rate approaches zero (reverts to CV).
     *
     * @param x The current state vector [x, y, z, vx, vy, vz, omega_z]^T.
     * @param u The control input (unused for basic CT model).
     * @param dt The time step.
     * @return The predicted state vector.
     */
    StateType f(const StateType& x, const ControlType& u, double dt) const override {
        (void)u; // Suppress unused parameter warning

        StateType predicted_x = x; // Start with current state

        double px = x(0);
        double py = x(1);
        double pz = x(2);
        double vx = x(3);
        double vy = x(4);
        double vz = x(5);
        double omega_z = x(6);

        // Handle omega_z approaching zero (reverts to Constant Velocity model)
        if (std::abs(omega_z) < 1e-6) { // Turn rate is very small, use CV approximation
            predicted_x(0) = px + vx * dt;
            predicted_x(1) = py + vy * dt;
            predicted_x(2) = pz + vz * dt;
            // Velocities and turn rate remain constant
            // predicted_x(3) = vx;
            // predicted_x(4) = vy;
            // predicted_x(5) = vz;
            // predicted_x(6) = omega_z;
        } else {
            // Predict position and velocity for Constant Turn model
            // Formulas from the plan (2D turn in XY plane)
            double sin_omega_dt = std::sin(omega_z * dt);
            double cos_omega_dt = std::cos(omega_z * dt);

            predicted_x(0) = px + (vx * sin_omega_dt - vy * (1 - cos_omega_dt)) / omega_z;
            predicted_x(1) = py + (vy * sin_omega_dt + vx * (1 - cos_omega_dt)) / omega_z;
            predicted_x(2) = pz + vz * dt; // Z-motion is constant velocity

            predicted_x(3) = vx * cos_omega_dt - vy * sin_omega_dt;
            predicted_x(4) = vy * cos_omega_dt + vx * sin_omega_dt;
            // predicted_x(5) = vz; // Z-velocity remains constant
            // predicted_x(6) = omega_z; // Turn rate remains constant
        }

        return predicted_x;
    }

    /**
     * @brief Returns the square root of the process noise covariance matrix (S_Q).
     * This version is typically used by SRUKF, which expects a constant S_Q.
     *
     * @note This version throws an error, as Q for CT is inherently time-dependent.
     * @return The S_Q matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        throw std::runtime_error("ConstantTurnSystemModel::getCovarianceSquareRoot() without dt is not recommended. Use getCovarianceSquareRoot(dt).");
    }

    /**
     * @brief Returns the square root of the process noise covariance matrix (S_Q)
     * that is time-step dependent.
     *
     * Q for a CT model is often derived from continuous-time white noise acceleration
     * and white noise on the turn rate.
     *
     * @param dt The time step.
     * @return The S_Q matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        CovarianceSquareRootType s_q_dt;
        s_q_dt.setZero(); // Initialize to zero

        double q_accel = process_noise_accel_std_dev_base_ * process_noise_accel_std_dev_base_; // q_a = sigma_a^2
        double q_omega = process_noise_turn_rate_std_dev_base_ * process_noise_turn_rate_std_dev_base_; // q_omega = sigma_omega^2

        // Process noise for CT model. This is a simplified diagonal S_Q.
        // For a full CT Q matrix, the derivation is more complex and involves
        // integrals of the state transition matrix components.
        // Here, we approximate with diagonal terms.

        // Noise on position (from acceleration noise)
        s_q_dt(0,0) = std::sqrt(q_accel * dt * dt * dt / 3.0); // x-pos noise
        s_q_dt(1,1) = std::sqrt(q_accel * dt * dt * dt / 3.0); // y-pos noise
        s_q_dt(2,2) = std::sqrt(q_accel * dt * dt * dt / 3.0); // z-pos noise (from accel noise)

        // Noise on velocity (from acceleration noise)
        s_q_dt(3,3) = std::sqrt(q_accel * dt); // vx-vel noise
        s_q_dt(4,4) = std::sqrt(q_accel * dt); // vy-vel noise
        s_q_dt(5,5) = std::sqrt(q_accel * dt); // vz-vel noise (from accel noise)

        // Noise on turn rate (from turn rate noise)
        s_q_dt(6,6) = std::sqrt(q_omega * dt); // omega_z noise

        // Note: A more rigorous Q for CT would have off-diagonal terms, especially
        // between position/velocity and turn rate, and would be derived from the
        // continuous-time noise model integrated over dt. For SRUKF, this diagonal
        // approximation is often used and the sigma point propagation helps.

        return s_q_dt;
    }

    /**
     * @brief Sets the full process noise covariance matrix (Q) and computes its Cholesky decomposition (S_Q).
     * @param full_Q_matrix The full, positive semi-definite Q matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     *
     * @note This method is used by higher-level adaptive logic.
     */
    void setFullProcessNoiseCovariance(const CovarianceMatrix& full_Q_matrix) override {
        // Ensure the matrix is symmetric before Cholesky decomposition for numerical stability
        CovarianceMatrix sym_Q = (full_Q_matrix + full_Q_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_Q);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("ConstantTurnSystemModel: Input Q matrix is not positive definite or symmetric.");
        }
        s_q_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Dynamically adjusts the process noise (S_Q) based on environmental context.
     * @param ctx The EnvironmentalContext containing dynamic environmental information.
     * @param x The current state vector.
     *
     * @note This is a heuristic example. Real-world adaptation functions can be complex.
     * This method would typically modify `process_noise_accel_std_dev_base_` and
     * `process_noise_turn_rate_std_dev_base_` which then affect `getCovarianceSquareRoot(dt)`.
     */
    void setProcessNoiseFromContext(const struct EnvironmentalContext& ctx, const StateType& x) override {
        // Example adaptation logic for CT model noise:
        // Higher building density or closer to obstacles -> higher process noise (more likely to maneuver)
        // Also, higher turn rate noise if in complex environments.
        double environment_factor = 1.0;
        environment_factor += ctx.building_density * 0.7; // More dense -> higher noise (stronger effect than CV)

        if (ctx.nearest_obstacle_distance < 70.0) { // Closer than 70m to an obstacle (CT might react more sharply)
            environment_factor += (70.0 - ctx.nearest_obstacle_distance) * 0.015; // Closer -> higher noise
        }
        environment_factor = std::max(1.0, environment_factor); // Ensure factor is at least 1.0

        // Apply factor to the base standard deviations
        process_noise_accel_std_dev_base_ = initial_process_noise_accel_std_dev_base_ * environment_factor;
        process_noise_turn_rate_std_dev_base_ = initial_process_noise_turn_rate_std_dev_base_ * environment_factor;

        // Note: The actual S_Q matrix will be recomputed in getCovarianceSquareRoot(dt)
        // using these updated base values.
    }

    /**
     * @brief Returns the dimension of the state vector.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the system model.
     */
    const char* getModelName() const override {
        return "ConstantTurn";
    }

private:
    double process_noise_accel_std_dev_base_; // Baseline standard deviation for acceleration noise
    double initial_process_noise_accel_std_dev_base_; // Store initial value for adaptation
    double process_noise_turn_rate_std_dev_base_; // Baseline standard deviation for turn rate noise
    double initial_process_noise_turn_rate_std_dev_base_; // Store initial value for adaptation
    CovarianceSquareRootType s_q_; // Square root of the process noise covariance matrix (S_Q)
};

/**
 * @brief Constant Acceleration (CA) System Model for 3D motion.
 *
 * This model assumes constant acceleration in 3D (x, y, z, vx, vy, vz, ax, ay, az).
 * The state vector is [x, y, z, vx, vy, vz, ax, ay, az]^T.
 *
 * @tparam StateDim The dimension of the state vector (must be 9 for this model).
 */
template<int StateDim>
class ConstantAccelerationSystemModel : public Kalman::SystemModelBase<StateDim> {
    // Compile-time check to ensure StateDim is 9
    static_assert(StateDim == 9, "ConstantAccelerationSystemModel requires a 9-dimensional state vector [x, y, z, vx, vy, vz, ax, ay, az].");

public:
    using StateType = Kalman::Vector<double, StateDim>;
    using ControlType = Kalman::Vector<double, 0>; // No control input for basic CA
    using CovarianceMatrix = Kalman::Matrix<double, StateDim, StateDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<StateType>;

    /**
     * @brief Constructor for the Constant Acceleration System Model.
     * @param process_noise_jerk_std_dev_initial Initial standard deviation for the process noise
     * (jerk components) for X, Y, Z axes. This is a baseline.
     */
    explicit ConstantAccelerationSystemModel(const Eigen::Vector3d& process_noise_jerk_std_dev_initial)
        : process_noise_jerk_std_dev_base_(process_noise_jerk_std_dev_initial),
          initial_process_noise_jerk_std_dev_base_(process_noise_jerk_std_dev_initial)
    {
        // Initialize the square root of the process noise covariance matrix (S_Q).
        // S_Q will be dynamically computed in getCovarianceSquareRoot(dt).
        // This initial s_q_ is primarily for setFullProcessNoiseCovariance if called.
        s_q_.setIdentity();
        // A nominal initialization for s_q_ (not dt-dependent here)
        s_q_(0,0) = process_noise_jerk_std_dev_base_(0);
        s_q_(1,1) = process_noise_jerk_std_dev_base_(1);
        s_q_(2,2) = process_noise_jerk_std_dev_base_(2);
        s_q_(3,3) = process_noise_jerk_std_dev_base_(0); // Assuming velocity noise related to jerk
        s_q_(4,4) = process_noise_jerk_std_dev_base_(1);
        s_q_(5,5) = process_noise_jerk_std_dev_base_(2);
        s_q_(6,6) = process_noise_jerk_std_dev_base_(0); // Assuming acceleration noise related to jerk
        s_q_(7,7) = process_noise_jerk_std_dev_base_(1);
        s_q_(8,8) = process_noise_jerk_std_dev_base_(2);
    }

    /**
     * @brief Implements the state prediction function for the Constant Acceleration model.
     *
     * This function predicts the state at time `t + dt` given the state at `t`.
     * It assumes constant acceleration over the time step `dt`.
     *
     * @param x The current state vector [x, y, z, vx, vy, vz, ax, ay, az]^T.
     * @param u The control input (unused for basic CA model).
     * @param dt The time step.
     * @return The predicted state vector.
     */
    StateType f(const StateType& x, const ControlType& u, double dt) const override {
        (void)u; // Suppress unused parameter warning

        StateType predicted_x = x; // Start with current state

        // Predict position
        predicted_x(0) += x(3) * dt + 0.5 * x(6) * dt * dt; // x = x + vx*dt + 0.5*ax*dt^2
        predicted_x(1) += x(4) * dt + 0.5 * x(7) * dt * dt; // y = y + vy*dt + 0.5*ay*dt^2
        predicted_x(2) += x(5) * dt + 0.5 * x(8) * dt * dt; // z = z + vz*dt + 0.5*az*dt^2

        // Predict velocity
        predicted_x(3) += x(6) * dt; // vx = vx + ax*dt
        predicted_x(4) += x(7) * dt; // vy = vy + ay*dt
        predicted_x(5) += x(8) * dt; // vz = vz + az*dt

        // Accelerations remain constant (ax, ay, az are x(6), x(7), x(8))
        // predicted_x(6) = x(6);
        // predicted_x(7) = x(7);
        // predicted_x(8) = x(8);

        return predicted_x;
    }

    /**
     * @brief Returns the square root of the process noise covariance matrix (S_Q).
     * This version is typically used by SRUKF, which expects a constant S_Q.
     *
     * @note This version throws an error, as Q for CA is inherently time-dependent.
     * @return The S_Q matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        throw std::runtime_error("ConstantAccelerationSystemModel::getCovarianceSquareRoot() without dt is not recommended. Use getCovarianceSquareRoot(dt).");
    }

    /**
     * @brief Returns the square root of the process noise covariance matrix (S_Q)
     * that is time-step dependent.
     *
     * Q for a CA model is derived from a continuous-time white noise jerk model.
     * For a 3D CA model with noise on jerk (q_j_x, q_j_y, q_j_z), the Q matrix
     * components related to position, velocity, and acceleration are fully correlated.
     *
     * @param dt The time step.
     * @return The S_Q matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        CovarianceMatrix Q = CovarianceMatrix::Zero(); // Initialize full Q matrix

        // Get per-axis jerk variances
        double q_j_x = process_noise_jerk_std_dev_base_(0) * process_noise_jerk_std_dev_base_(0);
        double q_j_y = process_noise_jerk_std_dev_base_(1) * process_noise_jerk_std_dev_base_(1);
        double q_j_z = process_noise_jerk_std_dev_base_(2) * process_noise_jerk_std_dev_base_(2);

        // Construct the full Q matrix with correlated pos-vel-accel noise for each axis
        // For a single axis (p, v, a) driven by white noise jerk (q_j), the Q block is:
        // Q_1D = q_j * [ dt^5/120  dt^4/24  dt^3/6;
        //                 dt^4/24   dt^3/3   dt^2/2;
        //                 dt^3/6    dt^2/2   dt ]

        // Block for X-axis (px, vx, ax)
        Q(0,0) = q_j_x * std::pow(dt, 5) / 120.0;
        Q(0,3) = q_j_x * std::pow(dt, 4) / 24.0;
        Q(0,6) = q_j_x * std::pow(dt, 3) / 6.0;

        Q(3,0) = Q(0,3);
        Q(3,3) = q_j_x * std::pow(dt, 3) / 3.0;
        Q(3,6) = q_j_x * std::pow(dt, 2) / 2.0;

        Q(6,0) = Q(0,6);
        Q(6,3) = Q(3,6);
        Q(6,6) = q_j_x * dt;

        // Block for Y-axis (py, vy, ay)
        Q(1,1) = q_j_y * std::pow(dt, 5) / 120.0;
        Q(1,4) = q_j_y * std::pow(dt, 4) / 24.0;
        Q(1,7) = q_j_y * std::pow(dt, 3) / 6.0;

        Q(4,1) = Q(1,4);
        Q(4,4) = q_j_y * std::pow(dt, 3) / 3.0;
        Q(4,7) = q_j_y * std::pow(dt, 2) / 2.0;

        Q(7,1) = Q(1,7);
        Q(7,4) = Q(4,7);
        Q(7,7) = q_j_y * dt;

        // Block for Z-axis (pz, vz, az)
        Q(2,2) = q_j_z * std::pow(dt, 5) / 120.0;
        Q(2,5) = q_j_z * std::pow(dt, 4) / 24.0;
        Q(2,8) = q_j_z * std::pow(dt, 3) / 6.0;

        Q(5,2) = Q(2,5);
        Q(5,5) = q_j_z * std::pow(dt, 3) / 3.0;
        Q(5,8) = q_j_z * std::pow(dt, 2) / 2.0;

        Q(8,2) = Q(2,8);
        Q(8,5) = Q(5,8);
        Q(8,8) = q_j_z * dt;

        // Ensure the matrix is symmetric before Cholesky decomposition for numerical stability
        CovarianceMatrix sym_Q = (Q + Q.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_Q);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("ConstantAccelerationSystemModel::getCovarianceSquareRoot(dt): Computed Q matrix is not positive definite or symmetric.");
        }
        return llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Sets the full process noise covariance matrix (Q) and computes its Cholesky decomposition (S_Q).
     * @param full_Q_matrix The full, positive semi-definite Q matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     *
     * @note This method is used by higher-level adaptive logic.
     */
    void setFullProcessNoiseCovariance(const CovarianceMatrix& full_Q_matrix) override {
        // Ensure the matrix is symmetric before Cholesky decomposition for numerical stability
        CovarianceMatrix sym_Q = (full_Q_matrix + full_Q_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_Q);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("ConstantAccelerationSystemModel: Input Q matrix is not positive definite or symmetric.");
        }
        s_q_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Dynamically adjusts the process noise (S_Q) based on environmental context.
     * @param ctx The EnvironmentalContext containing dynamic environmental information.
     * @param x The current state vector.
     *
     * @note This is a heuristic example. Real-world adaptation functions can be complex.
     * This method would typically modify `process_noise_jerk_std_dev_base_` which then affects
     * `getCovarianceSquareRoot(dt)`.
     */
    void setProcessNoiseFromContext(const struct EnvironmentalContext& ctx, const StateType& x) override {
        // Example adaptation logic for CA model noise:
        // Higher building density or closer to obstacles -> higher process noise (more likely to maneuver)
        double environment_factor = 1.0;
        environment_factor += ctx.building_density * 0.8; // Stronger effect for CA
        if (ctx.nearest_obstacle_distance < 50.0) { // CA might react even more sharply to close obstacles
            environment_factor += (50.0 - ctx.nearest_obstacle_distance) * 0.02;
        }
        environment_factor = std::max(1.0, environment_factor);

        // Apply factor to the base standard deviation for each axis
        process_noise_jerk_std_dev_base_ = initial_process_noise_jerk_std_dev_base_.array() * environment_factor;
    }

    /**
     * @brief Returns the dimension of the state vector.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the system model.
     */
    const char* getModelName() const override {
        return "ConstantAcceleration";
    }

private:
    Eigen::Vector3d process_noise_jerk_std_dev_base_; // Baseline standard deviation for process noise (jerk) per axis
    Eigen::Vector3d initial_process_noise_jerk_std_dev_base_; // Store initial value for adaptation
    CovarianceSquareRootType s_q_; // Square root of the process noise covariance matrix (S_Q)
};

#endif // SKYFILTER_SYSTEM_MODELS_HPP_
