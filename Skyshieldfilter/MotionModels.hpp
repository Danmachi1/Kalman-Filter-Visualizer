// MotionModels.hpp
#ifndef SKYFILTER_MOTION_MODELS_HPP_
#define SKYFILTER_MOTION_MODELS_HPP_

#include <Eigen/Dense> // For Eigen types
#include <cmath>       // For std::sin, std::cos, std::abs, std::fmod, std::exp
#include <algorithm>   // For std::clamp (C++17)
#include <stdexcept>   // For std::runtime_error

// Include necessary Kalman library headers for base classes and types
// Assuming these are available in your Kalman library setup:
#include "Kalman/Vector.hpp"
#include "Kalman/Matrix.hpp"
#include "Kalman/CovarianceSquareRoot.hpp" // For Kalman::CovarianceSquareRoot
#include "Kalman/UnscentedKalmanFilterBase.hpp" // For Kalman::SystemModelType


namespace Kalman {
    // Re-defining SystemModelBase to ensure it matches the expected interface of
    // Kalman::UnscentedKalmanFilterBase's SystemModelType template parameter.
    // This template parameter is typically expected to be a class that provides:
    // 1. A 'f' method for state transition.
    // 2. A 'getCovarianceSquareRoot' method for process noise.
    // We use Kalman::Vector and Kalman::Matrix for consistency with the Kalman library.
    template<int StateDim, int ControlDim>
    class SystemModelBase {
    public:
        using StateType = Kalman::Vector<double, StateDim>;
        using ControlType = Kalman::Vector<double, ControlDim>;
        using CovarianceMatrix = Kalman::Matrix<double, StateDim, StateDim>;
        using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<StateType>;

        // Pure virtual function for the state transition function f(x, u, dt)
        virtual StateType f(const StateType& x, const ControlType& u, double dt) const = 0;

        // Pure virtual function to get the square root of the process noise covariance (S_Q)
        // This is expected to be constant by the SRUKF's predict method's signature.
        virtual CovarianceSquareRootType getCovarianceSquareRoot() const = 0;

        // Virtual method to set the full process noise covariance matrix (Q)
        // This method will compute and store the Cholesky decomposition (S_Q).
        // Default implementation throws an error if not overridden, as not all models might support it.
        virtual void setFullProcessNoiseCovariance(const CovarianceMatrix& full_Q_matrix) {
            throw std::runtime_error("setFullProcessNoiseCovariance not implemented for this model.");
        }

        /**
         * @brief Virtual method to update the process noise based on time step or other factors.
         * This is intended for models like Singer that have dt-dependent Q matrices.
         * Derived classes should override this if their Q depends on dt or other dynamic parameters.
         * Default implementation does nothing.
         * @param dt The time step.
         */
        virtual void updateProcessNoise(double dt) {
            (void)dt; // Suppress unused parameter warning
            // Default implementation does nothing.
        }

        // Virtual method to get the model name for debugging/logging
        virtual const char* getModelName() const = 0;

        // Virtual destructor for proper polymorphic cleanup
        virtual ~SystemModelBase() = default;
    };
} // namespace Kalman

/**
 * @brief Utility function to wrap an angle to [-PI, PI) range.
 * @param theta The angle in radians.
 * @return The wrapped angle.
 */
inline double wrapAngleRad(double theta) {
    return std::fmod(theta + M_PI, 2 * M_PI) - M_PI;
}


/**
 * @brief Constant Velocity (CV) Motion Model
 *
 * This model assumes the target moves with constant velocity in 3D space.
 * State vector: [x, y, z, vx, vy, vz]^T (6 dimensions)
 * Control input: Not directly used for state transition in this basic model (u = 0).
 * Process noise: Modeled as directly affecting the velocity components, representing
 * uncertainty in the constant velocity assumption.
 */
class ConstantVelocityModel : public Kalman::SystemModelBase<6, 0> { // 6D state, 0D control
public:
    // Define the specific StateType, ControlType, etc. for this model
    using StateType = Kalman::Vector<double, 6>;
    using ControlType = Kalman::Vector<double, 0>; // Control input is 0-dimensional
    using CovarianceMatrix = Kalman::Matrix<double, 6, 6>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<StateType>;

    /**
     * @brief Constructor for the Constant Velocity Model.
     * @param velocity_process_noise_std_dev Standard deviation for the process noise
     * applied directly to the velocity components per time step.
     *
     * @note This implementation assumes that the process noise standard deviation
     * provided here directly corresponds to the uncertainty added to velocity
     * components per unit time. A more physically rigorous derivation for Q
     * from underlying white noise acceleration would result in a dt-dependent,
     * dense Q matrix. However, given the SRUKF's `getCovarianceSquareRoot()`
     * signature (which doesn't take dt), this simpler diagonal S_Q is used.
     * The SRUKF's internal `computeCovarianceSquareRootFromSigmaPoints` will
     * incorporate this S_Q.
     */
    explicit ConstantVelocityModel(double velocity_process_noise_std_dev)
        : velocity_noise_std_dev_(velocity_process_noise_std_dev) {
        // Initialize the square root of the process noise covariance matrix (S_Q).
        // For a 6D CV model, if noise is assumed to directly affect velocity:
        // S_Q is a 6x6 diagonal matrix.
        // Position components (0,1,2) have 0 noise directly, as their change is
        // derived from velocity.
        // Velocity components (3,4,5) have 'velocity_noise_std_dev_' as their
        // standard deviation.
        s_q_.setIdentity(); // Initialize as identity
        s_q_(0,0) = 0.0; // No direct position noise (derived from velocity noise)
        s_q_(1,1) = 0.0;
        s_q_(2,2) = 0.0;
        s_q_(3,3) = velocity_process_noise_std_dev; // Standard deviation for vx noise
        s_q_(4,4) = velocity_process_noise_std_dev; // Standard deviation for vy noise
        s_q_(5,5) = velocity_process_noise_std_dev; // Standard deviation for vz noise
    }

    /**
     * @brief Implements the state transition function for the Constant Velocity model.
     * @param x The current state vector.
     * @param u The control input vector (unused in basic CV).
     * @param dt The time step.
     * @return The predicted next state vector.
     */
    StateType f(const StateType& x, const ControlType& u, double dt) const override {
        // State: [x, y, z, vx, vy, vz]^T
        StateType x_next = x;

        // Position update: x_k+1 = x_k + v_k * dt
        x_next.segment<3>(0) += x.segment<3>(3) * dt;

        // Velocity remains constant: v_k+1 = v_k
        // (Implicit as x_next.segment<3>(3) is already x.segment<3>(3) from x_next = x)

        // Control input 'u' is not directly used in this basic CV model's state transition.
        (void)u; // Suppress unused parameter warning

        return x_next;
    }

    /**
     * @brief Returns the square root of the process noise covariance matrix (S_Q).
     * @return The S_Q matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_q_;
    }

    /**
     * @brief Sets the process noise standard deviation for the velocity components.
     * This allows for dynamic adjustment of the process noise.
     * @param new_sigma The new standard deviation for velocity process noise.
     */
    void setProcessNoise(double new_sigma) {
        velocity_noise_std_dev_ = new_sigma;
        s_q_(3,3) = new_sigma;
        s_q_(4,4) = new_sigma;
        s_q_(5,5) = new_sigma;
    }

    /**
     * @brief Sets the process noise standard deviation for the velocity components
     * using an analytical derivation from a white noise acceleration model.
     *
     * @param accel_std_dev Standard deviation of the underlying white noise acceleration.
     * @param dt The time step.
     *
     * @note This method *dynamically updates the S_Q matrix based on dt*.
     * If using this method, ensure it is called *before each prediction step*
     * with the current dt, as the SRUKF's `getCovarianceSquareRoot()` method
     * (which calls this model's `getCovarianceSquareRoot()`) does not take dt.
     * This makes the `SystemModel` effectively mutable for its Q.
     */
    void setProcessNoiseUsingWhiteNoiseAccel(double accel_std_dev, double dt) {
        // Classic motion model Q matrix terms for constant velocity,
        // derived from white noise acceleration.
        // We are setting the S_Q (square root of Q).
        // If Q = sigma_a^2 * [dt^4/4 ...; dt^3/2 ...; dt^2 ...], then S_Q is more complex.
        // For diagonal S_Q, we can approximate by taking sqrt of diagonal elements.
        // Here, we are setting the diagonal elements of S_Q directly.
        // This is a common simplification for the S_Q matrix when it needs to be dt-dependent
        // but the getCovarianceSquareRoot() method is const and takes no dt.
        // The values represent standard deviations.
        s_q_(0,0) = 0.5 * dt * dt * accel_std_dev; // Position noise derived from accel
        s_q_(1,1) = 0.5 * dt * dt * accel_std_dev;
        s_q_(2,2) = 0.5 * dt * dt * accel_std_dev;
        s_q_(3,3) = dt * accel_std_dev;           // Velocity noise derived from accel
        s_q_(4,4) = dt * accel_std_dev;
        s_q_(5,5) = dt * accel_std_dev;
    }

    /**
     * @brief Sets the full process noise covariance matrix (Q) and computes its Cholesky decomposition (S_Q).
     * @param full_Q_matrix The full, positive semi-definite Q matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     */
    void setFullProcessNoiseCovariance(const CovarianceMatrix& full_Q_matrix) override {
        Eigen::LLT<CovarianceMatrix> llt(full_Q_matrix);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("ConstantVelocityModel: Input Q matrix is not positive definite.");
        }
        s_q_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Implements the virtual updateProcessNoise method.
     * For Constant Velocity model, process noise is typically constant or set externally,
     * so this is a no-op unless setProcessNoiseUsingWhiteNoiseAccel is used.
     * @param dt The time step.
     */
    void updateProcessNoise(double dt) override {
        (void)dt; // Suppress unused parameter warning
        // No-op for this model's default behavior.
        // If setProcessNoiseUsingWhiteNoiseAccel is intended to be called here,
        // that logic needs to be added (e.g., setProcessNoiseUsingWhiteNoiseAccel(my_accel_std_dev, dt);)
    }

    /**
     * @brief Returns the name of the motion model.
     */
    const char* getModelName() const override {
        return "CV";
    }

private:
    double velocity_noise_std_dev_; // Standard deviation for process noise on velocity
    CovarianceSquareRootType s_q_; // Square root of the process noise covariance matrix (S_Q)
};

/**
 * @brief Constant Turn (CT) Motion Model
 *
 * This model assumes the target moves with a constant turn rate in the XY plane
 * and constant velocity in the Z direction.
 * State vector: [x, y, z, vx, vy, vz, omega_z]^T (7 dimensions)
 * Control input: Not directly used for state transition (u = 0).
 * Process noise: Modeled as directly affecting velocity components and turn rate,
 * representing uncertainty in the constant turn and velocity assumptions.
 * The process noise on `omega_z` implicitly allows for a "Random Walk Turn Model" behavior.
 */
class ConstantTurnModel : public Kalman::SystemModelBase<7, 0> { // 7D state, 0D control
public:
    // Define the specific StateType, ControlType, etc. for this model
    using StateType = Kalman::Vector<double, 7>;
    using ControlType = Kalman::Vector<double, 0>; // Control input is 0-dimensional
    using CovarianceMatrix = Kalman::Matrix<double, 7, 7>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<StateType>;

    /**
     * @brief Constructor for the Constant Turn Model.
     * @param turn_rate_process_noise_std_dev Standard deviation for the process noise
     * applied directly to the turn rate (omega_z).
     * @param velocity_process_noise_std_dev Standard deviation for the process noise
     * applied directly to the velocity components (vx, vy, vz).
     * @param max_turn_rate_rad_s Maximum allowed turn rate in radians/second for clamping.
     * This helps numerical stability and represents physical limits of the UAV.
     */
    ConstantTurnModel(double turn_rate_process_noise_std_dev, double velocity_process_noise_std_dev,
                      double max_turn_rate_rad_s = M_PI) // Default to PI rad/s (180 deg/s)
        : turn_rate_noise_std_dev_(turn_rate_process_noise_std_dev),
          velocity_noise_std_dev_(velocity_process_noise_std_dev),
          max_omega_z_(max_turn_rate_rad_s) {
        // Initialize the square root of the process noise covariance matrix (S_Q).
        // S_Q is a 7x7 diagonal matrix.
        // Position components (0,1,2) have 0 direct noise.
        // Velocity components (3,4,5) have 'velocity_noise_std_dev_'.
        // Turn rate component (6) has 'turn_rate_noise_std_dev_'.
        s_q_.setIdentity();
        s_q_(0,0) = 0.0; // x
        s_q_(1,1) = 0.0; // y
        s_q_(2,2) = 0.0; // z
        s_q_(3,3) = velocity_process_noise_std_dev; // vx
        s_q_(4,4) = velocity_process_noise_std_dev; // vy
        s_q_(5,5) = velocity_process_noise_std_dev; // vz
        s_q_(6,6) = turn_rate_process_noise_std_dev; // omega_z
    }

    /**
     * @brief Implements the state transition function for the Constant Turn model.
     * @param x The current state vector: [x, y, z, vx, vy, vz, omega_z]^T
     * @param u The control input vector (unused in basic CT).
     * @param dt The time step.
     * @return The predicted next state vector.
     */
    StateType f(const StateType& x, const ControlType& u, double dt) const override {
        // State: [x, y, z, vx, vy, vz, omega_z]^T
        StateType x_next = x;

        double px = x(0);
        double py = x(1);
        double pz = x(2);
        double vx = x(3);
        double vy = x(4);
        double vz = x(5);
        double omega_z = x(6); // Yaw rate

        // Clamp omega_z to a reasonable range for numerical stability.
        // This prevents extremely large or small values from causing issues with trig functions
        // and represents physical limits of the UAV.
        omega_z = std::clamp(omega_z, -max_omega_z_, max_omega_z_);

        double omega_dt = omega_z * dt;

        // Cache sine and cosine values to avoid redundant computations
        double sin_omega_dt = std::sin(omega_dt);
        double cos_omega_dt = std::cos(omega_dt);

        // Handle near-zero turn rate (straight line motion) to avoid division by zero
        // and ensure numerical stability for very small omega_z.
        if (std::abs(omega_z) < 1e-6) { // Effectively constant velocity in XY plane
            x_next(0) = px + vx * dt;
            x_next(1) = py + vy * dt;
            // vx, vy remain constant (already copied from x)
        } else {
            // Non-linear equations for Constant Turn in 2D (XY plane)
            // Position update
            x_next(0) = px + (vx * sin_omega_dt - vy * (cos_omega_dt - 1.0)) / omega_z;
            x_next(1) = py + (vx * (cos_omega_dt - 1.0) + vy * sin_omega_dt) / omega_z;

            // Velocity update (rotated velocities)
            // This is a rotation of the velocity vector (vx, vy) by angle omega_z * dt
            // into the new coordinate frame.
            x_next(3) = vx * cos_omega_dt - vy * sin_omega_dt;
            x_next(4) = vx * sin_omega_dt + vy * cos_omega_dt;
        }

        // Z-axis motion (constant velocity)
        x_next(2) = pz + vz * dt;
        // vz remains constant (already copied from x)

        // Turn rate remains constant
        x_next(6) = omega_z; // Already copied from x

        // Control input 'u' is not directly used in this basic CT model's state transition.
        (void)u; // Suppress unused parameter warning

        return x_next;
    }

    /**
     * @brief Returns the square root of the process noise covariance matrix (S_Q).
     * @return The S_Q matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_q_;
    }

    /**
     * @brief Sets the process noise standard deviation for the turn rate and velocity components.
     * This allows for dynamic adjustment of the process noise.
     * @param new_turn_rate_sigma The new standard deviation for turn rate process noise.
     * @param new_velocity_sigma The new standard deviation for velocity process noise.
     */
    void setProcessNoise(double new_turn_rate_sigma, double new_velocity_sigma) {
        turn_rate_noise_std_dev_ = new_turn_rate_sigma;
        velocity_noise_std_dev_ = new_velocity_sigma;

        s_q_(3,3) = new_velocity_sigma;
        s_q_(4,4) = new_velocity_sigma;
        s_q_(5,5) = new_velocity_sigma;
        s_q_(6,6) = new_turn_rate_sigma;
    }

    /**
     * @brief Sets the full process noise covariance matrix (Q) and computes its Cholesky decomposition (S_Q).
     * @param full_Q_matrix The full, positive semi-definite Q matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     */
    void setFullProcessNoiseCovariance(const CovarianceMatrix& full_Q_matrix) override {
        Eigen::LLT<CovarianceMatrix> llt(full_Q_matrix);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("ConstantTurnModel: Input Q matrix is not positive definite.");
        }
        s_q_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Implements the virtual updateProcessNoise method.
     * For Constant Turn model, process noise is typically constant or set externally,
     * so this is a no-op.
     * @param dt The time step.
     */
    void updateProcessNoise(double dt) override {
        (void)dt; // Suppress unused parameter warning
        // No-op for this model's default behavior.
    }

    /**
     * @brief Sets the maximum allowed turn rate for clamping.
     * @param new_max_omega_z The new maximum turn rate in radians/second.
     */
    void setMaxTurnRate(double new_max_omega_z) {
        if (new_max_omega_z > 0) { // Ensure it's a positive value
            max_omega_z_ = new_max_omega_z;
        }
    }

    /**
     * @brief Returns the name of the motion model.
     */
    const char* getModelName() const override {
        return "CT";
    }

private:
    double turn_rate_noise_std_dev_;
    double velocity_noise_std_dev_;
    double max_omega_z_; // Maximum allowed turn rate for clamping
    CovarianceSquareRootType s_q_; // Square root of the process noise covariance matrix (S_Q)
};

/**
 * @brief Constant Acceleration (CA) Motion Model
 *
 * This model assumes the target moves with constant acceleration in 3D space.
 * State vector: [x, y, z, vx, vy, vz, ax, ay, az]^T (9 dimensions)
 * Control input: Not directly used for state transition (u = 0).
 * Process noise: Modeled as directly affecting the acceleration components, representing
 * uncertainty in the constant acceleration assumption.
 */
class ConstantAccelerationModel : public Kalman::SystemModelBase<9, 0> { // 9D state, 0D control
public:
    // Define the specific StateType, ControlType, etc. for this model
    using StateType = Kalman::Vector<double, 9>;
    using ControlType = Kalman::Vector<double, 0>; // Control input is 0-dimensional
    using CovarianceMatrix = Kalman::Matrix<double, 9, 9>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<StateType>;

    /**
     * @brief Constructor for the Constant Acceleration Model.
     * @param acceleration_process_noise_std_dev Standard deviation for the process noise
     * applied directly to the acceleration components.
     *
     * @note Similar to CV and CT, this assumes a constant S_Q. A more rigorous
     * derivation from white noise jerk would be dt-dependent and more complex.
     */
    explicit ConstantAccelerationModel(double acceleration_process_noise_std_dev)
        : acceleration_noise_std_dev_(acceleration_process_noise_std_dev) {
        // Initialize the square root of the process noise covariance matrix (S_Q).
        // S_Q is a 9x9 diagonal matrix.
        // Position and velocity components have 0 direct noise (derived from acceleration noise).
        // Acceleration components (6,7,8) have 'acceleration_process_noise_std_dev_'.
        s_q_.setIdentity();
        s_q_(0,0) = 0.0; // x
        s_q_(1,1) = 0.0; // y
        s_q_(2,2) = 0.0; // z
        s_q_(3,3) = 0.0; // vx
        s_q_(4,4) = 0.0; // vy
        s_q_(5,5) = 0.0; // vz
        s_q_(6,6) = acceleration_process_noise_std_dev; // ax
        s_q_(7,7) = acceleration_process_noise_std_dev; // ay
        s_q_(8,8) = acceleration_process_noise_std_dev; // az
    }

    /**
     * @brief Implements the state transition function for the Constant Acceleration model.
     * @param x The current state vector: [x, y, z, vx, vy, vz, ax, ay, az]^T
     * @param u The control input vector (unused in basic CA).
     * @param dt The time step.
     * @return The predicted next state vector.
     */
    StateType f(const StateType& x, const ControlType& u, double dt) const override {
        // State: [x, y, z, vx, vy, vz, ax, ay, az]^T
        StateType x_next = x;

        // Position update: x_k+1 = x_k + v_k * dt + 0.5 * a_k * dt^2
        x_next.segment<3>(0) += x.segment<3>(3) * dt + 0.5 * x.segment<3>(6) * dt * dt;

        // Velocity update: v_k+1 = v_k + a_k * dt
        x_next.segment<3>(3) += x.segment<3>(6) * dt;

        // Acceleration remains constant: a_k+1 = a_k
        // (Implicit as x_next.segment<3>(6) is already x.segment<3>(6) from x_next = x)

        // Control input 'u' is not directly used in this basic CA model's state transition.
        (void)u; // Suppress unused parameter warning

        return x_next;
    }

    /**
     * @brief Returns the square root of the process noise covariance matrix (S_Q).
     * @return The S_Q matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_q_;
    }

    /**
     * @brief Sets the process noise standard deviation for the acceleration components.
     * This allows for dynamic adjustment of the process noise.
     * @param new_sigma The new standard deviation for acceleration process noise.
     */
    void setProcessNoise(double new_sigma) {
        acceleration_noise_std_dev_ = new_sigma;
        s_q_(6,6) = new_sigma;
        s_q_(7,7) = new_sigma;
        s_q_(8,8) = new_sigma;
    }

    /**
     * @brief Sets the full process noise covariance matrix (Q) and computes its Cholesky decomposition (S_Q).
     * @param full_Q_matrix The full, positive semi-definite Q matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     */
    void setFullProcessNoiseCovariance(const CovarianceMatrix& full_Q_matrix) override {
        Eigen::LLT<CovarianceMatrix> llt(full_Q_matrix);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("ConstantAccelerationModel: Input Q matrix is not positive definite.");
        }
        s_q_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Implements the virtual updateProcessNoise method.
     * For Constant Acceleration model, process noise is typically constant or set externally,
     * so this is a no-op.
     * @param dt The time step.
     */
    void updateProcessNoise(double dt) override {
        (void)dt; // Suppress unused parameter warning
        // No-op for this model's default behavior.
    }

    /**
     * @brief Returns the name of the motion model.
     */
    const char* getModelName() const override {
        return "CA";
    }

private:
    double acceleration_noise_std_dev_; // Standard deviation for process noise on acceleration
    CovarianceSquareRootType s_q_; // Square root of the process noise covariance matrix (S_Q)
};

/**
 * @brief Singer (Correlated Acceleration) Motion Model
 *
 * This model assumes acceleration is not constant but decays exponentially over time
 * and is driven by white noise (Ornstein-Uhlenbeck process).
 * State vector: [x, y, z, vx, vy, vz, ax, ay, az]^T (9 dimensions)
 * Control input: Not directly used for state transition (u = 0).
 * Process noise: Modeled as white noise affecting the acceleration's rate of change (jerk).
 */
class SingerModel : public Kalman::SystemModelBase<9, 0> { // 9D state, 0D control
public:
    // Define the specific StateType, ControlType, etc. for this model
    using StateType = Kalman::Vector<double, 9>;
    using ControlType = Kalman::Vector<double, 0>; // Control input is 0-dimensional
    using CovarianceMatrix = Kalman::Matrix<double, 9, 9>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<StateType>;

    /**
     * @brief Constructor for the Singer Model.
     * @param decay_rate The decay rate (alpha) of the acceleration (e.g., 0.5-2.0).
     * @param white_noise_accel_std_dev Standard deviation of the white noise driving the acceleration (jerk).
     *
     * @note The process noise covariance (Q) for the Singer model is typically
     * derived analytically and is dependent on `dt` and `alpha`.
     * As `getCovarianceSquareRoot()` is const and takes no `dt`, we will
     * provide a `setProcessNoise` method that computes S_Q based on `dt`.
     * This method *must* be called before each prediction step.
     */
    SingerModel(double decay_rate, double white_noise_accel_std_dev)
        : alpha_(decay_rate), white_noise_accel_std_dev_(white_noise_accel_std_dev) {
        // Initialize s_q_ to identity; it will be dynamically set by setProcessNoise.
        s_q_.setIdentity();
    }

    /**
     * @brief Implements the state transition function for the Singer model.
     * @param x The current state vector: [x, y, z, vx, vy, vz, ax, ay, az]^T
     * @param u The control input vector (unused in basic Singer).
     * @param dt The time step.
     * @return The predicted next state vector.
     */
    StateType f(const StateType& x, const ControlType& u, double dt) const override {
        // State: [x, y, z, vx, vy, vz, ax, ay, az]^T
        StateType x_next = x;

        // Extract current state components
        Eigen::Vector3d p_k = x.segment<3>(0);
        Eigen::Vector3d v_k = x.segment<3>(3);
        Eigen::Vector3d a_k = x.segment<3>(6);

        // Calculate transition terms based on Singer model
        double exp_alpha_dt = std::exp(-alpha_ * dt);
        double term1_vel = (1.0 - exp_alpha_dt) / alpha_;
        double term2_pos = (dt - term1_vel) / alpha_;

        // Predict next acceleration: a_k+1 = a_k * exp(-alpha * dt)
        x_next.segment<3>(6) = a_k * exp_alpha_dt;

        // Predict next velocity: v_k+1 = v_k + a_k * (1 - exp(-alpha * dt)) / alpha
        x_next.segment<3>(3) = v_k + a_k * term1_vel;

        // Predict next position: p_k+1 = p_k + v_k * dt + a_k * (dt - (1 - exp(-alpha * dt)) / alpha) / alpha
        x_next.segment<3>(0) = p_k + v_k * dt + a_k * term2_pos;

        // Control input 'u' is not directly used in this basic Singer model's state transition.
        (void)u; // Suppress unused parameter warning

        return x_next;
    }

    /**
     * @brief Returns the square root of the process noise covariance matrix (S_Q).
     * @return The S_Q matrix.
     *
     * @note This method returns the last computed S_Q. It is expected that
     * `setProcessNoise` or `setProcessNoiseUsingWhiteNoiseJerk` is called
     * before each prediction step with the current `dt`.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_q_;
    }

    /**
     * @brief Sets the process noise covariance (Q) for the Singer model based on
     * white noise jerk and time step.
     *
     * @param dt The time step.
     * @note This method *dynamically updates the S_Q matrix based on dt*.
     * It should be called *before each prediction step* with the current dt.
     */
    void setProcessNoise(double dt) {
        // Analytical derivation of Q for Singer model (from white noise jerk)
        // Q is a 9x9 matrix, often block-diagonal for x,y,z components.
        // For each dimension (e.g., x):
        // Q_xx = sigma_j^2 * [ (1 - exp(-2*alpha*dt)) / (2*alpha^3) + dt / alpha^2 + (1 - exp(-alpha*dt))^2 / alpha^3 ]
        // Q_vxvx = sigma_j^2 * [ (1 - exp(-2*alpha*dt)) / (2*alpha) ]
        // Q_axax = sigma_j^2 * [ (1 - exp(-2*alpha*dt)) * alpha / 2 ]
        // and cross-terms.

        // A simplified diagonal Q for the Singer model, where noise directly affects acceleration,
        // and its magnitude is influenced by the decay rate and jerk std dev.
        // For a more precise Q, the full analytical derivation should be used.
        // Here, we'll use a simplified approach that scales with dt and accel_std_dev
        // for the diagonal S_Q, similar to how it's handled in other models for now.

        double sigma_j_sq = white_noise_accel_std_dev_ * white_noise_accel_std_dev_;
        double alpha_sq = alpha_ * alpha_;
        double alpha_cubed = alpha_sq * alpha_;
        double exp_neg_alpha_dt = std::exp(-alpha_ * dt);
        double exp_neg_2_alpha_dt = std::exp(-2 * alpha_ * dt);

        // Q_11 (position variance)
        double q_pos_term = sigma_j_sq * (1.0 - exp_neg_2_alpha_dt) / (2.0 * alpha_cubed)
                            + sigma_j_sq * dt / alpha_sq
                            - 2.0 * sigma_j_sq * (1.0 - exp_neg_alpha_dt) / alpha_cubed;

        // Q_22 (velocity variance)
        double q_vel_term = sigma_j_sq * (1.0 - exp_neg_2_alpha_dt) / (2.0 * alpha_);

        // Q_33 (acceleration variance)
        double q_accel_term = sigma_j_sq * alpha_ * (1.0 - exp_neg_2_alpha_dt) / 2.0;

        // Ensure terms are non-negative for sqrt
        q_pos_term = std::max(0.0, q_pos_term);
        q_vel_term = std::max(0.0, q_vel_term);
        q_accel_term = std::max(0.0, q_accel_term);

        s_q_.setIdentity();
        s_q_(0,0) = std::sqrt(q_pos_term);
        s_q_(1,1) = std::sqrt(q_pos_term);
        s_q_(2,2) = std::sqrt(q_pos_term);
        s_q_(3,3) = std::sqrt(q_vel_term);
        s_q_(4,4) = std::sqrt(q_vel_term);
        s_q_(5,5) = std::sqrt(q_vel_term);
        s_q_(6,6) = std::sqrt(q_accel_term);
        s_q_(7,7) = std::sqrt(q_accel_term);
        s_q_(8,8) = std::sqrt(q_accel_term);
    }

    /**
     * @brief Sets the full process noise covariance matrix (Q) and computes its Cholesky decomposition (S_Q).
     * @param full_Q_matrix The full, positive semi-definite Q matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     */
    void setFullProcessNoiseCovariance(const CovarianceMatrix& full_Q_matrix) override {
        Eigen::LLT<CovarianceMatrix> llt(full_Q_matrix);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("SingerModel: Input Q matrix is not positive definite.");
        }
        s_q_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Implements the virtual updateProcessNoise method.
     * For Singer model, process noise is dynamically updated based on dt.
     * @param dt The time step.
     */
    void updateProcessNoise(double dt) override {
        setProcessNoise(dt); // Call the specific setProcessNoise for Singer model
    }

    /**
     * @brief Returns the name of the motion model.
     */
    const char* getModelName() const override {
        return "Singer";
    }

private:
    double alpha_; // Decay rate
    double white_noise_accel_std_dev_; // Standard deviation of the white noise (jerk)
    CovarianceSquareRootType s_q_; // Square root of the process noise covariance matrix (S_Q)
};

#endif // SKYFILTER_MOTION_MODELS_HPP_
