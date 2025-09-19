// measurement_models.hpp
#define _USE_MATH_DEFINES // Required for M_PI on some platforms (e.g., MSVC)
#ifndef SKYFILTER_MEASUREMENT_MODELS_HPP_
#define SKYFILTER_MEASUREMENT_MODELS_HPP_

// Standard Library Headers
#include <cmath>       // For std::sqrt, std::atan2, std::hypot, std::fmod, M_PI, std::log10
#include <limits>      // For std::numeric_limits (for infinity replacement)
#include <stdexcept>   // For std::runtime_error
#include <algorithm>   // For std::max

// Eigen Library Headers
#include <Eigen/Dense>    // For Eigen types (Vector, Matrix)
#include <Eigen/Geometry> // For Eigen::AngleAxisd, Eigen::Quaterniond

// Kalman Library Headers (assuming these are part of your core Kalman framework)
#include "Kalman/Vector.hpp"
#include "Kalman/Matrix.hpp"
#include "Kalman/CovarianceSquareRoot.hpp" // For Kalman::CovarianceSquareRoot
#include "Kalman/UnscentedKalmanFilterBase.hpp" // For Kalman::MeasurementModelType

// Project-specific Headers
#include "common_types.hpp" // For SensorContext and other enums/structs

// --- Class and Method Naming for measurement_models.hpp ---
// This section lists all classes and their methods to be defined in this file,
// derived from the skyfilter_plan, to ensure no mismatches during coding.

// Base Class: Kalman::MeasurementModelBase
//   (This is the base interface required by SquareRootUnscentedKalmanFilter)
//   Methods:
//     virtual MeasurementType h(const StateType& x) const = 0;
//     virtual CovarianceSquareRootType getCovarianceSquareRoot() const = 0;
//     virtual CovarianceSquareRootType getCovarianceSquareRoot(double dt) const;
//     virtual void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix);
//     virtual void setMeasurementNoiseFromContext(const struct SensorContext& ctx);
//     virtual int getMeasurementDim() const = 0;
//     virtual int getStateDim() const = 0;
//     virtual bool applyClutterRejection(const MeasurementType& raw_z, const StateType& predicted_x) const; // Parameter order changed
//     virtual const char* getModelName() const = 0;
//     virtual ~MeasurementModelBase() = default;
//     virtual Kalman::Matrix<double, MeasurementDim, StateDim> computeJacobian(const StateType& x) const; // Return type changed

// Derived Class: RadarPositionMeasurementModel
//   State: [x, y, z, vx, vy, vz, ax, ay, az]^T (max 9D state for compatibility)
//   Measurement: [x_m, y_m, z_m]^T (3D position)
//   Methods:
//     RadarPositionMeasurementModel(double position_measurement_noise_std_dev_initial);
//     MeasurementType h(const StateType& x) const override;
//     CovarianceSquareRootType getCovarianceSquareRoot() const override;
//     CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override;
//     void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override;
//     void setMeasurementNoiseFromContext(const struct SensorContext& ctx) override;
//     int getMeasurementDim() const override;
//     int getStateDim() const override;
//     const char* getModelName() const override;

// Derived Class: RadarRangeBearingElevationMeasurementModel
//   State: [x, y, z, vx, vy, vz, ax, ay, az]^T (max 9D state)
//   Measurement: [range, azimuth, elevation]^T (3D) // RCS_dBsm removed from spec
//   Methods:
//     RadarRangeBearingElevationMeasurementModel(double range_noise_std_dev, double azimuth_noise_std_dev, double elevation_noise_std_dev);
//     MeasurementType h(const StateType& x) const override;
//     CovarianceSquareRootType getCovarianceSquareRoot() const override;
//     CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override;
//     void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override;
//     void setMeasurementNoiseFromContext(const struct SensorContext& ctx) override;
//     int getMeasurementDim() const override;
//     int getStateDim() const override;
//     const char* getModelName() const override;

// Derived Class: RFMeasurementModel
//   State: [x, y, z, vx, vy, vz, RF_Power_dBm, ...]^T (max 9D state, assuming RF_Power_dBm is at index 6)
//   Measurement: [azimuth, elevation, RSSI_dBm]^T (3D)
//   Methods:
//     RFMeasurementModel(double azimuth_noise_std_dev, double elevation_noise_std_dev, double rssi_noise_std_dev, double path_loss_exponent, int RfPowerStateIdx = 6);
//     MeasurementType h(const StateType& x) const override;
//     CovarianceSquareRootType getCovarianceSquareRoot() const override;
//     CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override;
//     void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override;
//     void setMeasurementNoiseFromContext(const struct SensorContext& ctx) override;
//     int getMeasurementDim() const override;
//     int getStateDim() const override;
//     const char* getModelName() const override;

// Derived Class: IMUMeasurementModel
//   State: [x, y, z, vx, vy, vz, ax, ay, az, gx, gy, gz, ...]^T (max 9D state, assuming accel at 6, gyro at 9)
//   Measurement: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]^T (6D)
//   Methods:
//     IMUMeasurementModel(double accel_noise_std_dev, double gyro_noise_std_dev, double gravity_z, int AccelStateIdx = 6, int GyroStateIdx = 9);
//     MeasurementType h(const StateType& x) const override;
//     CovarianceSquareRootType getCovarianceSquareRoot() const override;
//     CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override;
//     void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override;
//     void setMeasurementNoiseFromContext(const struct SensorContext& ctx) override;
//     int getMeasurementDim() const override;
//     int getStateDim() const override;
//     const char* getModelName() const override;

// Derived Class: AcousticMeasurementModel
//   State: [x, y, z, vx, vy, vz, ...]^T (max 9D state)
//   Measurement: [azimuth, elevation]^T (2D)
//   Methods:
//     AcousticMeasurementModel(double azimuth_noise_std_dev, double elevation_noise_std_dev);
//     MeasurementType h(const StateType& x) const override;
//     CovarianceSquareRootType getCovarianceSquareRoot() const override;
//     CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override;
//     void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override;
//     void setMeasurementNoiseFromContext(const struct SensorContext& ctx) override;
//     int getMeasurementDim() const override;
//     int getStateDim() const override;
//     const char* getModelName() const override;

// Derived Class: VisionMeasurementModel
//   State: [x, y, z, vx, vy, vz, ...]^T (max 9D state)
//   Measurement: [x_pixel, y_pixel]^T (2D)
//   Methods:
//     VisionMeasurementModel(double pixel_x_noise_std_dev, double pixel_y_noise_std_dev, ...camera_params...);
//     MeasurementType h(const StateType& x) const override; // This will need camera intrinsics/extrinsics
//     CovarianceSquareRootType getCovarianceSquareRoot() const override;
//     CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override;
//     void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override;
//     void setMeasurementNoiseFromContext(const struct SensorContext& ctx) override;
//     int getMeasurementDim() const override;
//     int getStateDim() const override;
//     const char* getModelName() const override;

// Derived Class: ThermalMeasurementModel
//   State: [x, y, z, vx, vy, vz, ...]^T (max 9D state)
//   Measurement: [x_pixel, y_pixel]^T (2D)
//   Methods:
//     ThermalMeasurementModel(double pixel_x_noise_std_dev, double pixel_y_noise_std_dev, ...camera_params...);
//     MeasurementType h(const StateType& x) const override; // Similar to Vision, needs camera model
//     CovarianceSquareRootType getCovarianceSquareRoot() const override;
//     CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override;
//     void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override;
//     void setMeasurementNoiseFromContext(const struct SensorContext& ctx) override;
//     int getMeasurementDim() const override;
//     int getStateDim() const override;
//     const char* getModelName() const override;

// Derived Class: LidarMeasurementModel
//   State: [x, y, z, vx, vy, vz, ...]^T (max 9D state)
//   Measurement: [range, azimuth, elevation]^T (3D)
//   Methods:
//     LidarMeasurementModel(double range_noise_std_dev, double azimuth_noise_std_dev, double elevation_noise_std_dev);
//     MeasurementType h(const StateType& x) const override;
//     CovarianceSquareRootType getCovarianceSquareRoot() const override;
//     CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override;
//     void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override;
//     void setMeasurementNoiseFromContext(const struct SensorContext& ctx) override;
//     int getMeasurementDim() const override;
//     int getStateDim() const override;
//     const char* getModelName() const override;

// Derived Class: PassiveRFMeasurementModel
//   State: [x, y, z, vx, vy, vz, RF_Power_dBm, ...]^T (max 9D state, assuming RF_Power_dBm is at index 6)
//   Measurement: [azimuth, elevation]^T (2D)
//   Methods:
//     PassiveRFMeasurementModel(double azimuth_noise_std_dev, double elevation_noise_std_dev);
//     MeasurementType h(const StateType& x) const override;
//     CovarianceSquareRootType getCovarianceSquareRoot() const override;
//     CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override;
//     void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override;
//     void setMeasurementNoiseFromContext(const struct SensorContext& ctx) override;
//     int getMeasurementDim() const override;
//     int getStateDim() const override;
//     const char* getModelName() const override;


namespace Kalman {
    // Re-defining MeasurementModelBase to ensure it matches the expected interface of
    // Kalman::UnscentedKalmanFilterBase's MeasurementModelType template parameter.
    // This template parameter is typically expected to be a class that provides:
    // 1. A 'h' method for measurement prediction.
    // 2. A 'getCovarianceSquareRoot' method for measurement noise.
    // We use Kalman::Vector and Kalman::Matrix for consistency with the Kalman library.
    template<int MeasurementDim, int StateDim>
    class MeasurementModelBase {
    public:
        using MeasurementType = Kalman::Vector<double, MeasurementDim>;
        using StateType = Kalman::Vector<double, StateDim>;
        using CovarianceMatrix = Kalman::Matrix<double, MeasurementDim, MeasurementDim>;
        using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<MeasurementType>;

        // Pure virtual function for the measurement prediction function h(x)
        // Here, 'v' is measurement noise (implicitly handled by SRUKF)
        virtual MeasurementType h(const StateType& x) const = 0;

        // Pure virtual function to get the square root of the measurement noise covariance (S_R).
        // This version is typically used by SRUKF, which expects a constant S_R.
        virtual CovarianceSquareRootType getCovarianceSquareRoot() const = 0;

        // Optional: Virtual function to get the square root of the measurement noise covariance (S_R)
        // that might be time-step dependent. This is more common for process noise (Q),
        // but provided here for flexibility if a measurement noise model needs it (e.g., for EKF).
        // By default, it returns the current S_R, as measurement noise is usually
        // independent of time step, but dependent on sensor context (SNR, range).
        virtual CovarianceSquareRootType getCovarianceSquareRoot(double dt) const {
            (void)dt; // Suppress unused parameter warning
            // By default, measurement noise R is not directly time-step dependent.
            // It's usually adapted based on sensor context (SNR, range, etc.) via setMeasurementNoiseFromContext.
            return getCovarianceSquareRoot();
        }

        // Virtual method to set the full measurement noise covariance matrix (R)
        // This method will compute and store the Cholesky decomposition (S_R).
        // Default implementation throws an error if not overridden.
        virtual void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) {
            throw std::runtime_error("setFullMeasurementNoiseCovariance not implemented for this model.");
        }

        // Virtual method to dynamically adjust measurement noise based on sensor context.
        // This allows for SNR-based, range-based, or weather-based noise scaling.
        // Derived classes should override this to implement specific adaptation logic.
        virtual void setMeasurementNoiseFromContext(const struct SensorContext& ctx) {
            // Default implementation does nothing or throws, derived classes will implement.
            (void)ctx; // Suppress unused parameter warning
            throw std::runtime_error("setMeasurementNoiseFromContext not implemented for this model.");
        }

        // Pure virtual methods for dimension introspection
        virtual int getMeasurementDim() const = 0;
        virtual int getStateDim() const = 0;

        // Optional: Virtual method to compute the Jacobian of h(x) with respect to x.
        // This is primarily useful for Extended Kalman Filters (EKF) and is not strictly
        // required for SRUKF, but allows for model reuse.
        // Changed return type to fixed-size Kalman::Matrix to avoid dynamic allocation.
        virtual Kalman::Matrix<double, MeasurementDim, StateDim> computeJacobian(const StateType& x) const {
            (void)x; // Suppress unused parameter warning
            throw std::runtime_error("computeJacobian not implemented for this model. Only needed for EKF. Return type is Kalman::Matrix<double, MeasurementDim, StateDim>.");
        }

        // Optional: Virtual method for sensor-specific clutter rejection logic.
        // This would be called by the MeasurementAdapter before passing data to the filter.
        // Parameters order changed to match comment: raw_measurement, predicted_state.
        virtual bool applyClutterRejection(const MeasurementType& raw_measurement, const StateType& predicted_state) const {
            (void)raw_measurement; // Suppress unused parameter warning
            (void)predicted_state; // Suppress unused parameter warning
            // Default: No clutter rejection implemented at this level.
            // Derived classes can override for specific sensor types (e.g., Doppler filtering for radar).
            return true; // Assume measurement is not clutter by default
        }

        // Virtual method to get the model name for debugging/logging
        virtual const char* getModelName() const = 0;

        // Virtual destructor for proper polymorphic cleanup
        virtual ~MeasurementModelBase() = default;

    protected:
        /**
         * @brief Helper function to convert 3D Cartesian coordinates to Azimuth and Elevation.
         * This reduces code duplication across multiple measurement models.
         * @param px X-coordinate in Cartesian.
         * @param py Y-coordinate in Cartesian.
         * @param pz Z-coordinate in Cartesian.
         * @return An Eigen::Vector2d containing [azimuth, elevation].
         */
        static Eigen::Vector2d cartesianToAzimuthElevation(double px, double py, double pz) {
            Eigen::Vector2d az_el;

            // Calculate azimuth (bearing) and wrap to [-PI, PI)
            double horizontal_distance_xy = std::hypot(px, py);
            if (horizontal_distance_xy < 1e-6) { // If horizontal distance is near zero
                az_el(0) = 0.0; // Azimuth is undefined, set to a nominal value
            } else {
                az_el(0) = wrapAngleRad(std::atan2(py, px));
            }

            // Calculate elevation
            double range = std::hypot(px, py, pz);
            if (range < 1e-6) { // If range is near zero, target is at sensor origin
                az_el(1) = 0.0; // Elevation is undefined, set to nominal
            } else {
                az_el(1) = std::atan2(pz, horizontal_distance_xy);
            }
            return az_el;
        }
    };
} // namespace Kalman

/**
 * @brief Utility function to wrap an angle to (-PI, PI] range.
 *
 * This ensures that angles are consistently within a single revolution.
 * It's particularly useful for handling angle discontinuities in measurements
 * or residuals in the filter's update step.
 *
 * @param theta The angle in radians.
 * @return The wrapped angle in (-PI, PI].
 */
inline double wrapAngleRad(double theta) {
    // Using fmod first can be more efficient for very large angles,
    // then handle the edge cases with while loops.
    theta = std::fmod(theta, 2 * M_PI);
    while (theta <= -M_PI) theta += 2 * M_PI;
    while (theta > M_PI)    theta -= 2 * M_PI;
    return theta;
}

/**
 * @brief Radar Measurement Model for 3D Position (Cartesian).
 *
 * This model predicts 3D position measurements (x, y, z) from the filter's state.
 * It assumes the state vector contains position components at indices 0, 1, 2.
 *
 * Measurement vector: [px, py, pz]^T (3 dimensions)
 * State vector: Can be any dimension, as long as it contains 3D position.
 * Expected State Layout:
 * - Position: x (index 0), y (index 1), z (index 2)
 * - Other state components follow (e.g., velocity, acceleration)
 *
 * @tparam StateDim The dimension of the state vector (e.g., 6 for CV, 7 for CT, 9 for CA/Singer).
 */
template<int StateDim>
class RadarPositionMeasurementModel : public Kalman::MeasurementModelBase<3, StateDim> { // 3D measurement, flexible StateDim
public:
    // Define the specific MeasurementType, StateType, etc. for this model
    static const int MeasurementDim = 3; // Explicitly define MeasurementDim for consistency
    using MeasurementType = Kalman::Vector<double, MeasurementDim>;
    using StateType = Kalman::Vector<double, StateDim>;
    using CovarianceMatrix = Kalman::Matrix<double, MeasurementDim, MeasurementDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<MeasurementType>;

    /**
     * @brief Constructor for the Radar Position Measurement Model.
     * @param position_measurement_noise_std_dev_initial Initial standard deviation for the measurement noise
     * for each position component (x, y, z). This is a baseline. Assumes isotropic noise.
     */
    explicit RadarPositionMeasurementModel(double position_measurement_noise_std_dev_initial)
        : position_noise_std_dev_base_(position_measurement_noise_std_dev_initial),
          initial_position_noise_std_dev_base_(position_measurement_noise_std_dev_initial) // Store initial for adaptation
    {
        // Initialize the square root of the measurement noise covariance matrix (S_R) with baseline.
        s_r_.setIdentity(); // Initialize as identity
        s_r_(0,0) = position_noise_std_dev_base_; // px noise
        s_r_(1,1) = position_noise_std_dev_base_; // py noise
        s_r_(2,2) = position_noise_std_dev_base_; // pz noise
    }

    /**
     * @brief Implements the measurement prediction function for the Radar Position model.
     *
     * This function extracts the 3D position (x, y, z) from the filter's state vector.
     * It assumes that the position components are located at indices 0, 1, and 2
     * of the state vector.
     *
     * @param x The current state vector (e.g., [x, y, z, vx, vy, vz, ...]^T).
     * @return The predicted measurement vector [px, py, pz]^T.
     */
    MeasurementType h(const StateType& x) const override {
        // The measurement is simply the position components from the state.
        // Eigen's segment<N>(offset) extracts a sub-vector of size N starting at offset.
        // Assuming Kalman::Vector is an Eigen::Vector or has a compatible segment method.
        return x.template segment<3>(0);
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R).
     * This version is typically used by SRUKF, which expects a constant S_R.
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_r_;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R)
     * that might be time-step dependent.
     * @param dt The time step.
     * @return The S_R matrix.
     *
     * @note For this model, measurement noise R is not directly time-step dependent.
     * It's usually adapted based on sensor context (SNR, range, etc.) via setMeasurementNoiseFromContext.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        (void)dt; // Suppress unused parameter warning
        return s_r_;
    }

    /**
     * @brief Sets the measurement noise standard deviation for the position components.
     * This allows for dynamic adjustment of the measurement noise.
     * @param new_sigma The new standard deviation for position measurement noise.
     *
     * @note This method is used by higher-level adaptive logic (e.g., in MeasurementAdapter
     * or OnlineNoiseReEstimator) to dynamically adjust the measurement uncertainty
     * based on factors like SNR, range, or environmental clutter.
     */
    void setMeasurementNoise(double new_sigma) {
        // Update the internal base value if needed, or just directly update s_r_
        // For simplicity, we'll directly update s_r_ here.
        s_r_(0,0) = new_sigma;
        s_r_(1,1) = new_sigma;
        s_r_(2,2) = new_sigma;
    }

    /**
     * @brief Sets the full measurement noise covariance matrix (R) and computes its Cholesky decomposition (S_R).
     * @param full_R_matrix The full, positive semi-definite R matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     *
     * @note This method is used by higher-level adaptive logic (e.g., in MeasurementAdapter
     * or OnlineNoiseReEstimator) to dynamically adjust the measurement uncertainty
     * based on factors like SNR, range, or environmental clutter.
     */
    void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override {
        // Ensure the matrix is symmetric before Cholesky decomposition for numerical stability
        CovarianceMatrix sym_R = (full_R_matrix + full_R_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_R);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("RadarPositionMeasurementModel: Input R matrix is not positive definite or symmetric.");
        }
        s_r_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Dynamically adjusts the measurement noise (S_R) based on sensor context.
     * @param ctx The SensorContext containing dynamic sensor information.
     *
     * @note This is a heuristic example. Real-world adaptation functions can be complex
     * and should be thoroughly tested for stability, especially with extreme noise changes.
     */
    void setMeasurementNoiseFromContext(const SensorContext& ctx) override {
        // Example adaptation logic:
        // Lower SNR means higher noise. Higher range means higher noise.
        double snr_factor = 1.0;
        if (ctx.snr_db < 10.0) { // Below 10 dB, start increasing noise significantly
            snr_factor = 1.0 + (10.0 - ctx.snr_db) * 0.1; // Heuristic
        } else if (ctx.snr_db > 20.0) { // Very good SNR, slight reduction
             snr_factor = 1.0 - (ctx.snr_db - 20.0) * 0.01;
        }
        snr_factor = std::max(0.5, snr_factor); // Don't reduce noise too much

        double range_factor = 1.0 + ctx.range_m * 0.001; // Noise increases with range

        double adapted_sigma = initial_position_noise_std_dev_base_ * snr_factor * range_factor; // Use initial base for adaptation
        setMeasurementNoise(adapted_sigma); // Apply the calculated noise
    }

    /**
     * @brief Returns the dimension of the measurement vector.
     */
    int getMeasurementDim() const override {
        return MeasurementDim; // Using the static const int
    }

    /**
     * @brief Returns the dimension of the state vector this model expects.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the measurement model.
     */
    const char* getModelName() const override {
        return "RadarPositionMeasurementModel"; // Consistent naming
    }

private:
    double position_noise_std_dev_base_; // Current standard deviation for position measurement noise
    double initial_position_noise_std_dev_base_; // Store initial for adaptation
    CovarianceSquareRootType s_r_; // Square root of the measurement noise covariance matrix (S_R)
};

/**
 * @brief Radar Measurement Model for Range, Bearing, and Elevation.
 *
 * This model predicts polar measurements (range, azimuth, elevation) from the filter's state.
 * It assumes the state vector contains position components at indices 0, 1, 2.
 *
 * Measurement vector: [range, azimuth, elevation]^T (3 dimensions)
 * State vector: Can be any dimension, as long as it contains 3D position.
 * Expected State Layout:
 * - Position: x (index 0), y (index 1), z (index 2)
 * - Other state components follow (e.g., velocity, acceleration)
 *
 * @tparam StateDim The dimension of the state vector (e.g., 6 for CV, 7 for CT, 9 for CA/Singer).
 */
template<int StateDim>
class RadarRangeBearingElevationMeasurementModel : public Kalman::MeasurementModelBase<3, StateDim> {
public:
    // Define the specific MeasurementType, StateType, etc. for this model
    static const int MeasurementDim = 3; // Explicitly define MeasurementDim for consistency
    using MeasurementType = Kalman::Vector<double, MeasurementDim>;
    using StateType = Kalman::Vector<double, StateDim>;
    using CovarianceMatrix = Kalman::Matrix<double, MeasurementDim, MeasurementDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<MeasurementType>;

    /**
     * @brief Constructor for the Radar Range-Bearing-Elevation Model.
     * @param range_noise_std_dev_initial Initial standard deviation for range measurement noise.
     * @param bearing_noise_std_dev_initial Initial standard deviation for bearing (azimuth) measurement noise (radians).
     * @param elevation_noise_std_dev_initial Initial standard deviation for elevation measurement noise (radians).
     */
    RadarRangeBearingElevationMeasurementModel(double range_noise_std_dev_initial,
                                               double bearing_noise_std_dev_initial,
                                               double elevation_noise_std_dev_initial)
        : range_noise_std_dev_base_(range_noise_std_dev_initial),
          bearing_noise_std_dev_base_(bearing_noise_std_dev_initial),
          elevation_noise_std_dev_base_(elevation_noise_std_dev_initial),
          initial_range_noise_std_dev_base_(range_noise_std_dev_initial), // Store initial for adaptation
          initial_bearing_noise_std_dev_base_(bearing_noise_std_dev_initial),
          initial_elevation_noise_std_dev_base_(elevation_noise_std_dev_initial)
    {
        // Initialize the square root of the measurement noise covariance matrix (S_R).
        // S_R is a 3x3 diagonal matrix.
        s_r_.setIdentity();
        s_r_(0,0) = range_noise_std_dev_base_;      // Range noise
        s_r_(1,1) = bearing_noise_std_dev_base_;    // Azimuth noise (radians)
        s_r_(2,2) = elevation_noise_std_dev_base_; // Elevation noise (radians)
    }

    /**
     * @brief Implements the measurement prediction function for the Radar Range-Bearing-Elevation model.
     *
     * This function converts the 3D Cartesian position (x, y, z) from the filter's
     * state vector into polar coordinates (range, azimuth, elevation).
     * It assumes that the position components are located at indices 0, 1, and 2
     * of the state vector.
     *
     * @param x The current state vector (e.g., [x, y, z, vx, vy, vz, ...]^T).
     * @return The predicted measurement vector [range, azimuth, elevation]^T.
     */
    MeasurementType h(const StateType& x) const override {
        MeasurementType predicted_measurement;

        double px = x(0);
        double py = x(1);
        double pz = x(2);

        // Calculate range
        double range = std::hypot(px, py, pz); // sqrt(px^2 + py^2 + pz^2)
        predicted_measurement(0) = range;

        // Use the helper function for azimuth and elevation
        Eigen::Vector2d az_el = Kalman::MeasurementModelBase<MeasurementDim, StateDim>::cartesianToAzimuthElevation(px, py, pz);
        predicted_measurement(1) = az_el(0); // Azimuth
        predicted_measurement(2) = az_el(1); // Elevation

        return predicted_measurement;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R).
     * This version is typically used by SRUKF, which expects a constant S_R.
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_r_;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R)
     * that might be time-step dependent.
     * @param dt The time step.
     * @return The S_R matrix.
     *
     * @note For this model, measurement noise R is not directly time-step dependent.
     * It's usually adapted based on sensor context (SNR, range, etc.) via setMeasurementNoiseFromContext.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        (void)dt; // Suppress unused parameter warning
        return s_r_;
    }

    /**
     * @brief Sets the measurement noise standard deviations for range, bearing, and elevation.
     * This allows for dynamic adjustment of the measurement noise.
     * @param new_range_sigma The new standard deviation for range noise.
     * @param new_bearing_sigma The new standard deviation for bearing noise.
     * @param new_elevation_sigma The new standard deviation for elevation noise.
     *
     * @note This method is used by higher-level adaptive logic (e.g., in MeasurementAdapter
     * or OnlineNoiseReEstimator) to dynamically adjust the measurement uncertainty
     * based on factors like SNR, range, or environmental clutter.
     */
    void setMeasurementNoise(double new_range_sigma, double new_bearing_sigma, double new_elevation_sigma) {
        s_r_(0,0) = new_range_sigma;
        s_r_(1,1) = new_bearing_sigma;
        s_r_(2,2) = new_elevation_sigma;
    }

    /**
     * @brief Sets the full measurement noise covariance matrix (R) and computes its Cholesky decomposition (S_R).
     * @param full_R_matrix The full, positive semi-definite R matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     *
     * @note This method is used by higher-level adaptive logic (e.g., in MeasurementAdapter
     * or OnlineNoiseReEstimator) to dynamically adjust the measurement uncertainty
     * based on factors like SNR, range, or environmental clutter.
     */
    void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override {
        // Ensure the matrix is symmetric before Cholesky decomposition for numerical stability
        CovarianceMatrix sym_R = (full_R_matrix + full_R_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_R);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("RadarRangeBearingElevationMeasurementModel: Input R matrix is not positive definite or symmetric.");
        }
        s_r_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Dynamically adjusts the measurement noise (S_R) based on sensor context.
     * @param ctx The SensorContext containing dynamic sensor information.
     *
     * @note This is a heuristic example. Real-world adaptation functions can be complex.
     */
    void setMeasurementNoiseFromContext(const SensorContext& ctx) override {
        // Example adaptation logic:
        // Lower SNR means higher noise. Higher range means higher noise.
        double snr_factor = 1.0;
        if (ctx.snr_db < 10.0) {
            snr_factor = 1.0 + (10.0 - ctx.snr_db) * 0.1;
        } else if (ctx.snr_db > 20.0) {
             snr_factor = 1.0 - (ctx.snr_db - 20.0) * 0.01;
        }
        snr_factor = std::max(0.5, snr_factor);

        double range_factor = 1.0 + ctx.range_m * 0.001; // Noise increases with range

        // Apply factors to base standard deviations
        double adapted_range_sigma = initial_range_noise_std_dev_base_ * snr_factor * range_factor;
        double adapted_bearing_sigma = initial_bearing_noise_std_dev_base_ * snr_factor * range_factor;
        double adapted_elevation_sigma = initial_elevation_noise_std_dev_base_ * snr_factor * range_factor;

        setMeasurementNoise(adapted_range_sigma, adapted_bearing_sigma, adapted_elevation_sigma);
    }

    /**
     * @brief Returns the dimension of the measurement vector.
     */
    int getMeasurementDim() const override {
        return MeasurementDim;
    }

    /**
     * @brief Returns the dimension of the state vector this model expects.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the measurement model.
     */
    const char* getModelName() const override {
        return "RadarRangeBearingElevationMeasurementModel"; // Consistent naming
    }

private:
    double range_noise_std_dev_base_;
    double bearing_noise_std_dev_base_;
    double elevation_noise_std_dev_base_;
    double initial_range_noise_std_dev_base_; // Store initial for adaptation
    double initial_bearing_noise_std_dev_base_;
    double initial_elevation_noise_std_dev_base_;
    CovarianceSquareRootType s_r_; // Square root of the measurement noise covariance matrix (S_R)
};

/**
 * @brief RF Measurement Model for Angle of Arrival (AoA) and Received Signal Strength (RSSI).
 *
 * This model predicts Angle of Arrival (azimuth, elevation) and RSSI from the filter's state.
 * It assumes the state vector contains position components at indices 0, 1, 2, and
 * an augmented state for RF signal power (e.g., RF_Power_dBm) at a specific index.
 *
 * Measurement vector: [azimuth, elevation, RSSI]^T (3 dimensions)
 * State vector: Must contain 3D position and an RF_Power_dBm component.
 * Expected State Layout:
 * - Position: x (index 0), y (index 1), z (index 2)
 * - RF Power (dBm): at `RfPowerStateIdx`
 * - Other state components follow (e.g., velocity, acceleration)
 *
 * @tparam StateDim The dimension of the state vector.
 * @tparam RfPowerStateIdx The index in the state vector where RF_Power_dBm is stored.
 */
template<int StateDim, int RfPowerStateIdx>
class RFMeasurementModel : public Kalman::MeasurementModelBase<3, StateDim> {
    // Compile-time check to ensure RfPowerStateIdx is within StateDim bounds
    static_assert(RfPowerStateIdx >= 0 && RfPowerStateIdx < StateDim, "RfPowerStateIdx must be within StateDim bounds.");

public:
    // Define the specific MeasurementType, StateType, etc. for this model
    static const int MeasurementDim = 3; // Explicitly define MeasurementDim for consistency
    using MeasurementType = Kalman::Vector<double, MeasurementDim>;
    using StateType = Kalman::Vector<double, StateDim>;
    using CovarianceMatrix = Kalman::Matrix<double, MeasurementDim, MeasurementDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<MeasurementType>;

    /**
     * @brief Constructor for the RF Measurement Model.
     * @param bearing_noise_std_dev_initial Initial standard deviation for bearing (azimuth) measurement noise (radians).
     * @param elevation_noise_std_dev_initial Initial standard deviation for elevation measurement noise (radians).
     * @param rssi_noise_std_dev_initial Initial standard deviation for RSSI measurement noise (dBm).
     * @param antenna_baseline_m Optional: Baseline distance for multi-antenna setup (meters).
     * @param path_loss_exponent Optional: Environmental path loss exponent for RSSI prediction.
     * This parameter can influence the assumed noise characteristics.
     *
     * @note For multi-antenna AoA estimation, this model assumes the sensor internally
     * processes multiple antenna readings to produce a single, more accurate AoA.
     * The `antenna_baseline_m` can be used to inform the noise characteristics (R matrix)
     * if the sensor's accuracy improves with baseline. Full MIMO-style triangulation
     * involving multiple distinct bearings from different baselines is typically
     * handled at a higher-level data association or fusion layer, where multiple
     * `RFMeasurementModel` instances (or their outputs) would be combined.
     */
    RFMeasurementModel(double bearing_noise_std_dev_initial,
                       double elevation_noise_std_dev_initial,
                       double rssi_noise_std_dev_initial,
                       double antenna_baseline_m = 0.0, // Default to 0 if not multi-antenna
                       double path_loss_exponent = 2.0) // Default to free space path loss
        : bearing_noise_std_dev_base_(bearing_noise_std_dev_initial),
          elevation_noise_std_dev_base_(elevation_noise_std_dev_initial),
          rssi_noise_std_dev_base_(rssi_noise_std_dev_initial),
          initial_bearing_noise_std_dev_base_(bearing_noise_std_dev_initial), // Store initial for adaptation
          initial_elevation_noise_std_dev_base_(elevation_noise_std_dev_initial),
          initial_rssi_noise_std_dev_base_(rssi_noise_std_dev_initial),
          antenna_baseline_m_(antenna_baseline_m),
          path_loss_exponent_(path_loss_exponent) {
        // Initialize the square root of the measurement noise covariance matrix (S_R).
        // S_R is a 3x3 diagonal matrix.
        s_r_.setIdentity();
        s_r_(0,0) = bearing_noise_std_dev_base_;    // Azimuth noise (radians)
        s_r_(1,1) = elevation_noise_std_dev_base_; // Elevation noise (radians)
        s_r_(2,2) = rssi_noise_std_dev_base_;      // RSSI noise (dBm)

        // Optionally, adjust noise based on antenna_baseline_m if it implies better accuracy
        // For example, larger baseline might reduce angular noise:
        // if (antenna_baseline_m_ > 0.1) { // Arbitrary threshold
        //     s_r_(0,0) *= (1.0 / (1.0 + antenna_baseline_m_ / 10.0)); // Heuristic reduction
        //     s_r_(1,1) *= (1.0 / (1.0 + antenna_baseline_m_ / 10.0));
        // }
    }

    /**
     * @brief Implements the measurement prediction function for the RF model.
     *
     * This function predicts Angle of Arrival (azimuth, elevation) from the 3D Cartesian
     * position (x, y, z) in the state, and predicts RSSI from the augmented RF power state.
     * It assumes position is at indices 0, 1, 2 and RF_Power_dBm at RfPowerStateIdx.
     *
     * @param x The current state vector (e.g., [x, y, z, ..., RF_Power_dBm, ...]^T).
     * @return The predicted measurement vector [azimuth, elevation, RSSI]^T.
     */
    MeasurementType h(const StateType& x) const override {
        MeasurementType predicted_measurement;

        double px = x(0);
        double py = x(1);
        double pz = x(2);
        double rf_power_dbm_at_source = x(RfPowerStateIdx); // Augmented state for RF power

        // Use the helper function for azimuth and elevation
        Eigen::Vector2d az_el = Kalman::MeasurementModelBase<MeasurementDim, StateDim>::cartesianToAzimuthElevation(px, py, pz);
        predicted_measurement(0) = az_el(0); // Azimuth
        predicted_measurement(1) = az_el(1); // Elevation

        // Predict RSSI using a log-distance path loss model.
        // The '+ 1.0' in log10(range + 1.0) is a small offset to prevent log(0) if range is exactly 0.
        // The path_loss_exponent_ can be parameterized via environmental constant for urban clutter etc.
        double range = std::hypot(px,py,pz);
        predicted_measurement(2) = rf_power_dbm_at_source - 10.0 * path_loss_exponent_ * std::log10(range + 1.0);

        return predicted_measurement;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R).
     * This version is typically used by SRUKF, which expects a constant S_R.
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_r_;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R)
     * that might be time-step dependent.
     * @param dt The time step.
     * @return The S_R matrix.
     *
     * @note For this model, measurement noise R is not directly time-step dependent.
     * It's usually adapted based on sensor context (SNR, range, etc.) via setMeasurementNoiseFromContext.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        (void)dt; // Suppress unused parameter warning
        return s_r_;
    }

    /**
     * @brief Sets the measurement noise standard deviations for bearing, elevation, and RSSI.
     * This allows for dynamic adjustment of the measurement noise.
     * @param new_bearing_sigma The new standard deviation for bearing noise.
     * @param new_elevation_sigma The new standard deviation for elevation noise.
     * @param new_rssi_sigma The new standard deviation for RSSI noise.
     *
     * @note This method is used by higher-level adaptive logic (e.g., in MeasurementAdapter
     * or OnlineNoiseReEstimator) to dynamically adjust the measurement uncertainty
     * based on factors like SNR, range, or environmental clutter.
     */
    void setMeasurementNoise(double new_bearing_sigma, double new_elevation_sigma, double new_rssi_sigma) {
        s_r_(0,0) = new_bearing_sigma;
        s_r_(1,1) = new_elevation_sigma;
        s_r_(2,2) = new_rssi_sigma;
    }

    /**
     * @brief Sets the full measurement noise covariance matrix (R) and computes its Cholesky decomposition (S_R).
     * @param full_R_matrix The full, positive semi-definite R matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     *
     * @note This method is used by higher-level adaptive logic (e.g., in MeasurementAdapter
     * or OnlineNoiseReEstimator) to dynamically adjust the measurement uncertainty
     * based on factors like SNR, range, or environmental clutter.
     */
    void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override {
        // Ensure the matrix is symmetric before Cholesky decomposition for numerical stability
        CovarianceMatrix sym_R = (full_R_matrix + full_R_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_R);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("RFMeasurementModel: Input R matrix is not positive definite or symmetric.");
        }
        s_r_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Dynamically adjusts the measurement noise (S_R) based on sensor context.
     * @param ctx The SensorContext containing dynamic sensor information.
     *
     * @note This is a heuristic example. Real-world adaptation functions can be complex.
     * Path-loss exponent adaptation based on environmental context could be a future enhancement.
     */
    void setMeasurementNoiseFromContext(const SensorContext& ctx) override {
        // Example adaptation logic:
        // Lower SNR means higher noise. Higher range means higher noise.
        // Higher clutter level means higher RSSI noise.
        double snr_factor = 1.0;
        if (ctx.snr_db < 10.0) {
            snr_factor = 1.0 + (10.0 - ctx.snr_db) * 0.1;
        } else if (ctx.snr_db > 20.0) {
             snr_factor = 1.0 - (ctx.snr_db - 20.0) * 0.01;
        }
        snr_factor = std::max(0.5, snr_factor);

        double range_factor = 1.0 + ctx.range_m * 0.001; // Noise increases with range

        double clutter_factor = 1.0 + ctx.clutter_level * 0.5; // Higher clutter -> higher RSSI noise

        // Apply factors to base standard deviations
        double adapted_bearing_sigma = initial_bearing_noise_std_dev_base_ * snr_factor * range_factor;
        double adapted_elevation_sigma = initial_elevation_noise_std_dev_base_ * snr_factor * range_factor;
        double adapted_rssi_sigma = initial_rssi_noise_std_dev_base_ * snr_factor * range_factor * clutter_factor;

        setMeasurementNoise(adapted_bearing_sigma, adapted_elevation_sigma, adapted_rssi_sigma);
    }

    /**
     * @brief Returns the dimension of the measurement vector.
     */
    int getMeasurementDim() const override {
        return MeasurementDim;
    }

    /**
     * @brief Returns the dimension of the state vector this model expects.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the measurement model.
     */
    const char* getModelName() const override {
        return "RFMeasurementModel"; // Consistent naming
    }

private:
    double bearing_noise_std_dev_base_;
    double elevation_noise_std_dev_base_;
    double rssi_noise_std_dev_base_;
    double initial_bearing_noise_std_dev_base_; // Store initial for adaptation
    double initial_elevation_noise_std_dev_base_;
    double initial_rssi_noise_std_dev_base_;
    double antenna_baseline_m_; // For potential future noise adaptation based on baseline
    double path_loss_exponent_; // For RSSI prediction
    CovarianceSquareRootType s_r_; // Square root of the measurement noise covariance matrix (S_R)
};

/**
 * @brief IMU Measurement Model for Accelerations and Angular Velocities.
 *
 * This model predicts IMU measurements (linear accelerations and angular velocities)
 * from the filter's state. It assumes the state vector contains acceleration components
 * and optionally angular velocity (gyro) components at specific indices.
 *
 * Measurement vector: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]^T (6 dimensions)
 * State vector: Must contain 3D acceleration, and optionally 3D angular velocity.
 * Expected State Layout:
 * - Position: x (index 0), y (index 1), z (index 2)
 * - Velocity: vx (index 3), vy (index 4), vz (index 5)
 * - Acceleration: ax (at `AccelStateIdx`), ay (at `AccelStateIdx + 1`), az (at `AccelStateIdx + 2`)
 * - Angular Velocity (Gyro): gx (at `GyroStateIdx`), gy (at `GyroStateIdx + 1`), gz (at `GyroStateIdx + 2`)
 * - Other state components follow
 *
 * @tparam StateDim The dimension of the state vector.
 * @tparam AccelStateIdx The starting index in the state vector where ax, ay, az are stored.
 * @tparam GyroStateIdx The starting index in the state vector where gx, gy, gz are stored.
 * @note If the state does not include angular velocity, GyroStateIdx can be set to StateDim
 * or greater. In this case, the corresponding measurement components will be predicted as zero
 * and their noise should be set to a very high value to effectively ignore them in the filter.
 */
template<int StateDim, int AccelStateIdx, int GyroStateIdx>
class IMUMeasurementModel : public Kalman::MeasurementModelBase<6, StateDim> {
    // Compile-time checks for state indices
    static_assert(AccelStateIdx >= 0 && AccelStateIdx + 2 < StateDim, "AccelStateIdx must be within StateDim bounds.");
    // No static_assert for GyroStateIdx, as it can be intentionally out of bounds if gyro is not in state.

public:
    // Define the specific MeasurementType, StateType, etc. for this model
    static const int MeasurementDim = 6; // Explicitly define MeasurementDim for consistency
    using MeasurementType = Kalman::Vector<double, MeasurementDim>;
    using StateType = Kalman::Vector<double, StateDim>;
    using CovarianceMatrix = Kalman::Matrix<double, MeasurementDim, MeasurementDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<MeasurementType>;

    /**
     * @brief Constructor for the IMU Measurement Model.
     * @param accel_noise_std_dev_initial Initial standard deviation for accelerometer noise (m/s^2).
     * @param gyro_noise_std_dev_initial Initial standard deviation for gyroscope noise (rad/s).
     * @param gravity_z The gravitational acceleration component along the Z-axis (m/s^2).
     * @param accel_bias_std_dev_initial Optional initial standard deviation for accelerometer bias.
     * @param gyro_bias_std_dev_initial Optional initial standard deviation for gyroscope bias.
     *
     * @note This model assumes the IMU is aligned with the global coordinate system, and
     * that the state's acceleration components (ax, ay, az) represent *non-gravitational*
     * (body-frame or compensated) accelerations. Gravity is added to the predicted Z-acceleration.
     * If the IMU can rotate, a more complex model involving orientation (e.g., quaternions)
     * in the state and coordinate transformations in h() would be required.
     *
     * @note The `accel_bias_std_dev_initial` and `gyro_bias_std_dev_initial` parameters
     * represent the *initial uncertainty* of potential IMU biases. They are used to
     * configure the measurement noise covariance (S_R) or to initialize the covariance
     * of augmented bias states if the filter explicitly estimates biases. They are not
     * directly subtracted in the `h()` function, as this model assumes the state's
     * accelerations are already the true, bias-compensated values.
     */
    IMUMeasurementModel(double accel_noise_std_dev_initial,
                        double gyro_noise_std_dev_initial,
                        double gravity_z,
                        double accel_bias_std_dev_initial = 0.0, // Initial bias uncertainty
                        double gyro_bias_std_dev_initial = 0.0)
        : accel_noise_std_dev_base_(accel_noise_std_dev_initial),
          gyro_noise_std_dev_base_(gyro_noise_std_dev_initial),
          initial_accel_noise_std_dev_base_(accel_noise_std_dev_initial), // Store initial for adaptation
          initial_gyro_noise_std_dev_base_(gyro_noise_std_dev_initial),
          gravity_z_(gravity_z),
          accel_bias_std_dev_base_(accel_bias_std_dev_initial),
          gyro_bias_std_dev_base_(gyro_bias_std_dev_initial)
    {
        // Initialize the square root of the measurement noise covariance matrix (S_R).
        // S_R is a 6x6 diagonal matrix.
        s_r_.setIdentity();
        s_r_(0,0) = accel_noise_std_dev_base_; // Accel X noise
        s_r_(1,1) = accel_noise_std_dev_base_; // Accel Y noise
        s_r_(2,2) = accel_noise_std_dev_base_; // Accel Z noise
        s_r_(3,3) = gyro_noise_std_dev_base_;  // Gyro X noise
        s_r_(4,4) = gyro_noise_std_dev_base_;  // Gyro Y noise
        s_r_(5,5) = gyro_noise_std_dev_base_;  // Gyro Z noise
    }

    /**
     * @brief Implements the measurement prediction function for the IMU model.
     *
     * This function predicts IMU measurements (accelerations and angular velocities)
     * from the filter's state. It adds gravity to the predicted Z-acceleration.
     *
     * @param x The current state vector (e.g., [..., ax, ay, az, gx, gy, gz, ...]^T).
     * @return The predicted measurement vector [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]^T.
     */
    MeasurementType h(const StateType& x) const override {
        MeasurementType predicted_measurement;
        predicted_measurement.setZero(); // Initialize to zero

        // Predict linear accelerations
        // Assumes state contains acceleration at AccelStateIdx
        predicted_measurement(0) = x(AccelStateIdx);
        predicted_measurement(1) = x(AccelStateIdx + 1);
        predicted_measurement(2) = x(AccelStateIdx + 2) + gravity_z_; // Add gravity to Z-acceleration

        // Predict angular velocities (gyro)
        // Check if GyroStateIdx is within bounds of the state vector
        if (GyroStateIdx >= 0 && GyroStateIdx + 2 < StateDim) {
            predicted_measurement(3) = x(GyroStateIdx);
            predicted_measurement(4) = x(GyroStateIdx + 1);
            predicted_measurement(5) = x(GyroStateIdx + 2);
        } else {
            // If angular velocity is not in the state, predict zero angular velocity.
            // The noise for these components should be set very high in R if they are not estimated.
            predicted_measurement(3) = 0.0;
            predicted_measurement(4) = 0.0;
            predicted_measurement(5) = 0.0;
        }

        return predicted_measurement;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R).
     * This version is typically used by SRUKF, which expects a constant S_R.
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_r_;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R)
     * that might be time-step dependent.
     * @param dt The time step.
     * @return The S_R matrix.
     *
     * @note For this model, measurement noise R is not directly time-step dependent.
     * It's usually adapted based on sensor context (e.g., temperature, vibration).
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        (void)dt; // Suppress unused parameter warning
        return s_r_;
    }

    /**
     * @brief Sets the measurement noise standard deviations for accelerometer and gyroscope.
     * @param new_accel_sigma The new standard deviation for accelerometer noise.
     * @param new_gyro_sigma The new standard deviation for gyroscope noise.
     */
    void setMeasurementNoise(double new_accel_sigma, double new_gyro_sigma) {
        s_r_(0,0) = new_accel_sigma;
        s_r_(1,1) = new_accel_sigma;
        s_r_(2,2) = new_accel_sigma;
        s_r_(3,3) = new_gyro_sigma;
        s_r_(4,4) = new_gyro_sigma;
        s_r_(5,5) = new_gyro_sigma;
    }

    /**
     * @brief Sets the full measurement noise covariance matrix (R) and computes its Cholesky decomposition (S_R).
     * @param full_R_matrix The full, positive semi-definite R matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     */
    void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override {
        // Ensure the matrix is symmetric before Cholesky decomposition for numerical stability
        CovarianceMatrix sym_R = (full_R_matrix + full_R_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_R);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("IMUMeasurementModel: Input R matrix is not positive definite or symmetric.");
        }
        s_r_ = llt.matrixL(); // Store the lower triangular Cholesky factor
    }

    /**
     * @brief Dynamically adjusts the measurement noise (S_R) based on sensor context.
     * @param ctx The SensorContext containing dynamic sensor information (e.g., vibration, temperature).
     *
     * @note This is a heuristic example. Real-world adaptation functions can be complex.
     */
    void setMeasurementNoiseFromContext(const SensorContext& ctx) override {
        // Example adaptation logic:
        // Higher ambient noise/vibration -> higher IMU noise
        double noise_factor = 1.0 + ctx.ambient_noise_db * 0.1; // Heuristic: 0.1 noise increase per dB of ambient noise
        noise_factor = std::max(1.0, noise_factor);

        double adapted_accel_sigma = initial_accel_noise_std_dev_base_ * noise_factor;
        double adapted_gyro_sigma = initial_gyro_noise_std_dev_base_ * noise_factor;

        setMeasurementNoise(adapted_accel_sigma, adapted_gyro_sigma);
    }

    /**
     * @brief Returns the dimension of the measurement vector.
     */
    int getMeasurementDim() const override {
        return MeasurementDim;
    }

    /**
     * @brief Returns the dimension of the state vector this model expects.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the measurement model.
     */
    const char* getModelName() const override {
        return "IMUMeasurementModel"; // Consistent naming
    }

private:
    double accel_noise_std_dev_base_;
    double gyro_noise_std_dev_base_;
    double initial_accel_noise_std_dev_base_; // Store initial for adaptation
    double initial_gyro_noise_std_dev_base_;
    double gravity_z_; // Constant for gravity compensation
    double accel_bias_std_dev_base_; // Placeholder for bias uncertainty (not used in h() directly yet)
    double gyro_bias_std_dev_base_;  // Placeholder for bias uncertainty (not used in h() directly yet)
    CovarianceSquareRootType s_r_; // Square root of the measurement noise covariance matrix (S_R)
};

/**
 * @brief Acoustic Measurement Model for Azimuth and Elevation.
 *
 * This model predicts Angle of Arrival (azimuth, elevation) from acoustic sensors.
 * It assumes the state vector contains position components at indices 0, 1, 2.
 *
 * Measurement vector: [azimuth, elevation]^T (2 dimensions)
 * State vector: Can be any dimension, as long as it contains 3D position.
 * Expected State Layout:
 * - Position: x (index 0), y (index 1), z (index 2)
 * - Other state components follow (e.g., velocity, acceleration)
 *
 * @tparam StateDim The dimension of the state vector.
 */
template<int StateDim>
class AcousticMeasurementModel : public Kalman::MeasurementModelBase<2, StateDim> {
public:
    // Define the specific MeasurementType, StateType, etc. for this model
    static const int MeasurementDim = 2; // Explicitly define MeasurementDim for consistency
    using MeasurementType = Kalman::Vector<double, MeasurementDim>;
    using StateType = Kalman::Vector<double, StateDim>;
    using CovarianceMatrix = Kalman::Matrix<double, MeasurementDim, MeasurementDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<MeasurementType>;

    /**
     * @brief Constructor for the Acoustic Measurement Model.
     * @param azimuth_noise_std_dev_initial Initial standard deviation for azimuth measurement noise (radians).
     * @param elevation_noise_std_dev_initial Initial standard deviation for elevation measurement noise (radians).
     */
    AcousticMeasurementModel(double azimuth_noise_std_dev_initial,
                             double elevation_noise_std_dev_initial)
        : azimuth_noise_std_dev_base_(azimuth_noise_std_dev_initial),
          elevation_noise_std_dev_base_(elevation_noise_std_dev_initial),
          initial_azimuth_noise_std_dev_base_(azimuth_noise_std_dev_initial), // Store initial for adaptation
          initial_elevation_noise_std_dev_base_(elevation_noise_std_dev_initial)
    {
        // Initialize the square root of the measurement noise covariance matrix (S_R).
        // S_R is a 2x2 diagonal matrix.
        s_r_.setIdentity();
        s_r_(0,0) = azimuth_noise_std_dev_base_;   // Azimuth noise
        s_r_(1,1) = elevation_noise_std_dev_base_; // Elevation noise
    }

    /**
     * @brief Implements the measurement prediction function for the Acoustic model.
     *
     * This function predicts Angle of Arrival (azimuth, elevation) from the 3D Cartesian
     * position (x, y, z) in the state.
     *
     * @param x The current state vector (e.g., [x, y, z, ...]^T).
     * @return The predicted measurement vector [azimuth, elevation]^T.
     */
    MeasurementType h(const StateType& x) const override {
        MeasurementType predicted_measurement;

        double px = x(0);
        double py = x(1);
        double pz = x(2);

        // Use the helper function for azimuth and elevation
        Eigen::Vector2d az_el = Kalman::MeasurementModelBase<MeasurementDim, StateDim>::cartesianToAzimuthElevation(px, py, pz);
        predicted_measurement(0) = az_el(0); // Azimuth
        predicted_measurement(1) = az_el(1); // Elevation

        return predicted_measurement;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R).
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_r_;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R)
     * that might be time-step dependent.
     * @param dt The time step.
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        (void)dt; // Suppress unused parameter warning
        return s_r_;
    }

    /**
     * @brief Sets the measurement noise standard deviations for azimuth and elevation.
     * @param new_azimuth_sigma The new standard deviation for azimuth noise.
     * @param new_elevation_sigma The new standard deviation for elevation noise.
     */
    void setMeasurementNoise(double new_azimuth_sigma, double new_elevation_sigma) {
        s_r_(0,0) = new_azimuth_sigma;
        s_r_(1,1) = new_elevation_sigma;
    }

    /**
     * @brief Sets the full measurement noise covariance matrix (R) and computes its Cholesky decomposition (S_R).
     * @param full_R_matrix The full, positive semi-definite R matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     */
    void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override {
        CovarianceMatrix sym_R = (full_R_matrix + full_R_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_R);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("AcousticMeasurementModel: Input R matrix is not positive definite or symmetric.");
        }
        s_r_ = llt.matrixL();
    }

    /**
     * @brief Dynamically adjusts the measurement noise (S_R) based on sensor context.
     * @param ctx The SensorContext containing dynamic sensor information (e.g., ambient noise).
     */
    void setMeasurementNoiseFromContext(const SensorContext& ctx) override {
        // Example adaptation logic:
        // Higher ambient noise -> higher acoustic noise
        double noise_factor = 1.0 + ctx.ambient_noise_db * 0.05; // Heuristic: 0.05 noise increase per dB of ambient noise
        // Clamp upper bound for stability
        noise_factor = std::min(5.0, std::max(1.0, noise_factor)); // Ensure factor is between 1.0 and 5.0

        double adapted_azimuth_sigma = initial_azimuth_noise_std_dev_base_ * noise_factor;
        double adapted_elevation_sigma = initial_elevation_noise_std_dev_base_ * noise_factor;

        setMeasurementNoise(adapted_azimuth_sigma, adapted_elevation_sigma);
    }

    /**
     * @brief Returns the dimension of the measurement vector.
     */
    int getMeasurementDim() const override {
        return MeasurementDim;
    }

    /**
     * @brief Returns the dimension of the state vector this model expects.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the measurement model.
     */
    const char* getModelName() const override {
        return "AcousticMeasurementModel"; // Consistent naming
    }

private:
    double azimuth_noise_std_dev_base_;
    double elevation_noise_std_dev_base_;
    double initial_azimuth_noise_std_dev_base_; // Store initial for adaptation
    double initial_elevation_noise_std_dev_base_;
    CovarianceSquareRootType s_r_;
};

/**
 * @brief Vision Measurement Model for 2D Pixel Coordinates.
 *
 * This model predicts 2D pixel coordinates (u, v) of a target's projection onto an image plane.
 * It assumes the state vector contains 3D position components (x, y, z).
 *
 * Measurement vector: [u_pixel, v_pixel]^T (2 dimensions)
 * State vector: Must contain 3D position.
 * Expected State Layout:
 * - Position: x (index 0), y (index 1), z (index 2)
 * - Other state components follow (e.g., velocity, acceleration)
 *
 * @tparam StateDim The dimension of the state vector.
 * @note This model requires camera intrinsic and extrinsic parameters (e.g., focal length, principal point,
 * camera position/orientation) to project 3D world coordinates to 2D pixel coordinates.
 * These parameters are assumed to be known and constant for simplicity in this model,
 * or managed externally and passed during construction/update if they are dynamic.
 */
template<int StateDim>
class VisionMeasurementModel : public Kalman::MeasurementModelBase<2, StateDim> {
public:
    // Define the specific MeasurementType, StateType, etc. for this model
    static const int MeasurementDim = 2; // Explicitly define MeasurementDim for consistency
    using MeasurementType = Kalman::Vector<double, MeasurementDim>;
    using StateType = Kalman::Vector<double, StateDim>;
    using CovarianceMatrix = Kalman::Matrix<double, MeasurementDim, MeasurementDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<MeasurementType>;

    /**
     * @brief Constructor for the Vision Measurement Model.
     * @param pixel_x_noise_std_dev_initial Initial standard deviation for X pixel noise.
     * @param pixel_y_noise_std_dev_initial Initial standard deviation for Y pixel noise.
     * @param focal_length_x_px Focal length in pixels along X-axis.
     * @param focal_length_y_px Focal length in pixels along Y-axis.
     * @param principal_point_x_px Principal point (optical center) X-coordinate in pixels.
     * @param principal_point_y_px Principal point (optical center) Y-coordinate in pixels.
     * @param camera_pos_x Camera X position in world coordinates.
     * @param camera_pos_y Camera Y position in world coordinates.
     * @param camera_pos_z Camera Z position in world coordinates.
     * @param camera_roll_rad Camera roll angle (radians).
     * @param camera_pitch_rad Camera pitch angle (radians).
     * @param camera_yaw_rad Camera yaw angle (radians).
     *
     * @note For simplicity, this model assumes a pinhole camera model and static camera parameters.
     * For dynamic camera motion, these parameters would need to be updated.
     */
    VisionMeasurementModel(double pixel_x_noise_std_dev_initial,
                           double pixel_y_noise_std_dev_initial,
                           double focal_length_x_px,
                           double focal_length_y_px,
                           double principal_point_x_px,
                           double principal_point_y_px,
                           double camera_pos_x,
                           double camera_pos_y,
                           double camera_pos_z,
                           double camera_roll_rad,
                           double camera_pitch_rad,
                           double camera_yaw_rad)
        : pixel_x_noise_std_dev_base_(pixel_x_noise_std_dev_initial),
          pixel_y_noise_std_dev_base_(pixel_y_noise_std_dev_initial),
          initial_pixel_x_noise_std_dev_base_(pixel_x_noise_std_dev_initial), // Store initial for adaptation
          initial_pixel_y_noise_std_dev_base_(pixel_y_noise_std_dev_initial),
          fx_(focal_length_x_px), fy_(focal_length_y_px),
          cx_(principal_point_x_px), cy_(principal_point_y_px),
          camera_pos_w_(camera_pos_x, camera_pos_y, camera_pos_z)
    {
        // Initialize the square root of the measurement noise covariance matrix (S_R).
        s_r_.setIdentity();
        s_r_(0,0) = pixel_x_noise_std_dev_base_; // Pixel X noise
        s_r_(1,1) = pixel_y_noise_std_dev_base_; // Pixel Y noise

        // Precompute camera rotation matrix (world to camera frame)
        // Roll (X), Pitch (Y), Yaw (Z) - extrinsic rotation.
        // Assuming ZYX (yaw-pitch-roll) convention for extrinsic rotation (world to camera).
        Eigen::AngleAxisd rollAngle(camera_roll_rad, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(camera_pitch_rad, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(camera_yaw_rad, Eigen::Vector3d::UnitZ());

        // Combined rotation (order matters: yaw-pitch-roll for ZYX)
        camera_rot_w_to_c_ = yawAngle * pitchAngle * rollAngle;
    }

    /**
     * @brief Implements the measurement prediction function for the Vision model.
     *
     * This function projects the 3D world position of the target onto the 2D image plane
     * using a pinhole camera model and known camera parameters.
     *
     * @param x The current state vector (e.g., [x, y, z, ...]^T).
     * @return The predicted measurement vector [u_pixel, v_pixel]^T.
     */
    MeasurementType h(const StateType& x) const override {
        MeasurementType predicted_measurement;

        // Target position in world coordinates
        Eigen::Vector3d target_pos_w(x(0), x(1), x(2));

        // 1. Transform target position from world frame to camera frame
        Eigen::Vector3d target_pos_c = camera_rot_w_to_c_ * (target_pos_w - camera_pos_w_);

        // Check if target is in front of the camera (positive Z in camera frame)
        // Returning a large finite value instead of infinity for better SRUKF compatibility.
        if (target_pos_c.z() <= 1e-6) { // Target is behind or at the camera plane
            const double large_finite_value = 1e100; // A very large finite number
            predicted_measurement(0) = large_finite_value;
            predicted_measurement(1) = large_finite_value;
            return predicted_measurement;
        }

        // 2. Project 3D camera coordinates to 2D normalized image coordinates (x_n, y_n)
        double x_n = target_pos_c.x() / target_pos_c.z();
        double y_n = target_pos_c.y() / target_pos_c.z();

        // 3. Apply camera intrinsics to get pixel coordinates (u, v)
        predicted_measurement(0) = fx_ * x_n + cx_; // u_pixel
        predicted_measurement(1) = fy_ * y_n + cy_; // v_pixel

        return predicted_measurement;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R).
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_r_;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R)
     * that might be time-step dependent.
     * @param dt The time step.
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        (void)dt; // Suppress unused parameter warning
        return s_r_;
    }

    /**
     * @brief Sets the measurement noise standard deviations for X and Y pixels.
     * @param new_pixel_x_sigma The new standard deviation for X pixel noise.
     * @param new_pixel_y_sigma The new standard deviation for Y pixel noise.
     */
    void setMeasurementNoise(double new_pixel_x_sigma, double new_pixel_y_sigma) {
        s_r_(0,0) = new_pixel_x_sigma;
        s_r_(1,1) = new_pixel_y_sigma;
    }

    /**
     * @brief Sets the full measurement noise covariance matrix (R) and computes its Cholesky decomposition (S_R).
     * @param full_R_matrix The full, positive semi-definite R matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     */
    void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override {
        CovarianceMatrix sym_R = (full_R_matrix + full_R_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_R);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("VisionMeasurementModel: Input R matrix is not positive definite or symmetric.");
        }
        s_r_ = llt.matrixL();
    }

    /**
     * @brief Dynamically adjusts the measurement noise (S_R) based on sensor context.
     * @param ctx The SensorContext containing dynamic sensor information (e.g., pixel quality, object size).
     */
    void setMeasurementNoiseFromContext(const SensorContext& ctx) override {
        // Example adaptation logic:
        // Lower pixel quality -> higher noise
        // Smaller object size -> higher noise
        double quality_factor = 1.0;
        if (ctx.pixel_quality < 0.7) { // Below 70% quality, increase noise
            quality_factor = 1.0 + (0.7 - ctx.pixel_quality) * 0.5; // Heuristic
        } else if (ctx.pixel_quality > 0.9) { // Very high quality, slight reduction
            quality_factor = 1.0 - (ctx.pixel_quality - 0.9) * 0.1;
        }
        quality_factor = std::max(0.5, quality_factor);

        double size_factor = 1.0;
        if (ctx.object_size_pixels < 20.0) { // Small object, increase noise
            size_factor = 1.0 + (20.0 - ctx.object_size_pixels) * 0.05; // Heuristic
        }
        size_factor = std::max(1.0, size_factor);

        double adapted_pixel_x_sigma = initial_pixel_x_noise_std_dev_base_ * quality_factor * size_factor;
        double adapted_pixel_y_sigma = initial_pixel_y_noise_std_dev_base_ * quality_factor * size_factor;

        setMeasurementNoise(adapted_pixel_x_sigma, adapted_pixel_y_sigma);
    }

    /**
     * @brief Returns the dimension of the measurement vector.
     */
    int getMeasurementDim() const override {
        return MeasurementDim;
    }

    /**
     * @brief Returns the dimension of the state vector this model expects.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the measurement model.
     */
    const char* getModelName() const override {
        return "VisionMeasurementModel"; // Consistent naming
    }

private:
    double pixel_x_noise_std_dev_base_;
    double pixel_y_noise_std_dev_base_;
    double initial_pixel_x_noise_std_dev_base_; // Store initial for adaptation
    double initial_pixel_y_noise_std_dev_base_;
    double fx_, fy_; // Focal lengths
    double cx_, cy_; // Principal point
    Eigen::Vector3d camera_pos_w_; // Camera position in world frame
    Eigen::Quaterniond camera_rot_w_to_c_; // Camera orientation (world to camera rotation)
    CovarianceSquareRootType s_r_;
};

/**
 * @brief Thermal Measurement Model for 2D Pixel Coordinates.
 *
 * This model is similar to the VisionMeasurementModel but is tailored for thermal cameras.
 * It predicts 2D pixel coordinates (u, v) of a target's projection onto an image plane.
 * It assumes the state vector contains 3D position components (x, y, z).
 *
 * Measurement vector: [u_pixel, v_pixel]^T (2 dimensions)
 * State vector: Must contain 3D position.
 * Expected State Layout:
 * - Position: x (index 0), y (index 1), z (index 2)
 * - Other state components follow (e.g., velocity, acceleration)
 *
 * @tparam StateDim The dimension of the state vector.
 * @note This model requires camera intrinsic and extrinsic parameters, similar to a standard
 * vision camera. It can also incorporate thermal signature features if augmented in the state.
 */
template<int StateDim>
class ThermalMeasurementModel : public Kalman::MeasurementModelBase<2, StateDim> {
public:
    // Define the specific MeasurementType, StateType, etc. for this model
    static const int MeasurementDim = 2; // Explicitly define MeasurementDim for consistency
    using MeasurementType = Kalman::Vector<double, MeasurementDim>;
    using StateType = Kalman::Vector<double, StateDim>;
    using CovarianceMatrix = Kalman::Matrix<double, MeasurementDim, MeasurementDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<MeasurementType>;

    /**
     * @brief Constructor for the Thermal Measurement Model.
     * @param pixel_x_noise_std_dev_initial Initial standard deviation for X pixel noise.
     * @param pixel_y_noise_std_dev_initial Initial standard deviation for Y pixel noise.
     * @param focal_length_x_px Focal length in pixels along X-axis.
     * @param focal_length_y_px Focal length in pixels along Y-axis.
     * @param principal_point_x_px Principal point (optical center) X-coordinate in pixels.
     * @param principal_point_y_px Principal point (optical center) Y-coordinate in pixels.
     * @param camera_pos_x Camera X position in world coordinates.
     * @param camera_pos_y Camera Y position in world coordinates.
     * @param camera_pos_z Camera Z position in world coordinates.
     * @param camera_roll_rad Camera roll angle (radians).
     * @param camera_pitch_rad Camera pitch angle (radians).
     * @param camera_yaw_rad Camera yaw angle (radians).
     *
     * @note This model assumes a pinhole camera model and static camera parameters.
     * For dynamic camera motion, these parameters would need to be updated.
     */
    ThermalMeasurementModel(double pixel_x_noise_std_dev_initial,
                            double pixel_y_noise_std_dev_initial,
                            double focal_length_x_px,
                            double focal_length_y_px,
                            double principal_point_x_px,
                            double principal_point_y_px,
                            double camera_pos_x,
                            double camera_pos_y,
                            double camera_pos_z,
                            double camera_roll_rad,
                            double camera_pitch_rad,
                            double camera_yaw_rad)
        : pixel_x_noise_std_dev_base_(pixel_x_noise_std_dev_initial),
          pixel_y_noise_std_dev_base_(pixel_y_noise_std_dev_initial),
          initial_pixel_x_noise_std_dev_base_(pixel_x_noise_std_dev_initial), // Store initial for adaptation
          initial_pixel_y_noise_std_dev_base_(pixel_y_noise_std_dev_initial),
          fx_(focal_length_x_px), fy_(focal_length_y_px),
          cx_(principal_point_x_px), cy_(principal_point_y_px),
          camera_pos_w_(camera_pos_x, camera_pos_y, camera_pos_z)
    {
        // Initialize the square root of the measurement noise covariance matrix (S_R).
        s_r_.setIdentity();
        s_r_(0,0) = pixel_x_noise_std_dev_base_; // Pixel X noise
        s_r_(1,1) = pixel_y_noise_std_dev_base_; // Pixel Y noise

        // Precompute camera rotation matrix (world to camera frame)
        Eigen::AngleAxisd rollAngle(camera_roll_rad, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(camera_pitch_rad, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(camera_yaw_rad, Eigen::Vector3d::UnitZ());

        camera_rot_w_to_c_ = yawAngle * pitchAngle * rollAngle;
    }

    /**
     * @brief Implements the measurement prediction function for the Thermal model.
     *
     * This function projects the 3D world position of the target onto the 2D thermal image plane
     * using a pinhole camera model and known camera parameters.
     *
     * @param x The current state vector (e.g., [x, y, z, ...]^T).
     * @return The predicted measurement vector [u_pixel, v_pixel]^T.
     */
    MeasurementType h(const StateType& x) const override {
        // The h() function is identical to VisionMeasurementModel's h()
        // as the projection geometry is the same for any camera.
        MeasurementType predicted_measurement;

        // Target position in world coordinates
        Eigen::Vector3d target_pos_w(x(0), x(1), x(2));

        // 1. Transform target position from world frame to camera frame
        Eigen::Vector3d target_pos_c = camera_rot_w_to_c_ * (target_pos_w - camera_pos_w_);

        // Check if target is in front of the camera (positive Z in camera frame)
        // Returning a large finite value instead of infinity for better SRUKF compatibility.
        if (target_pos_c.z() <= 1e-6) { // Target is behind or at the camera plane
            const double large_finite_value = 1e100; // A very large finite number
            predicted_measurement(0) = large_finite_value;
            predicted_measurement(1) = large_finite_value;
            return predicted_measurement;
        }

        // 2. Project 3D camera coordinates to 2D normalized image coordinates (x_n, y_n)
        double x_n = target_pos_c.x() / target_pos_c.z();
        double y_n = target_pos_c.y() / target_pos_c.z();

        // 3. Apply camera intrinsics to get pixel coordinates (u, v)
        predicted_measurement(0) = fx_ * x_n + cx_; // u_pixel
        predicted_measurement(1) = fy_ * y_n + cy_; // v_pixel

        return predicted_measurement;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R).
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_r_;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R)
     * that might be time-step dependent.
     * @param dt The time step.
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        (void)dt; // Suppress unused parameter warning
        return s_r_;
    }

    /**
     * @brief Sets the measurement noise standard deviations for X and Y pixels.
     * @param new_pixel_x_sigma The new standard deviation for X pixel noise.
     * @param new_pixel_y_sigma The new standard deviation for Y pixel noise.
     */
    void setMeasurementNoise(double new_pixel_x_sigma, double new_pixel_y_sigma) {
        s_r_(0,0) = new_pixel_x_sigma;
        s_r_(1,1) = new_pixel_y_sigma;
    }

    /**
     * @brief Sets the full measurement noise covariance matrix (R) and computes its Cholesky decomposition (S_R).
     * @param full_R_matrix The full, positive semi-definite R matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     */
    void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override {
        CovarianceMatrix sym_R = (full_R_matrix + full_R_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_R);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("ThermalMeasurementModel: Input R matrix is not positive definite or symmetric.");
        }
        s_r_ = llt.matrixL();
    }

    /**
     * @brief Dynamically adjusts the measurement noise (S_R) based on sensor context.
     * @param ctx The SensorContext containing dynamic sensor information (e.g., pixel quality, object size, thermal contrast).
     */
    void setMeasurementNoiseFromContext(const SensorContext& ctx) override {
        // Example adaptation logic:
        // Lower pixel quality -> higher noise
        // Smaller object size -> higher noise
        // Lower thermal contrast -> higher noise
        double quality_factor = 1.0;
        if (ctx.pixel_quality < 0.7) { // Below 70% quality, increase noise
            quality_factor = 1.0 + (0.7 - ctx.pixel_quality) * 0.5;
        } else if (ctx.pixel_quality > 0.9) {
            quality_factor = 1.0 - (ctx.pixel_quality - 0.9) * 0.1;
        }
        quality_factor = std::max(0.5, quality_factor);

        double size_factor = 1.0;
        if (ctx.object_size_pixels < 20.0) { // Small object, increase noise
            size_factor = 1.0 + (20.0 - ctx.object_size_pixels) * 0.05;
        }
        size_factor = std::max(1.0, size_factor);

        // Assuming ctx.raw_confidence_metrics(1) contains a thermal contrast score (e.g., 0-1)
        // Ensure index is valid before accessing and provide a default if not.
        double thermal_contrast = 1.0; // Default to high contrast if metric not available
        if (ctx.raw_confidence_metrics.size() > 1) {
            thermal_contrast = ctx.raw_confidence_metrics(1);
        }
        double contrast_factor = 1.0;
        if (thermal_contrast < 0.5) { // Low contrast, increase noise
            contrast_factor = 1.0 + (0.5 - thermal_contrast) * 1.0;
        }
        contrast_factor = std::max(1.0, contrast_factor);

        double adapted_pixel_x_sigma = initial_pixel_x_noise_std_dev_base_ * quality_factor * size_factor * contrast_factor;
        double adapted_pixel_y_sigma = initial_pixel_y_noise_std_dev_base_ * quality_factor * size_factor * contrast_factor;

        setMeasurementNoise(adapted_pixel_x_sigma, adapted_pixel_y_sigma);
    }

    /**
     * @brief Returns the dimension of the measurement vector.
     */
    int getMeasurementDim() const override {
        return MeasurementDim;
    }

    /**
     * @brief Returns the dimension of the state vector this model expects.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the measurement model.
     */
    const char* getModelName() const override {
        return "ThermalMeasurementModel"; // Consistent naming
    }

private:
    double pixel_x_noise_std_dev_base_;
    double pixel_y_noise_std_dev_base_;
    double initial_pixel_x_noise_std_dev_base_; // Store initial for adaptation
    double initial_pixel_y_noise_std_dev_base_;
    double fx_, fy_; // Focal lengths
    double cx_, cy_; // Principal point
    Eigen::Vector3d camera_pos_w_; // Camera position in world frame
    Eigen::Quaterniond camera_rot_w_to_c_; // Camera orientation (world to camera rotation)
    CovarianceSquareRootType s_r_;
};

/**
 * @brief Lidar Measurement Model for 3D Range, Bearing, and Elevation.
 *
 * This model predicts 3D polar measurements (range, azimuth, elevation) from a Lidar sensor.
 * It is similar to the RadarRangeBearingElevationMeasurementModel but tailored for Lidar characteristics.
 * It assumes the state vector contains position components at indices 0, 1, 2.
 *
 * Measurement vector: [range, azimuth, elevation]^T (3 dimensions)
 * State vector: Can be any dimension, as long as it contains 3D position.
 * Expected State Layout:
 * - Position: x (index 0), y (index 1), z (index 2)
 * - Other state components follow (e.g., velocity, acceleration)
 *
 * @tparam StateDim The dimension of the state vector.
 */
template<int StateDim>
class LidarMeasurementModel : public Kalman::MeasurementModelBase<3, StateDim> {
public:
    // Define the specific MeasurementType, StateType, etc. for this model
    static const int MeasurementDim = 3; // Explicitly define MeasurementDim for consistency
    using MeasurementType = Kalman::Vector<double, MeasurementDim>;
    using StateType = Kalman::Vector<double, StateDim>;
    using CovarianceMatrix = Kalman::Matrix<double, MeasurementDim, MeasurementDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<MeasurementType>;

    /**
     * @brief Constructor for the Lidar Measurement Model.
     * @param range_noise_std_dev_initial Initial standard deviation for range measurement noise.
     * @param bearing_noise_std_dev_initial Initial standard deviation for bearing (azimuth) measurement noise (radians).
     * @param elevation_noise_std_dev_initial Initial standard deviation for elevation measurement noise (radians).
     */
    LidarMeasurementModel(double range_noise_std_dev_initial,
                          double bearing_noise_std_dev_initial,
                          double elevation_noise_std_dev_initial)
        : range_noise_std_dev_base_(range_noise_std_dev_initial),
          bearing_noise_std_dev_base_(bearing_noise_std_dev_initial),
          elevation_noise_std_dev_base_(elevation_noise_std_dev_initial),
          initial_range_noise_std_dev_base_(range_noise_std_dev_initial), // Store initial for adaptation
          initial_bearing_noise_std_dev_base_(bearing_noise_std_dev_initial),
          initial_elevation_noise_std_dev_base_(elevation_noise_std_dev_initial)
    {
        // Initialize the square root of the measurement noise covariance matrix (S_R).
        s_r_.setIdentity();
        s_r_(0,0) = range_noise_std_dev_base_;      // Range noise
        s_r_(1,1) = bearing_noise_std_dev_base_;    // Azimuth noise (radians)
        s_r_(2,2) = elevation_noise_std_dev_base_; // Elevation noise (radians)
    }

    /**
     * @brief Implements the measurement prediction function for the Lidar model.
     *
     * This function converts the 3D Cartesian position (x, y, z) from the filter's
     * state vector into polar coordinates (range, azimuth, elevation).
     *
     * @param x The current state vector (e.g., [x, y, z, ...]^T).
     * @return The predicted measurement vector [range, azimuth, elevation]^T.
     */
    MeasurementType h(const StateType& x) const override {
        // Identical to RadarRangeBearingElevationMeasurementModel's h()
        MeasurementType predicted_measurement;

        double px = x(0);
        double py = x(1);
        double pz = x(2);

        // Calculate range
        double range = std::hypot(px, py, pz);
        predicted_measurement(0) = range;

        // Use the helper function for azimuth and elevation
        Eigen::Vector2d az_el = Kalman::MeasurementModelBase<MeasurementDim, StateDim>::cartesianToAzimuthElevation(px, py, pz);
        predicted_measurement(1) = az_el(0); // Azimuth
        predicted_measurement(2) = az_el(1); // Elevation

        return predicted_measurement;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R).
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_r_;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R)
     * that might be time-step dependent.
     * @param dt The time step.
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        (void)dt; // Suppress unused parameter warning
        return s_r_;
    }

    /**
     * @brief Sets the measurement noise standard deviations for range, bearing, and elevation.
     * @param new_range_sigma The new standard deviation for range noise.
     * @param new_bearing_sigma The new standard deviation for bearing noise.
     * @param new_elevation_sigma The new standard deviation for elevation noise.
     */
    void setMeasurementNoise(double new_range_sigma, double new_bearing_sigma, double new_elevation_sigma) {
        s_r_(0,0) = new_range_sigma;
        s_r_(1,1) = new_bearing_sigma;
        s_r_(2,2) = new_elevation_sigma;
    }

    /**
     * @brief Sets the full measurement noise covariance matrix (R) and computes its Cholesky decomposition (S_R).
     * @param full_R_matrix The full, positive semi-definite R matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     */
    void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override {
        CovarianceMatrix sym_R = (full_R_matrix + full_R_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_R);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("LidarMeasurementModel: Input R matrix is not positive definite or symmetric.");
        }
        s_r_ = llt.matrixL();
    }

    /**
     * @brief Dynamically adjusts the measurement noise (S_R) based on sensor context.
     * @param ctx The SensorContext containing dynamic sensor information (e.g., reflectivity, weather).
     */
    void setMeasurementNoiseFromContext(const SensorContext& ctx) override {
        // Example adaptation logic:
        // Lower reflectivity -> higher noise (for Lidar, this is like SNR)
        // Higher clutter/weather -> higher noise
        double reflectivity_factor = 1.0;
        // Assuming ctx.raw_confidence_metrics(0) contains reflectivity (e.g., 0-1)
        // Ensure index is valid before accessing and provide a default if not.
        double reflectivity = 1.0; // Default to high reflectivity if metric not available
        if (ctx.raw_confidence_metrics.size() > 0) {
            reflectivity = ctx.raw_confidence_metrics(0);
        }
        if (reflectivity < 0.5) { // Low reflectivity, increase noise
            reflectivity_factor = 1.0 + (0.5 - reflectivity) * 0.5;
        } else if (reflectivity > 0.9) { // High reflectivity, slight reduction
            reflectivity_factor = 1.0 - (reflectivity - 0.9) * 0.1;
        }
        reflectivity_factor = std::max(0.5, reflectivity_factor);

        double clutter_factor = 1.0 + ctx.clutter_level * 0.2; // Higher clutter -> higher noise
        double weather_factor = 1.0 + ctx.weather_index * 0.3; // Worse weather -> higher noise

        // Apply factors to base standard deviations
        double adapted_range_sigma = initial_range_noise_std_dev_base_ * reflectivity_factor * clutter_factor * weather_factor;
        double adapted_bearing_sigma = initial_bearing_noise_std_dev_base_ * reflectivity_factor * clutter_factor * weather_factor;
        double adapted_elevation_sigma = initial_elevation_noise_std_dev_base_ * reflectivity_factor * clutter_factor * weather_factor;

        setMeasurementNoise(adapted_range_sigma, adapted_bearing_sigma, adapted_elevation_sigma);
    }

    /**
     * @brief Returns the dimension of the measurement vector.
     */
    int getMeasurementDim() const override {
        return MeasurementDim;
    }

    /**
     * @brief Returns the dimension of the state vector this model expects.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the measurement model.
     */
    const char* getModelName() const override {
        return "LidarMeasurementModel"; // Consistent naming
    }

private:
    double range_noise_std_dev_base_;
    double bearing_noise_std_dev_base_;
    double elevation_noise_std_dev_base_;
    double initial_range_noise_std_dev_base_; // Store initial for adaptation
    double initial_bearing_noise_std_dev_base_;
    double initial_elevation_noise_std_dev_base_;
    CovarianceSquareRootType s_r_;
};

/**
 * @brief Passive RF Measurement Model for Angle of Arrival (AoA).
 *
 * This model predicts Angle of Arrival (azimuth, elevation) from passive RF sensors.
 * It assumes the state vector contains position components at indices 0, 1, 2.
 * It does NOT assume an augmented state for RF power, as it's passive.
 *
 * Measurement vector: [azimuth, elevation]^T (2 dimensions)
 * State vector: Can be any dimension, as long as it contains 3D position.
 * Expected State Layout:
 * - Position: x (index 0), y (index 1), z (index 2)
 * - Other state components follow (e.g., velocity, acceleration)
 *
 * @tparam StateDim The dimension of the state vector.
 */
template<int StateDim>
class PassiveRFMeasurementModel : public Kalman::MeasurementModelBase<2, StateDim> {
public:
    // Define the specific MeasurementType, StateType, etc. for this model
    static const int MeasurementDim = 2; // Explicitly define MeasurementDim for consistency
    using MeasurementType = Kalman::Vector<double, MeasurementDim>;
    using StateType = Kalman::Vector<double, StateDim>;
    using CovarianceMatrix = Kalman::Matrix<double, MeasurementDim, MeasurementDim>;
    using CovarianceSquareRootType = Kalman::CovarianceSquareRoot<MeasurementType>;

    /**
     * @brief Constructor for the Passive RF Measurement Model.
     * @param azimuth_noise_std_dev_initial Initial standard deviation for azimuth measurement noise (radians).
     * @param elevation_noise_std_dev_initial Initial standard deviation for elevation measurement noise (radians).
     */
    PassiveRFMeasurementModel(double azimuth_noise_std_dev_initial,
                              double elevation_noise_std_dev_initial)
        : azimuth_noise_std_dev_base_(azimuth_noise_std_dev_initial),
          elevation_noise_std_dev_base_(elevation_noise_std_dev_initial),
          initial_azimuth_noise_std_dev_base_(azimuth_noise_std_dev_initial), // Store initial for adaptation
          initial_elevation_noise_std_dev_base_(elevation_noise_std_dev_initial)
    {
        // Initialize the square root of the measurement noise covariance matrix (S_R).
        s_r_.setIdentity();
        s_r_(0,0) = azimuth_noise_std_dev_base_;   // Azimuth noise
        s_r_(1,1) = elevation_noise_std_dev_base_; // Elevation noise
    }

    /**
     * @brief Implements the measurement prediction function for the Passive RF model.
     *
     * This function predicts Angle of Arrival (azimuth, elevation) from the 3D Cartesian
     * position (x, y, z) in the state.
     *
     * @param x The current state vector (e.g., [x, y, z, ...]^T).
     * @return The predicted measurement vector [azimuth, elevation]^T.
     */
    MeasurementType h(const StateType& x) const override {
        // Identical to AcousticMeasurementModel's h()
        MeasurementType predicted_measurement;

        double px = x(0);
        double py = x(1);
        double pz = x(2);

        // Use the helper function for azimuth and elevation
        Eigen::Vector2d az_el = Kalman::MeasurementModelBase<MeasurementDim, StateDim>::cartesianToAzimuthElevation(px, py, pz);
        predicted_measurement(0) = az_el(0); // Azimuth
        predicted_measurement(1) = az_el(1); // Elevation

        return predicted_measurement;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R).
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot() const override {
        return s_r_;
    }

    /**
     * @brief Returns the square root of the measurement noise covariance matrix (S_R)
     * that might be time-step dependent.
     * @param dt The time step.
     * @return The S_R matrix.
     */
    CovarianceSquareRootType getCovarianceSquareRoot(double dt) const override {
        (void)dt; // Suppress unused parameter warning
        return s_r_;
    }

    /**
     * @brief Sets the measurement noise standard deviations for azimuth and elevation.
     * @param new_azimuth_sigma The new standard deviation for azimuth noise.
     * @param new_elevation_sigma The new standard deviation for elevation noise.
     */
    void setMeasurementNoise(double new_azimuth_sigma, double new_elevation_sigma) {
        s_r_(0,0) = new_azimuth_sigma;
        s_r_(1,1) = new_elevation_sigma;
    }

    /**
     * @brief Sets the full measurement noise covariance matrix (R) and computes its Cholesky decomposition (S_R).
     * @param full_R_matrix The full, positive semi-definite R matrix.
     * @throws std::runtime_error if the matrix is not positive definite.
     */
    void setFullMeasurementNoiseCovariance(const CovarianceMatrix& full_R_matrix) override {
        CovarianceMatrix sym_R = (full_R_matrix + full_R_matrix.transpose()) / 2.0;
        Eigen::LLT<CovarianceMatrix> llt(sym_R);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("PassiveRFMeasurementModel: Input R matrix is not positive definite or symmetric.");
        }
        s_r_ = llt.matrixL();
    }

    /**
     * @brief Dynamically adjusts the measurement noise (S_R) based on sensor context.
     * @param ctx The SensorContext containing dynamic sensor information (e.g., jamming level, signal strength).
     */
    void setMeasurementNoiseFromContext(const SensorContext& ctx) override {
        // Example adaptation logic:
        // Higher jamming level -> higher noise
        // Lower signal strength (from raw_confidence_metrics) -> higher noise
        double jamming_factor = 1.0 + ctx.jamming_level * 0.3; // Higher jamming -> higher noise
        jamming_factor = std::max(1.0, jamming_factor);

        double signal_strength_factor = 1.0;
        // Assuming ctx.raw_confidence_metrics(0) contains a normalized signal strength (e.g., 0-1)
        // Ensure index is valid before accessing and provide a default if not.
        double signal_strength = 1.0; // Default to strong signal if metric not available
        if (ctx.raw_confidence_metrics.size() > 0) {
            signal_strength = ctx.raw_confidence_metrics(0);
        }
        if (signal_strength < 0.5) { // Weak signal, increase noise
            signal_strength_factor = 1.0 + (0.5 - signal_strength) * 0.5;
        } else if (signal_strength > 0.9) { // Strong signal, slight reduction
            signal_strength_factor = 1.0 - (signal_strength - 0.9) * 0.1;
        }
        signal_strength_factor = std::max(0.5, signal_strength_factor);

        double adapted_azimuth_sigma = initial_azimuth_noise_std_dev_base_ * jamming_factor * signal_strength_factor;
        double adapted_elevation_sigma = initial_elevation_noise_std_dev_base_ * jamming_factor * signal_strength_factor;

        setMeasurementNoise(adapted_azimuth_sigma, adapted_elevation_sigma);
    }

    /**
     * @brief Returns the dimension of the measurement vector.
     */
    int getMeasurementDim() const override {
        return MeasurementDim;
    }

    /**
     * @brief Returns the dimension of the state vector this model expects.
     */
    int getStateDim() const override {
        return StateDim;
    }

    /**
     * @brief Returns the name of the measurement model.
     */
    const char* getModelName() const override {
        return "PassiveRFMeasurementModel"; // Consistent naming
    }

private:
    double azimuth_noise_std_dev_base_;
    double elevation_noise_std_dev_base_;
    double initial_azimuth_noise_std_dev_base_; // Store initial for adaptation
    double initial_elevation_noise_std_dev_base_;
    CovarianceSquareRootType s_r_;
};

#endif // SKYFILTER_MEASUREMENT_MODELS_HPP_
