// measurement_adapter.hpp
#ifndef SKYFILTER_MEASUREMENT_ADAPTER_HPP_
#define SKYFILTER_MEASUREMENT_ADAPTER_HPP_

// Removed: #define _USE_MATH_DEFINES // M_PI is not directly used here

// Standard Library Headers
#include <cmath>       // For std::sqrt, std::isnan, std::isfinite
#include <stdexcept>   // For std::runtime_error
#include <string>      // For error messages
#include <algorithm>   // Added for std::min, std::max

// Eigen Library Headers
#include <Eigen/Dense>     // For Eigen::VectorXd, Eigen::MatrixXd
#include <Eigen/Cholesky>  // Added for Eigen::LLT
#include <Eigen/Core>      // For .array().isNaN().any() if needed, though Dense often pulls it in

// Project-specific Headers
#include "common_types.hpp" // For SensorType, RawSensorMeasurement, AdaptedMeasurement, SensorContext

/**
 * @brief The MeasurementAdapter class is responsible for normalizing and preparing
 * diverse raw sensor inputs into a common format suitable for the Kalman filter.
 * It extracts relevant data, estimates initial measurement noise covariance (S_R),
 * and calculates sensor-specific confidence metrics, and computing combined_confidence.
 * The raw sensor confidence (raw_sensor_confidence) is derived either from SNR
 * (for Radar/RF) or from custom heuristics (for other sensor types).
 * The combined confidence is calculated as the product of the raw sensor's confidence
 * and an external meta-score, reflecting a multiplicative effect on overall trust.
 */
class MeasurementAdapter {
public:
    // Constants for clamping and default values
    static constexpr double kMinCovarianceValue = 1e-9;
    static constexpr double kMinMetaScore = 0.1;
    static constexpr double kMinSNR = 0.1;
    static constexpr double kNominalSNR = 20.0; // Nominal SNR for sensors without direct SNR metrics

    /**
     * @brief Constructor for the MeasurementAdapter.
     * @param default_radar_range_noise_std_dev Default std dev for radar range (m).
     * @param default_radar_angle_noise_std_dev Default std dev for radar angles (rad).
     * @param default_rf_angle_noise_std_dev Default std dev for RF angles (rad).
     * @param default_rf_rssi_noise_std_dev Default std dev for RF RSSI (dBm).
     * @param default_imu_accel_noise_std_dev Default std dev for IMU acceleration (m/s^2).
     * @param default_imu_gyro_noise_std_dev Default std dev for IMU gyroscope (rad/s).
     * @param default_acoustic_angle_noise_std_dev Default std dev for acoustic angles (rad).
     * @param default_pixel_noise_std_dev Default std dev for vision/thermal pixels.
     * @param default_lidar_range_noise_std_dev Default std dev for lidar range (m).
     * @param default_lidar_angle_noise_std_dev Default std dev for lidar angles (rad).
     * @param default_passive_rf_angle_noise_std_dev Default std dev for passive RF angles (rad).
     */
    MeasurementAdapter(double default_radar_range_noise_std_dev,
                       double default_radar_angle_noise_std_dev,
                       double default_rf_angle_noise_std_dev,
                       double default_rf_rssi_noise_std_dev,
                       double default_imu_accel_noise_std_dev,
                       double default_imu_gyro_noise_std_dev,
                       double default_acoustic_angle_noise_std_dev,
                       double default_pixel_noise_std_dev,
                       double default_lidar_range_noise_std_dev,
                       double default_lidar_angle_noise_std_dev,
                       double default_passive_rf_angle_noise_std_dev)
        : default_radar_range_noise_std_dev_(default_radar_range_noise_std_dev),
          default_radar_angle_noise_std_dev_(default_radar_angle_noise_std_dev),
          default_rf_angle_noise_std_dev_(default_rf_angle_noise_std_dev),
          default_rf_rssi_noise_std_dev_(default_rf_rssi_noise_std_dev),
          default_imu_accel_noise_std_dev_(default_imu_accel_noise_std_dev),
          default_imu_gyro_noise_std_dev_(default_imu_gyro_noise_std_dev),
          default_acoustic_angle_noise_std_dev_(default_acoustic_angle_noise_std_dev),
          default_pixel_noise_std_dev_(default_pixel_noise_std_dev),
          default_lidar_range_noise_std_dev_(default_lidar_range_noise_std_dev),
          default_lidar_angle_noise_std_dev_(default_lidar_angle_noise_std_dev),
          default_passive_rf_angle_noise_std_dev_(default_passive_rf_angle_noise_std_dev) {}

    /**
     * @brief Adapts a raw sensor measurement into a filter-compatible format.
     * This involves extracting the measurement vector, calculating its initial
     * noise covariance (S_R), and deriving a sensor-specific confidence score.
     * It also populates a SensorContext struct for use by adaptive measurement models.
     *
     * @param rawMeasurement The raw sensor data.
     * @param external_meta_score A trust score for this specific sensor instance (0.0 to 1.0).
     * @param out_sensor_context Output parameter: Populated SensorContext for adaptive models.
     * @return An AdaptedMeasurement struct ready for the Kalman filter.
     * @throws std::runtime_error if the sensor type is unsupported or raw_data is malformed.
     */
    AdaptedMeasurement adapt(const RawSensorMeasurement& rawMeasurement,
                             double external_meta_score,
                             SensorContext& out_sensor_context) const {
        // Validate external_meta_score
        if (external_meta_score < 0.0 || external_meta_score > 1.0) {
            throw std::runtime_error("external_meta_score must be in [0,1].");
        }

        // Check for NaNs in raw_data and raw_confidence_metrics
        if (rawMeasurement.raw_data.array().isNaN().any()) {
            throw std::runtime_error("Raw sensor data contains NaN values.");
        }
        if (rawMeasurement.raw_confidence_metrics.array().isNaN().any()) {
            throw std::runtime_error("Raw sensor confidence metrics contain NaN values.");
        }

        AdaptedMeasurement adapted;
        adapted.sensor_type = rawMeasurement.type;
        adapted.external_meta_score = external_meta_score;

        // Initialize sensor context with default/safe values
        out_sensor_context = SensorContext();
        out_sensor_context.raw_confidence_metrics = rawMeasurement.raw_confidence_metrics;

        // Populate AdaptedMeasurement and SensorContext based on sensor type
        switch (rawMeasurement.type) {
            case SensorType::RADAR:
                adapted = adaptRadarMeasurement(rawMeasurement, external_meta_score, out_sensor_context);
                break;
            case SensorType::RF:
                adapted = adaptRFMeasurement(rawMeasurement, external_meta_score, out_sensor_context);
                break;
            case SensorType::IMU:
                adapted = adaptIMUMeasurement(rawMeasurement, external_meta_score, out_sensor_context);
                break;
            case SensorType::ACOUSTIC:
                adapted = adaptAcousticMeasurement(rawMeasurement, external_meta_score, out_sensor_context);
                break;
            case SensorType::VISION:
                adapted = adaptVisionMeasurement(rawMeasurement, external_meta_score, out_sensor_context);
                break;
            case SensorType::THERMAL:
                adapted = adaptThermalMeasurement(rawMeasurement, external_meta_score, out_sensor_context);
                break;
            case SensorType::LIDAR:
                adapted = adaptLidarMeasurement(rawMeasurement, external_meta_score, out_sensor_context);
                break;
            case SensorType::PASSIVE_RF:
                adapted = adaptPassiveRFMeasurement(rawMeasurement, external_meta_score, out_sensor_context);
                break;
            case SensorType::UNKNOWN:
            default:
                throw std::runtime_error("Unsupported sensor type for MeasurementAdapter.");
        }

        // Ensure R_covariance_sqrt is always positive definite and symmetric
        for (int i = 0; i < adapted.R_covariance_sqrt.rows(); ++i) {
            if (adapted.R_covariance_sqrt(i,i) < kMinCovarianceValue) { // Prevent zero or negative diagonal elements
                adapted.R_covariance_sqrt(i,i) = kMinCovarianceValue;
            }
        }

        return adapted;
    }

private:
    // Default noise parameters for each sensor type (tuned during system integration)
    const double default_radar_range_noise_std_dev_;
    const double default_radar_angle_noise_std_dev_;
    const double default_rf_angle_noise_std_dev_;
    const double default_rf_rssi_noise_std_dev_;
    const double default_imu_accel_noise_std_dev_;
    const double default_imu_gyro_noise_std_dev_;
    const double default_acoustic_angle_noise_std_dev_;
    const double default_pixel_noise_std_dev_;
    const double default_lidar_range_noise_std_dev_;
    const double default_lidar_angle_noise_std_dev_;
    const double default_passive_rf_angle_noise_std_dev_;

    /**
     * @brief Helper to check if all elements in an Eigen vector are finite.
     * @tparam Derived The Eigen derived type (e.g., Eigen::VectorXd, Eigen::Vector2d).
     * @param vec The Eigen vector to check.
     * @param name A descriptive name for the vector (for error messages).
     * @throws std::runtime_error if any element is not finite.
     */
    template<typename Derived>
    void checkFiniteAndThrow(const Eigen::MatrixBase<Derived>& vec, const std::string& name) const {
        for (int i = 0; i < vec.size(); ++i) {
            if (!std::isfinite(vec[i])) {
                throw std::runtime_error(name + " contains non-finite value at index " + std::to_string(i) + ".");
            }
        }
    }

    /**
     * @brief Helper to calculate adaptive R_covariance_sqrt based on SNR and meta-score.
     * @param R_base_variance The base R matrix (variance).
     * @param snr_db Signal-to-Noise Ratio in dB.
     * @param external_meta_score Trust score for the sensor (0.0 to 1.0).
     * @return The scaled square root of the covariance matrix.
     */
    Eigen::MatrixXd calculateAdaptiveRCovarianceSqrt(const Eigen::MatrixXd& R_base_variance, double snr_db, double external_meta_score) const {
        // Apply heuristic scaling factors to the variance matrix (R_base_variance)
        double clamped_snr_db = std::max(kMinSNR, snr_db);
        double snr_scaling_factor = 1.0 + std::max(0.0, (15.0 - clamped_snr_db) / 10.0);

        double meta_score_scaling_factor = 1.0 / std::max(kMinMetaScore, external_meta_score);

        Eigen::MatrixXd R_scaled = R_base_variance;
        // Apply scaling to the diagonal elements of the variance matrix
        for (int i = 0; i < R_scaled.rows(); ++i) {
            R_scaled(i,i) *= snr_scaling_factor * meta_score_scaling_factor;
        }

        // Ensure R_scaled is symmetric before Cholesky decomposition
        R_scaled = (R_scaled + R_scaled.transpose()) / 2.0;

        Eigen::LLT<Eigen::MatrixXd> llt(R_scaled);
        if (llt.info() != Eigen::Success) {
            // If R_scaled is not positive definite after scaling, throw an error.
            // In a production system, a more robust fallback (e.g., return identity or a very large diagonal)
            // might be considered, but for now, throwing is appropriate.
            throw std::runtime_error("calculateAdaptiveRCovarianceSqrt: Scaled R matrix is not positive definite.");
        }
        return llt.matrixL(); // Return the lower triangular Cholesky factor (S_R)
    }

    /**
     * @brief Helper to calculate a normalized confidence score from SNR.
     * @param snr_db Signal-to-Noise Ratio in dB.
     * @return Confidence score from 0.0 to 1.0.
     */
    double calculateConfidenceFromSNR(double snr_db) const {
        // Map SNR from -10dB to 30dB to confidence 0.0 to 1.0
        // Below -10dB, confidence is 0. Above 30dB, confidence is 1.
        double normalized_snr = (snr_db + 10.0) / 40.0; // Scale from [-10, 30] to [0, 1]
        return std::max(0.0, std::min(1.0, normalized_snr));
    }

    /**
     * @brief Adapts a raw Radar measurement.
     */
    AdaptedMeasurement adaptRadarMeasurement(const RawSensorMeasurement& rawMeasurement,
                                             double external_meta_score,
                                             SensorContext& out_sensor_context) const {
        AdaptedMeasurement adapted;
        adapted.sensor_type = SensorType::RADAR;
        adapted.external_meta_score = external_meta_score;

        // Expected raw_data: [range_m, azimuth_rad, elevation_rad, RCS_dBsm] (4D)
        // Expected raw_confidence_metrics: [SNR_db] (1D)
        if (rawMeasurement.raw_data.size() < 3) {
            throw std::runtime_error("Radar raw_data must have at least 3 dimensions (range, azimuth, elevation).");
        }
        double range_val = rawMeasurement.raw_data[0];
        double azimuth_val = rawMeasurement.raw_data[1];
        double elevation_val = rawMeasurement.raw_data[2];

        if (range_val <= 0.0) {
            range_val = kMinCovarianceValue; // Clamp to a small positive value
        }
        checkFiniteAndThrow(Eigen::Vector2d(azimuth_val, elevation_val), "Radar angles");

        adapted.measurement_vector.resize(3);
        adapted.measurement_vector << range_val, // range
                                      azimuth_val, // azimuth
                                      elevation_val; // elevation

        // Populate SensorContext
        if (rawMeasurement.raw_confidence_metrics.size() > 0) {
            out_sensor_context.snr_db = rawMeasurement.raw_confidence_metrics[0];
        }
        out_sensor_context.range_m = range_val; // Use clamped value

        // Calculate initial R_covariance_sqrt (R_base is variance)
        Eigen::Matrix3d R_base_variance = Eigen::Matrix3d::Identity();
        R_base_variance(0,0) = default_radar_range_noise_std_dev_ * default_radar_range_noise_std_dev_;
        R_base_variance(1,1) = default_radar_angle_noise_std_dev_ * default_radar_angle_noise_std_dev_;
        R_base_variance(2,2) = default_radar_angle_noise_std_dev_ * default_radar_angle_noise_std_dev_;

        adapted.R_covariance_sqrt = calculateAdaptiveRCovarianceSqrt(R_base_variance, out_sensor_context.snr_db, external_meta_score);
        adapted.raw_sensor_confidence = calculateConfidenceFromSNR(out_sensor_context.snr_db);
        adapted.combined_confidence = adapted.raw_sensor_confidence * adapted.external_meta_score;
        return adapted;
    }

    /**
     * @brief Adapts a raw RF measurement.
     */
    AdaptedMeasurement adaptRFMeasurement(const RawSensorMeasurement& rawMeasurement,
                                          double external_meta_score,
                                          SensorContext& out_sensor_context) const {
        AdaptedMeasurement adapted;
        adapted.sensor_type = SensorType::RF;
        adapted.external_meta_score = external_meta_score;

        // Expected raw_data: [azimuth_rad, elevation_rad, RSSI_dBm] (3D)
        // Expected raw_confidence_metrics: [SNR_db, jamming_level] (2D)
        if (rawMeasurement.raw_data.size() < 3) {
            throw std::runtime_error("RF raw_data must have 3 dimensions (azimuth, elevation, RSSI).");
        }
        double azimuth_val = rawMeasurement.raw_data[0];
        double elevation_val = rawMeasurement.raw_data[1];
        double rssi_val = rawMeasurement.raw_data[2];

        checkFiniteAndThrow(Eigen::Vector3d(azimuth_val, elevation_val, rssi_val), "RF measurements (azimuth, elevation, RSSI)");

        adapted.measurement_vector.resize(3);
        adapted.measurement_vector << azimuth_val, // azimuth
                                      elevation_val, // elevation
                                      rssi_val; // RSSI

        // Populate SensorContext
        if (rawMeasurement.raw_confidence_metrics.size() > 0) {
            out_sensor_context.snr_db = rawMeasurement.raw_confidence_metrics[0];
        }
        if (rawMeasurement.raw_confidence_metrics.size() > 1) {
            out_sensor_context.jamming_level = rawMeasurement.raw_confidence_metrics[1];
        }

        // Calculate initial R_covariance_sqrt (R_base is variance)
        Eigen::Matrix3d R_base_variance = Eigen::Matrix3d::Identity();
        R_base_variance(0,0) = default_rf_angle_noise_std_dev_ * default_rf_angle_noise_std_dev_;
        R_base_variance(1,1) = default_rf_angle_noise_std_dev_ * default_rf_angle_noise_std_dev_;
        R_base_variance(2,2) = default_rf_rssi_noise_std_dev_ * default_rf_rssi_noise_std_dev_;

        adapted.R_covariance_sqrt = calculateAdaptiveRCovarianceSqrt(R_base_variance, out_sensor_context.snr_db, external_meta_score);
        adapted.raw_sensor_confidence = calculateConfidenceFromSNR(out_sensor_context.snr_db);
        adapted.combined_confidence = adapted.raw_sensor_confidence * adapted.external_meta_score;
        return adapted;
    }

    /**
     * @brief Adapts a raw IMU measurement.
     */
    AdaptedMeasurement adaptIMUMeasurement(const RawSensorMeasurement& rawMeasurement,
                                           double external_meta_score,
                                           SensorContext& out_sensor_context) const {
        AdaptedMeasurement adapted;
        adapted.sensor_type = SensorType::IMU;
        adapted.external_meta_score = external_meta_score;

        // Expected raw_data: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] (6D)
        // No specific raw_confidence_metrics for IMU usually, but could be vibration/temperature
        if (rawMeasurement.raw_data.size() < 6) {
            throw std::runtime_error("IMU raw_data must have 6 dimensions (accel_x,y,z, gyro_x,y,z).");
        }
        adapted.measurement_vector.resize(6);
        adapted.measurement_vector = rawMeasurement.raw_data.head(6);

        checkFiniteAndThrow(adapted.measurement_vector, "IMU measurements (accel, gyro)");

        // No specific SNR for IMU, use a default confidence or derive from internal IMU health
        adapted.raw_sensor_confidence = 1.0; // Assume high confidence by default

        // Calculate initial R_covariance_sqrt (R_base is variance)
        Eigen::MatrixXd R_base_variance = Eigen::MatrixXd::Identity(6,6);
        R_base_variance(0,0) = default_imu_accel_noise_std_dev_ * default_imu_accel_noise_std_dev_;
        R_base_variance(1,1) = default_imu_accel_noise_std_dev_ * default_imu_accel_noise_std_dev_;
        R_base_variance(2,2) = default_imu_accel_noise_std_dev_ * default_imu_accel_noise_std_dev_;
        R_base_variance(3,3) = default_imu_gyro_noise_std_dev_ * default_imu_gyro_noise_std_dev_;
        R_base_variance(4,4) = default_imu_gyro_noise_std_dev_ * default_imu_gyro_noise_std_dev_;
        R_base_variance(5,5) = default_imu_gyro_noise_std_dev_ * default_imu_gyro_noise_std_dev_;

        // For IMU, meta-score might be more critical, or environmental factors like vibration
        // Use a nominal SNR for scaling, as IMU doesn't have a direct SNR.
        adapted.R_covariance_sqrt = calculateAdaptiveRCovarianceSqrt(R_base_variance, kNominalSNR, external_meta_score);
        adapted.combined_confidence = adapted.raw_sensor_confidence * adapted.external_meta_score;
        return adapted;
    }

    /**
     * @brief Adapts a raw Acoustic measurement.
     */
    AdaptedMeasurement adaptAcousticMeasurement(const RawSensorMeasurement& rawMeasurement,
                                                double external_meta_score,
                                                SensorContext& out_sensor_context) const {
        AdaptedMeasurement adapted;
        adapted.sensor_type = SensorType::ACOUSTIC;
        adapted.external_meta_score = external_meta_score;

        // Expected raw_data: [azimuth_rad, elevation_rad] (2D)
        // Expected raw_confidence_metrics: [ambient_noise_db] (1D)
        if (rawMeasurement.raw_data.size() < 2) {
            throw std::runtime_error("Acoustic raw_data must have 2 dimensions (azimuth, elevation).");
        }
        double azimuth_val = rawMeasurement.raw_data[0];
        double elevation_val = rawMeasurement.raw_data[1];

        checkFiniteAndThrow(Eigen::Vector2d(azimuth_val, elevation_val), "Acoustic angles");

        adapted.measurement_vector.resize(2);
        adapted.measurement_vector << azimuth_val, // azimuth
                                      elevation_val; // elevation

        // Populate SensorContext
        if (rawMeasurement.raw_confidence_metrics.size() > 0) {
            out_sensor_context.ambient_noise_db = rawMeasurement.raw_confidence_metrics[0];
        }

        // Calculate initial R_covariance_sqrt (R_base is variance)
        Eigen::Matrix2d R_base_variance = Eigen::Matrix2d::Identity();
        R_base_variance(0,0) = default_acoustic_angle_noise_std_dev_ * default_acoustic_angle_noise_std_dev_;
        R_base_variance(1,1) = default_acoustic_angle_noise_std_dev_ * default_acoustic_angle_noise_std_dev_;

        // Acoustic noise might scale with ambient noise, not SNR directly
        double noise_scale_from_ambient = 1.0 + std::max(0.0, out_sensor_context.ambient_noise_db / 50.0); // Heuristic
        // Call the general adaptive R calculation
        adapted.R_covariance_sqrt = calculateAdaptiveRCovarianceSqrt(R_base_variance, kNominalSNR, external_meta_score); // Use nominal SNR for scaling
        // Then apply ambient noise specific scaling
        adapted.R_covariance_sqrt *= noise_scale_from_ambient;

        adapted.raw_sensor_confidence = 1.0 - std::min(1.0, out_sensor_context.ambient_noise_db / 100.0); // Higher ambient noise -> lower confidence
        adapted.combined_confidence = adapted.raw_sensor_confidence * adapted.external_meta_score;
        return adapted;
    }

    /**
     * @brief Adapts a raw Vision measurement.
     */
    AdaptedMeasurement adaptVisionMeasurement(const RawSensorMeasurement& rawMeasurement,
                                              double external_meta_score,
                                              SensorContext& out_sensor_context) const {
        AdaptedMeasurement adapted;
        adapted.sensor_type = SensorType::VISION;
        adapted.external_meta_score = external_meta_score;

        // Expected raw_data: [pixel_x, pixel_y] (2D)
        // Expected raw_confidence_metrics: [pixel_quality, object_size_pixels] (2D)
        if (rawMeasurement.raw_data.size() < 2) {
            throw std::runtime_error("Vision raw_data must have 2 dimensions (pixel_x, pixel_y).");
        }
        double pixel_x_val = rawMeasurement.raw_data[0];
        double pixel_y_val = rawMeasurement.raw_data[1];

        checkFiniteAndThrow(Eigen::Vector2d(pixel_x_val, pixel_y_val), "Vision pixel coordinates");

        adapted.measurement_vector.resize(2);
        adapted.measurement_vector << pixel_x_val, // pixel_x
                                      pixel_y_val; // pixel_y

        // Populate SensorContext
        if (rawMeasurement.raw_confidence_metrics.size() > 0) {
            out_sensor_context.pixel_quality = rawMeasurement.raw_confidence_metrics[0];
        }
        if (rawMeasurement.raw_confidence_metrics.size() > 1) {
            out_sensor_context.object_size_pixels = rawMeasurement.raw_confidence_metrics[1];
        }

        // Calculate initial R_covariance_sqrt (R_base is variance)
        Eigen::Matrix2d R_base_variance = Eigen::Matrix2d::Identity();
        R_base_variance(0,0) = default_pixel_noise_std_dev_ * default_pixel_noise_std_dev_;
        R_base_variance(1,1) = default_pixel_noise_std_dev_ * default_pixel_noise_std_dev_;

        double quality_factor = 1.0 / std::max(kMinMetaScore, out_sensor_context.pixel_quality); // Lower quality -> higher noise
        double size_factor = 1.0 + std::max(0.0, (50.0 - out_sensor_context.object_size_pixels) / 50.0); // Smaller object -> higher noise
        // Call the general adaptive R calculation
        adapted.R_covariance_sqrt = calculateAdaptiveRCovarianceSqrt(R_base_variance, kNominalSNR, external_meta_score); // Use nominal SNR
        // Then apply vision-specific scaling
        adapted.R_covariance_sqrt *= quality_factor * size_factor;

        adapted.raw_sensor_confidence = out_sensor_context.pixel_quality * std::min(1.0, out_sensor_context.object_size_pixels / 100.0); // Combine
        adapted.combined_confidence = adapted.raw_sensor_confidence * adapted.external_meta_score;
        return adapted;
    }

    /**
     * @brief Adapts a raw Thermal measurement.
     */
    AdaptedMeasurement adaptThermalMeasurement(const RawSensorMeasurement& rawMeasurement,
                                               double external_meta_score,
                                               SensorContext& out_sensor_context) const {
        AdaptedMeasurement adapted;
        adapted.sensor_type = SensorType::THERMAL;
        adapted.external_meta_score = external_meta_score;

        // Expected raw_data: [pixel_x, pixel_y] (2D)
        // Expected raw_confidence_metrics: [pixel_quality, object_size_pixels, thermal_contrast] (3D)
        if (rawMeasurement.raw_data.size() < 2) {
            throw std::runtime_error("Thermal raw_data must have 2 dimensions (pixel_x, pixel_y).");
        }
        double pixel_x_val = rawMeasurement.raw_data[0];
        double pixel_y_val = rawMeasurement.raw_data[1];

        checkFiniteAndThrow(Eigen::Vector2d(pixel_x_val, pixel_y_val), "Thermal pixel coordinates");

        adapted.measurement_vector.resize(2);
        adapted.measurement_vector << pixel_x_val, // pixel_x
                                      pixel_y_val; // pixel_y

        // Populate SensorContext
        if (rawMeasurement.raw_confidence_metrics.size() > 0) {
            out_sensor_context.pixel_quality = rawMeasurement.raw_confidence_metrics[0];
        }
        if (rawMeasurement.raw_confidence_metrics.size() > 1) {
            out_sensor_context.object_size_pixels = rawMeasurement.raw_confidence_metrics[1];
        }
        // Assuming raw_confidence_metrics[2] is thermal contrast
        if (rawMeasurement.raw_confidence_metrics.size() > 2) {
            double thermal_contrast_val = rawMeasurement.raw_confidence_metrics[2];
            if (!std::isfinite(thermal_contrast_val)) {
                throw std::runtime_error("Thermal contrast must be finite.");
            }
            out_sensor_context.thermal_contrast = thermal_contrast_val; // Use dedicated field
        }

        // Calculate initial R_covariance_sqrt (R_base is variance)
        Eigen::Matrix2d R_base_variance = Eigen::Matrix2d::Identity();
        R_base_variance(0,0) = default_pixel_noise_std_dev_ * default_pixel_noise_std_dev_;
        R_base_variance(1,1) = default_pixel_noise_std_dev_ * default_pixel_noise_std_dev_;

        double quality_factor = 1.0 / std::max(kMinMetaScore, out_sensor_context.pixel_quality);
        double size_factor = 1.0 + std::max(0.0, (50.0 - out_sensor_context.object_size_pixels) / 50.0);
        double contrast_factor = 1.0 / std::max(kMinMetaScore, out_sensor_context.thermal_contrast); // Use dedicated field
        // Call the general adaptive R calculation
        adapted.R_covariance_sqrt = calculateAdaptiveRCovarianceSqrt(R_base_variance, kNominalSNR, external_meta_score); // Use nominal SNR
        // Then apply thermal-specific scaling
        adapted.R_covariance_sqrt *= quality_factor * size_factor * contrast_factor;

        adapted.raw_sensor_confidence = out_sensor_context.pixel_quality * std::min(1.0, out_sensor_context.object_size_pixels / 100.0) * out_sensor_context.thermal_contrast; // Use dedicated field
        adapted.combined_confidence = adapted.raw_sensor_confidence * adapted.external_meta_score;
        return adapted;
    }

    /**
     * @brief Adapts a raw Lidar measurement.
     */
    AdaptedMeasurement adaptLidarMeasurement(const RawSensorMeasurement& rawMeasurement,
                                             double external_meta_score,
                                             SensorContext& out_sensor_context) const {
        AdaptedMeasurement adapted;
        adapted.sensor_type = SensorType::LIDAR;
        adapted.external_meta_score = external_meta_score;

        // Expected raw_data: [range_m, azimuth_rad, elevation_rad] (3D)
        // Expected raw_confidence_metrics: [reflectivity, clutter_level, weather_index] (3D)
        if (rawMeasurement.raw_data.size() < 3) {
            throw std::runtime_error("Lidar raw_data must have 3 dimensions (range, azimuth, elevation).");
        }
        double range_val = rawMeasurement.raw_data[0];
        double azimuth_val = rawMeasurement.raw_data[1];
        double elevation_val = rawMeasurement.raw_data[2];

        if (range_val <= 0.0) {
            range_val = kMinCovarianceValue; // Clamp to a small positive value
        }
        checkFiniteAndThrow(Eigen::Vector2d(azimuth_val, elevation_val), "Lidar angles");

        adapted.measurement_vector.resize(3);
        adapted.measurement_vector << range_val, // range
                                      azimuth_val, // azimuth
                                      elevation_val; // elevation

        // Populate SensorContext
        if (rawMeasurement.raw_confidence_metrics.size() > 0) {
            double reflectivity_val = rawMeasurement.raw_confidence_metrics[0];
            if (!std::isfinite(reflectivity_val)) {
                throw std::runtime_error("Lidar reflectivity must be finite.");
            }
            out_sensor_context.raw_confidence_metrics[0] = reflectivity_val; // Reflectivity
        }
        if (rawMeasurement.raw_confidence_metrics.size() > 1) {
            double clutter_level_val = rawMeasurement.raw_confidence_metrics[1];
            if (!std::isfinite(clutter_level_val)) {
                throw std::runtime_error("Lidar clutter level must be finite.");
            }
            out_sensor_context.clutter_level = clutter_level_val;
        }
        if (rawMeasurement.raw_confidence_metrics.size() > 2) {
            double weather_index_val = rawMeasurement.raw_confidence_metrics[2];
            if (!std::isfinite(weather_index_val)) {
                throw std::runtime_error("Lidar weather index must be finite.");
            }
            out_sensor_context.weather_index = weather_index_val;
        }
        out_sensor_context.range_m = range_val;

        // Calculate initial R_covariance_sqrt (R_base is variance)
        Eigen::Matrix3d R_base_variance = Eigen::Matrix3d::Identity();
        R_base_variance(0,0) = default_lidar_range_noise_std_dev_ * default_lidar_range_noise_std_dev_;
        R_base_variance(1,1) = default_lidar_angle_noise_std_dev_ * default_lidar_angle_noise_std_dev_;
        R_base_variance(2,2) = default_lidar_angle_noise_std_dev_ * default_lidar_angle_noise_std_dev_;

        double reflectivity_factor = 1.0 / std::max(kMinMetaScore, out_sensor_context.raw_confidence_metrics[0]);
        double clutter_factor = 1.0 + out_sensor_context.clutter_level;
        double weather_factor = 1.0 + out_sensor_context.weather_index;
        // Call the general adaptive R calculation
        adapted.R_covariance_sqrt = calculateAdaptiveRCovarianceSqrt(R_base_variance, kNominalSNR, external_meta_score); // Use nominal SNR
        // Then apply lidar-specific scaling
        adapted.R_covariance_sqrt *= reflectivity_factor * clutter_factor * weather_factor;

        adapted.raw_sensor_confidence = out_sensor_context.raw_confidence_metrics[0] * (1.0 - out_sensor_context.clutter_level) * (1.0 - out_sensor_context.weather_index);
        adapted.combined_confidence = adapted.raw_sensor_confidence * adapted.external_meta_score;
        return adapted;
    }

    /**
     * @brief Adapts a raw Passive RF measurement.
     */
    AdaptedMeasurement adaptPassiveRFMeasurement(const RawSensorMeasurement& rawMeasurement,
                                                  double external_meta_score,
                                                  SensorContext& out_sensor_context) const {
        AdaptedMeasurement adapted;
        adapted.sensor_type = SensorType::PASSIVE_RF;
        adapted.external_meta_score = external_meta_score;

        // Expected raw_data: [azimuth_rad, elevation_rad] (2D)
        // Expected raw_confidence_metrics: [signal_strength_normalized, jamming_level] (2D)
        if (rawMeasurement.raw_data.size() < 2) {
            throw std::runtime_error("Passive RF raw_data must have 2 dimensions (azimuth, elevation).");
        }
        double azimuth_val = rawMeasurement.raw_data[0];
        double elevation_val = rawMeasurement.raw_data[1];

        checkFiniteAndThrow(Eigen::Vector2d(azimuth_val, elevation_val), "Passive RF angles");

        adapted.measurement_vector.resize(2);
        adapted.measurement_vector << azimuth_val, // azimuth
                                      elevation_val; // elevation

        // Populate SensorContext
        if (rawMeasurement.raw_confidence_metrics.size() > 0) {
            double signal_strength_val = rawMeasurement.raw_confidence_metrics[0];
            if (!std::isfinite(signal_strength_val)) {
                throw std::runtime_error("Passive RF signal strength must be finite.");
            }
            out_sensor_context.raw_confidence_metrics[0] = signal_strength_val; // Signal strength
        }
        if (rawMeasurement.raw_confidence_metrics.size() > 1) {
            double jamming_level_val = rawMeasurement.raw_confidence_metrics[1];
            if (!std::isfinite(jamming_level_val)) {
                throw std::runtime_error("Passive RF jamming level must be finite.");
            }
            out_sensor_context.jamming_level = jamming_level_val;
        }

        // Calculate initial R_covariance_sqrt (R_base is variance)
        Eigen::Matrix2d R_base_variance = Eigen::Matrix2d::Identity();
        R_base_variance(0,0) = default_passive_rf_angle_noise_std_dev_ * default_passive_rf_angle_noise_std_dev_;
        R_base_variance(1,1) = default_passive_rf_angle_noise_std_dev_ * default_passive_rf_angle_noise_std_dev_;

        double signal_strength_factor = 1.0 / std::max(kMinMetaScore, out_sensor_context.raw_confidence_metrics[0]);
        double jamming_factor = 1.0 + out_sensor_context.jamming_level;
        // Call the general adaptive R calculation
        adapted.R_covariance_sqrt = calculateAdaptiveRCovarianceSqrt(R_base_variance, kNominalSNR, external_meta_score); // Use nominal SNR
        // Then apply passive RF-specific scaling
        adapted.R_covariance_sqrt *= signal_strength_factor * jamming_factor;

        adapted.raw_sensor_confidence = out_sensor_context.raw_confidence_metrics[0] * (1.0 - out_sensor_context.jamming_level);
        adapted.combined_confidence = adapted.raw_sensor_confidence * adapted.external_meta_score;
        return adapted;
    }

};

#endif // SKYFILTER_MEASUREMENT_ADAPTER_HPP_
