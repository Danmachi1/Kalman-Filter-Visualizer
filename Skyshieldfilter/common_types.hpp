// common_types.hpp
#ifndef SKYFILTER_COMMON_TYPES_HPP_
#define SKYFILTER_COMMON_TYPES_HPP_

#include <Eigen/Dense> // For Eigen::VectorXd, Eigen::MatrixXd
#include <string>      // For sensor names, etc.
#include <vector>      // For std::vector

// --- Enums and Structs ---

// Enum for motion models (used by IMMManager, not directly by MeasurementAdapter)
enum class MotionModelType {
    CONSTANT_VELOCITY,
    CONSTANT_TURN,
    CONSTANT_ACCELERATION,
    SINGER // Added Singer model type
};

// Enum for sensor types
enum class SensorType {
    RADAR,
    RF,
    VISION,
    IMU,
    ACOUSTIC,
    LIDAR,      // Added Lidar as per measurement_models.hpp
    THERMAL,    // Added Thermal as per measurement_models.hpp
    PASSIVE_RF, // Added Passive RF as per measurement_models.hpp
    UNKNOWN
};

// Enum for track quality (reflecting state machine)
enum class TrackQuality {
    DORMANT,     // No track yet, or deleted
    SUSPICIOUS,  // Ghost track initiated, low confidence
    ACTIVE,      // Consistent weak support, building confidence
    CONFIRMED,   // High confidence, strong consistent support (STABLE)
    GHOST_TRACKED, // Lost measurements, in particle fallback
    LOST         // Faded out, deleted
};

/**
 * @brief Structure for raw sensor data as received from the sensor fusion engine.
 * This is the input to the MeasurementAdapter.
 */
struct RawSensorMeasurement {
    SensorType type;
    double timestamp;
    // Generic container for raw sensor data.
    // The interpretation of elements depends on 'type'.
    // e.g., for RADAR: [range, azimuth, elevation, RCS_dBsm]
    // e.g., for IMU: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    Eigen::VectorXd raw_data;

    // Generic container for raw confidence metrics from the sensor.
    // The interpretation depends on 'type'.
    // e.g., for RADAR/RF: [SNR_db]
    // e.g., for VISION/THERMAL: [pixel_quality, object_size_pixels]
    // e.g., for THERMAL: [pixel_quality, object_size_pixels, thermal_contrast]
    Eigen::VectorXd raw_confidence_metrics;
};

/**
 * @brief Structure for sensor context, used by Measurement Models for adaptive noise.
 * This is populated by the MeasurementAdapter.
 *
 * @note The default constructor explicitly initializes all numeric fields to 0.0
 * to ensure a safe default state and prevent uninitialized values.
 */
struct SensorContext {
    double snr_db = 0.0;             // Signal-to-Noise Ratio (dB)
    double range_m = 0.0;            // Range to target (meters)
    double ambient_noise_db = 0.0;   // Ambient noise level (dB)
    double clutter_level = 0.0;      // Clutter density/level (e.g., 0.0 to 1.0)
    double weather_index = 0.0;      // Weather severity index (e.g., 0.0 for clear, 1.0 for severe)
    double pixel_quality = 0.0;      // Pixel quality / clarity (e.g., 0.0 to 1.0 for vision/thermal)
    double object_size_pixels = 0.0; // Perceived object size (pixels) for vision/thermal
    double jamming_level = 0.0;      // Jamming level (e.g., 0.0 to 1.0) for RF
    double thermal_contrast = 0.0;   // Thermal contrast (e.g., 0.0 to 1.0) for thermal sensors
    // Raw confidence metrics directly passed through for model-specific use
    Eigen::VectorXd raw_confidence_metrics;

    // Default constructor explicitly initializes Eigen::VectorXd
    SensorContext() : raw_confidence_metrics(0) {} // Initialize with 0 size
};

/**
 * @brief Structure for adapted measurement data, ready for the Kalman filter.
 * This is the output of the MeasurementAdapter.
 */
struct AdaptedMeasurement {
    Eigen::VectorXd measurement_vector; // z_k: The actual measurement vector (e.g., [x,y,z] or [range,azimuth,elevation])
    Eigen::MatrixXd R_covariance_sqrt;  // S_R: Square root of the measurement noise covariance matrix
    double raw_sensor_confidence;      // Normalized sensor confidence [0,1], derived by adapter (e.g., from SNR)
    double external_meta_score;        // Trust score for this sensor [0,1], provided externally
    double combined_confidence;        // raw_sensor_confidence * external_meta_score
    SensorType sensor_type;              // Original sensor type
    // Optional: Add specific signature data if applicable, e.g., microDoppler_features
};

/**
 * @brief Structure for environmental context, provided by external map/environment module.
 * Used by IMMManager and SystemModels for adaptive behavior.
 */
struct EnvironmentalContext {
    double building_density = 0.0;        // e.g., 0.0 (open) to 1.0 (dense urban)
    double nearest_obstacle_distance = 1e6; // meters, large value if no obstacles nearby
    // Add other relevant environmental factors as needed
};

/**
 * @brief Data structure for the final filter output.
 */
struct FilterOutput {
    double x, y, z;
    double vx, vy, vz; // Velocity components
    double confidence_score; // Composite confidence score [0,1]
    TrackQuality quality;    // Track quality enum
    SensorType last_sensor_support; // Type of last sensor that successfully updated the track
    Eigen::MatrixXd covariance; // Full covariance matrix for external use
};

#endif // SKYFILTER_COMMON_TYPES_HPP_
