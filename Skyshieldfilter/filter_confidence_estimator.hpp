// filter_confidence_estimator.hpp
#ifndef SKYFILTER_FILTER_CONFIDENCE_ESTIMATOR_HPP_
#define SKYFILTER_FILTER_CONFIDENCE_ESTIMATOR_HPP_

#include <Eigen/Dense>   // For Eigen::MatrixXd
#include <cmath>         // For std::exp, std::log, std::min, std::max
#include <vector>        // For std::vector
#include <stdexcept>     // For std::runtime_error

// Project-specific Headers
#include "common_types.hpp" // For TrackQuality, SensorType

/**
 * @brief The FilterConfidenceEstimator class is responsible for calculating a
 * composite confidence score for a track and managing its quality state machine.
 * This aligns with Section 5.1 of the skyfilter_plan.
 *
 * It combines various internal filter metrics into a single, continuous confidence
 * score [0.0, 1.0] and uses this, along with event triggers, to transition the
 * track through different `TrackQuality` states.
 */
class FilterConfidenceEstimator {
public:
    TrackQuality current_track_quality_; // The current state of the track
    double confidence_score_;            // Continuous confidence score [0.0, 1.0]

    // State machine parameters (timeouts, thresholds)
    double suspicious_timeout_s_;   // Time in SUSPICIOUS state before fading
    double active_timeout_s_;       // Time in ACTIVE state before transitioning to GHOST_TRACKED
    double confirmed_timeout_s_;    // Time in CONFIRMED state before transitioning to GHOST_TRACKED
    double ghost_tracked_timeout_s_; // Time in GHOST_TRACKED before fading

    int consistent_updates_threshold_active_; // Number of consistent updates to go from SUSPICIOUS to ACTIVE
    int consistent_updates_threshold_confirmed_; // Number of consistent updates to go from ACTIVE to CONFIRMED

    double confidence_threshold_active_;   // Confidence score to become ACTIVE
    double confidence_threshold_confirmed_; // Confidence score to become CONFIRMED

    // Internal counters/trackers
    double time_in_current_state_s_;
    int consecutive_consistent_updates_;
    int consecutive_outlier_measurements_;
    double total_outlier_rate_; // Moving average or cumulative
    double time_since_last_measurement_s_; // From SkyFilterDetectX

    /**
     * @brief Constructor for FilterConfidenceEstimator.
     * @param suspicious_timeout_s Timeout for SUSPICIOUS state (seconds).
     * @param active_timeout_s Timeout for ACTIVE state (seconds).
     * @param confirmed_timeout_s Timeout for CONFIRMED state (seconds).
     * @param ghost_tracked_timeout_s Timeout for GHOST_TRACKED state (seconds).
     * @param consistent_updates_threshold_active Number of consistent updates for SUSPICIOUS to ACTIVE.
     * @param consistent_updates_threshold_confirmed Number of consistent updates for ACTIVE to CONFIRMED.
     * @param confidence_threshold_active Confidence score threshold for ACTIVE.
     * @param confidence_threshold_confirmed Confidence score threshold for CONFIRMED.
     */
    FilterConfidenceEstimator(double suspicious_timeout_s = 5.0,
                              double active_timeout_s = 10.0,
                              double confirmed_timeout_s = 20.0,
                              double ghost_tracked_timeout_s = 15.0,
                              int consistent_updates_threshold_active = 3,
                              int consistent_updates_threshold_confirmed = 5,
                              double confidence_threshold_active = 0.5,
                              double confidence_threshold_confirmed = 0.8)
        : current_track_quality_(TrackQuality::DORMANT),
          confidence_score_(0.0),
          suspicious_timeout_s_(suspicious_timeout_s),
          active_timeout_s_(active_timeout_s),
          confirmed_timeout_s_(confirmed_timeout_s),
          ghost_tracked_timeout_s_(ghost_tracked_timeout_s),
          consistent_updates_threshold_active_(consistent_updates_threshold_active),
          consistent_updates_threshold_confirmed_(consistent_updates_threshold_confirmed),
          confidence_threshold_active_(confidence_threshold_active),
          confidence_threshold_confirmed_(confidence_threshold_confirmed),
          time_in_current_state_s_(0.0),
          consecutive_consistent_updates_(0),
          consecutive_outlier_measurements_(0),
          total_outlier_rate_(0.0),
          time_since_last_measurement_s_(std::numeric_limits<double>::max()) // Initially no measurement
    {}

    /**
     * @brief Calculates a composite confidence score for the track.
     * This score is a weighted combination of various filter metrics.
     *
     * @param covariance The current combined state covariance matrix (from IMM or PF).
     * @param time_since_last_update Time in seconds since the last measurement update.
     * @param consistent_updates_count Number of recent consistent updates.
     * @param outlier_rate Moving average or cumulative rate of rejected measurements.
     * @param last_sensor The type of the last sensor that provided a successful update.
     * @param particle_filter_spread If in GHOST_TRACKED mode, the determinant of the particle filter's covariance.
     * @param imm_model_probabilities Optional: Vector of current IMM model probabilities.
     * @param imm_model_types Optional: Vector of IMM model types corresponding to probabilities.
     * @return A continuous confidence score from 0.0 to 1.0.
     */
    double calculateConfidence(const Eigen::MatrixXd& covariance,
                               double time_since_last_update,
                               int consistent_updates_count,
                               double outlier_rate,
                               SensorType last_sensor, // Passed for potential sensor-specific weighting
                               double particle_filter_spread = 0.0, // 0 if not in PF mode
                               const Eigen::VectorXd* imm_model_probabilities = nullptr,
                               const std::vector<MotionModelType>* imm_model_types = nullptr) {
        
        // Factor 1: Covariance Determinant (Inverse of uncertainty)
        // Use log-determinant for numerical stability, especially for high dimensions.
        // A smaller determinant means higher confidence.
        double log_det_cov = 0.0;
        if (covariance.rows() > 0) {
            Eigen::LLT<Eigen::MatrixXd> llt_cov(covariance);
            if (llt_cov.info() == Eigen::Success) {
                log_det_cov = 2.0 * llt_cov.matrixL().diagonal().array().log().sum();
            } else {
                // If covariance is not PD, it indicates severe divergence/issue.
                // Assign a very low confidence.
                return 0.0;
            }
        } else {
            return 0.0; // Zero-dimensional covariance, no confidence
        }
        // Heuristic: scale log_det to a [0,1] range, smaller log_det -> higher factor
        // Adjust the scaling factor (e.g., 0.1) and offset (e.g., 5.0) based on expected log_det range.
        double cov_det_factor = std::exp(-0.1 * log_det_cov - 5.0); // Example scaling
        cov_det_factor = std::max(0.0, std::min(1.0, cov_det_factor)); // Clamp to [0,1]

        // Factor 2: Recency of Last Update (Exponential decay)
        double recency_factor = std::exp(-0.05 * time_since_last_update); // Decay rate 0.05 (tuned)
        recency_factor = std::max(0.0, std::min(1.0, recency_factor));

        // Factor 3: Number of Recent Consistent Updates
        double consistency_factor = std::min(1.0, consistent_updates_count / static_cast<double>(consistent_updates_threshold_confirmed_ + 2)); // Max at threshold + some buffer
        consistency_factor = std::max(0.0, std::min(1.0, consistency_factor));

        // Factor 4: Outlier Rejection Rate (Higher rate -> lower confidence)
        double outlier_factor = 1.0 - std::min(1.0, outlier_rate); // outlier_rate assumed to be [0,1]
        outlier_factor = std::max(0.0, std::min(1.0, outlier_factor));

        // Factor 5: Particle Filter Spread (if in GHOST_TRACKED mode)
        double pf_spread_factor = 1.0;
        if (particle_filter_spread > 0.0) { // Only if PF is active and spread is meaningful
            // Similar to cov_det_factor, but for PF spread
            double log_pf_spread = std::log(particle_filter_spread); // Assuming spread is determinant of PF covariance
            pf_spread_factor = std::exp(-0.1 * log_pf_spread - 5.0); // Example scaling
            pf_spread_factor = std::max(0.0, std::min(1.0, pf_spread_factor));
        }

        // Factor 6: IMM Model Probabilities (Bias towards more dynamic models)
        double imm_model_bias_factor = 0.0;
        if (imm_model_probabilities && imm_model_types && imm_model_probabilities->size() == imm_model_types->size()) {
            for (size_t i = 0; i < imm_model_probabilities->size(); ++i) {
                // Assign higher weight to more complex/dynamic models if their probabilities are high
                if ((*imm_model_types)[i] == MotionModelType::CONSTANT_ACCELERATION) {
                    imm_model_bias_factor += (*imm_model_probabilities)[i] * 0.3; // Higher weight for CA
                } else if ((*imm_model_types)[i] == MotionModelType::CONSTANT_TURN) {
                    imm_model_bias_factor += (*imm_model_probabilities)[i] * 0.2; // Medium weight for CT
                } else if ((*imm_model_types)[i] == MotionModelType::CONSTANT_VELOCITY) {
                    imm_model_bias_factor += (*imm_model_probabilities)[i] * 0.1; // Lower weight for CV
                }
                // Singer model could also be weighted if applicable
            }
        }
        imm_model_bias_factor = std::max(0.0, std::min(1.0, imm_model_bias_factor));


        // Weighted sum of factors to get composite confidence score
        // Weights need careful tuning based on desired system behavior
        confidence_score_ = (0.30 * cov_det_factor +
                             0.20 * recency_factor +
                             0.15 * consistency_factor +
                             0.15 * outlier_factor +
                             0.10 * pf_spread_factor +
                             0.10 * imm_model_bias_factor);

        return std::max(0.0, std::min(1.0, confidence_score_)); // Ensure score is clamped [0,1]
    }

    /**
     * @brief Updates the track quality state machine based on confidence and event triggers.
     * This method advances the state of the track based on its performance and external events.
     *
     * @param delta_time The time elapsed since the last update to the state machine.
     * @param has_consistent_measurements True if recent measurements are consistent (passed gating, low NIS).
     * @param is_ghost_initiated True if this track was just initiated as a ghost track.
     * @param has_lost_measurements_recently True if measurements have been absent for a while.
     * @param is_diverging True if the filter's consistency checks indicate divergence.
     * @param is_particle_filter_active True if the track is currently managed by a particle filter (stealth or recovery).
     * @param is_pf_converged True if the active particle filter has converged enough for SRUKF handover.
     */
    void updateState(double delta_time,
                     bool has_consistent_measurements,
                     bool is_ghost_initiated,
                     bool has_lost_measurements_recently,
                     bool is_diverging,
                     bool is_particle_filter_active,
                     bool is_pf_converged) {

        time_in_current_state_s_ += delta_time;
        time_since_last_measurement_s_ += delta_time; // This should be updated by SkyFilterDetectX on actual measurement receipt

        // Update consecutive consistent/outlier counters
        if (has_consistent_measurements) {
            consecutive_consistent_updates_++;
            consecutive_outlier_measurements_ = 0; // Reset outlier counter
            time_since_last_measurement_s_ = 0.0; // Reset time since last measurement
        } else {
            consecutive_consistent_updates_ = 0; // Reset consistent counter
            // Only increment outlier if a measurement was actually processed and rejected/outliered
            // (This logic might need to be driven from the IMM update loop)
            // For now, assume if not consistent, it contributes to outlier.
            consecutive_outlier_measurements_++;
        }
        // Simple update for total_outlier_rate_ (could be moving average)
        total_outlier_rate_ = static_cast<double>(consecutive_outlier_measurements_) / 
                              (consecutive_consistent_updates_ + consecutive_outlier_measurements_ + 1e-6);


        TrackQuality next_state = current_track_quality_;

        switch (current_track_quality_) {
            case TrackQuality::DORMANT:
                if (is_ghost_initiated) {
                    next_state = TrackQuality::SUSPICIOUS;
                    time_in_current_state_s_ = 0.0; // Reset timer for new state
                }
                break;

            case TrackQuality::SUSPICIOUS:
                if (confidence_score_ > confidence_threshold_active_ &&
                    consecutive_consistent_updates_ >= consistent_updates_threshold_active_) {
                    next_state = TrackQuality::ACTIVE;
                    time_in_current_state_s_ = 0.0;
                } else if (time_in_current_state_s_ >= suspicious_timeout_s_ || is_diverging) {
                    next_state = TrackQuality::LOST; // Fade out
                    time_in_current_state_s_ = 0.0;
                }
                break;

            case TrackQuality::ACTIVE:
                if (confidence_score_ > confidence_threshold_confirmed_ &&
                    consecutive_consistent_updates_ >= consistent_updates_threshold_confirmed_) {
                    next_state = TrackQuality::CONFIRMED;
                    time_in_current_state_s_ = 0.0;
                } else if (time_since_last_measurement_s_ >= active_timeout_s_ && !is_particle_filter_active) {
                    next_state = TrackQuality::GHOST_TRACKED; // Transition to PF fallback
                    time_in_current_state_s_ = 0.0;
                } else if (is_diverging || time_in_current_state_s_ >= (active_timeout_s_ * 2.0)) { // Hard timeout if no measurements and no PF
                    next_state = TrackQuality::LOST;
                    time_in_current_state_s_ = 0.0;
                }
                break;

            case TrackQuality::CONFIRMED:
                if (time_since_last_measurement_s_ >= confirmed_timeout_s_ && !is_particle_filter_active) {
                    next_state = TrackQuality::GHOST_TRACKED; // Transition to PF fallback
                    time_in_current_state_s_ = 0.0;
                } else if (is_diverging || time_in_current_state_s_ >= (confirmed_timeout_s_ * 2.0)) { // Hard timeout
                    next_state = TrackQuality::LOST;
                    time_in_current_state_s_ = 0.0;
                }
                break;

            case TrackQuality::GHOST_TRACKED:
                if (is_pf_converged && has_consistent_measurements) {
                    // Re-acquired and PF has converged, transition back to ACTIVE or CONFIRMED based on confidence
                    if (confidence_score_ > confidence_threshold_confirmed_) {
                        next_state = TrackQuality::CONFIRMED;
                    } else {
                        next_state = TrackQuality::ACTIVE;
                    }
                    time_in_current_state_s_ = 0.0;
                } else if (time_in_current_state_s_ >= ghost_tracked_timeout_s_ || is_diverging) {
                    next_state = TrackQuality::LOST; // Fade out
                    time_in_current_state_s_ = 0.0;
                }
                break;

            case TrackQuality::LOST:
                // Stays LOST until explicitly re-initiated as a new track (e.g., DORMANT -> SUSPICIOUS)
                break;
        }
        current_track_quality_ = next_state;
    }

    /**
     * @brief Resets the estimator to a DORMANT state, clearing all counters.
     * This is typically called when a track is deleted or re-initialized.
     */
    void reset() {
        current_track_quality_ = TrackQuality::DORMANT;
        confidence_score_ = 0.0;
        time_in_current_state_s_ = 0.0;
        consecutive_consistent_updates_ = 0;
        consecutive_outlier_measurements_ = 0;
        total_outlier_rate_ = 0.0;
        time_since_last_measurement_s_ = std::numeric_limits<double>::max();
    }

    // Public getters for current state and confidence
    TrackQuality getTrackQuality() const { return current_track_quality_; }
    double getConfidenceScore() const { return confidence_score_; }
};

// The TrackQualityEstimator class is now largely redundant if FilterConfidenceEstimator
// directly manages the state machine and exposes the current_track_quality_.
// However, if a simple mapping from a raw confidence score (not necessarily the one
// calculated by FilterConfidenceEstimator's internal logic) to a TrackQuality enum
// is still desired for external use or quick checks, it can be kept.
// For now, we'll keep a minimal version that maps a score to quality, assuming
// the primary state management is done by FilterConfidenceEstimator.
/**
 * @brief The TrackQualityEstimator provides a simple mapping from a numerical
 * confidence score to a `TrackQuality` enum.
 *
 * @note In the refined architecture, `FilterConfidenceEstimator` directly manages
 * the `TrackQuality` state machine. This class serves as a utility for external
 * components that might need to interpret a raw confidence score into a quality enum
 * without needing the full state machine logic.
 */
class TrackQualityEstimator {
public:
    /**
     * @brief Maps a continuous confidence score to a discrete TrackQuality enum.
     * @param confidenceScore The numerical confidence score [0.0, 1.0].
     * @return The corresponding TrackQuality enum.
     */
    TrackQuality getTrackQuality(double confidenceScore) const {
        if (confidenceScore >= 0.8) return TrackQuality::CONFIRMED;
        if (confidenceScore >= 0.5) return TrackQuality::ACTIVE;
        if (confidenceScore >= 0.1) return TrackQuality::SUSPICIOUS;
        return TrackQuality::LOST; // Default for very low confidence
    }
};

#endif // SKYFILTER_FILTER_CONFIDENCE_ESTIMATOR_HPP_
