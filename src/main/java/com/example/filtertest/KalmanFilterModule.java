package com.example.filtertest;

/**
 * KalmanFilterModule – interface that all filter wrappers must implement.
 * Each filter will predict and update based on (x, y) input.
 */
public interface KalmanFilterModule {

    /**
     * Called once to initialize the filter (optional).
     */
    default void init() {}

    /**
     * Perform the prediction step.
     */
    void predict();

    /**
     * Perform the update step using the latest observation.
     * @param x observed X
     * @param y observed Y
     */
    void update(double x, double y);

    /**
     * @return the current estimated X value
     */
    double getEstimateX();

    /**
     * @return the current estimated Y value
     */
    double getEstimateY();

    /**
     * @return name of the filter (for display/logging)
     */
    String getName();
}
