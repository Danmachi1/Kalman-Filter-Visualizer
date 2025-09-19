package com.example.filtertest;

import java.util.List;

/**
 * FilterRunner – Updates all filters with the latest observation.
 */
public class FilterRunner {

    private final List<KalmanFilterModule> filters;

    public FilterRunner(List<KalmanFilterModule> filters) {
        this.filters = filters;
        for (KalmanFilterModule f : filters) {
            f.init();
        }
    }

    /**
     * Run one predict/update step for each filter.
     * @param x current observed X
     * @param y current observed Y
     */
    public void step(double x, double y) {
        for (KalmanFilterModule f : filters) {
            f.predict();
            f.update(x, y);
        }
    }
}
