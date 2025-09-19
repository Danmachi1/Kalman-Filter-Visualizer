package com.example.filtertest;

import javafx.scene.Scene;
import javafx.stage.Window;

/**
 * MouseTracker – Tracks the latest known mouse position inside the JavaFX window.
 * Used as the "true" reference for filter comparisons.
 */
public class MouseTracker {

    private static volatile double mouseX = 0;
    private static volatile double mouseY = 0;

    public MouseTracker() {
        // Nothing to initialize yet
    }

    public void attachToScene(Scene scene) {
        scene.setOnMouseMoved(e -> {
            mouseX = e.getX();
            mouseY = e.getY();
        });

        scene.setOnMouseDragged(e -> {
            mouseX = e.getX();
            mouseY = e.getY();
        });
    }

    public double getX() {
        return mouseX;
    }

    public double getY() {
        return mouseY;
    }
    public void setX(double x) {
        this.mouseX = x;
    }

    public void setY(double y) {
        this.mouseY = y;
    }

    public void set(double x, double y) {
        this.mouseX = x;
        this.mouseY = y;
    }
}
