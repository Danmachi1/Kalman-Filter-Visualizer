package com.example.filtertest;

import javafx.animation.AnimationTimer;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;

import java.util.*;

/**
 * FilterVisualizerPane – draws real-time mouse and filter paths.
 */
public class FilterVisualizerPane extends Pane {

    private final Canvas canvas = new Canvas(1000, 800);
    private final GraphicsContext gc = canvas.getGraphicsContext2D();

    private final MouseTracker mouseTracker = new MouseTracker();
    private final FilterRunner filterRunner;
    private final List<KalmanFilterModule> filters;
    private final Map<String, List<double[]>> filterPaths = new HashMap<>();
    private final List<double[]> mousePath = new ArrayList<>();

    // 🔵 Extended color list for 10+ filters
    private final Color[] COLORS = {
            Color.RED, Color.BLUE, Color.YELLOW, Color.ORANGE,
            Color.PURPLE, Color.BROWN, Color.CYAN, Color.MAGENTA,
            Color.DARKGRAY, Color.PINK
    };

    public FilterVisualizerPane(List<KalmanFilterModule> filters) {
        this.filters = filters;
        this.filterRunner = new FilterRunner(filters);
        getChildren().add(canvas);

        // Initialize path storage
        for (KalmanFilterModule f : filters) {
            filterPaths.put(f.getName(), new ArrayList<>());
        }

        // 🟩 Hook mouse movement to canvas
        canvas.setOnMouseMoved(e -> {
            double x = clamp(e.getX(), 0, canvas.getWidth());
            double y = clamp(e.getY(), 0, canvas.getHeight());
            mouseTracker.setX(x);
            mouseTracker.setY(y);
        });

        canvas.setOnMouseDragged(e -> {
            double x = clamp(e.getX(), 0, canvas.getWidth());
            double y = clamp(e.getY(), 0, canvas.getHeight());
            mouseTracker.setX(x);
            mouseTracker.setY(y);
        });
    }

    public void start() {
        new AnimationTimer() {
            @Override
            public void handle(long now) {
                double mouseX = mouseTracker.getX();
                double mouseY = mouseTracker.getY();

                // Record actual path
                mousePath.add(new double[]{mouseX, mouseY});

                // Update filters
                filterRunner.step(mouseX, mouseY);

                // Record each filter's output
                for (KalmanFilterModule f : filters) {
                    filterPaths.get(f.getName()).add(new double[]{
                            f.getEstimateX(),
                            f.getEstimateY()
                    });
                }

                System.out.println("Filter names at init:");
                for (KalmanFilterModule f : filters) {
                    System.out.println(" - " + f.getName());
                }

                draw();
            }
        }.start();
    }

    private void draw() {
        System.out.println("Currently drawing:");
        for (Map.Entry<String, List<double[]>> entry : filterPaths.entrySet()) {
            System.out.println(" > " + entry.getKey() + " → " + entry.getValue().size() + " points");
        }

        int last = mousePath.size() - 1;
        if (last < 1) return;

        double[] prevMouse = mousePath.get(last - 1);
        double[] currMouse = mousePath.get(last);

        // Draw mouse trail
        gc.setStroke(Color.LIMEGREEN);
        gc.setLineWidth(2);
        gc.strokeLine(prevMouse[0], prevMouse[1], currMouse[0], currMouse[1]);
        gc.setFill(Color.LIMEGREEN);
        gc.fillOval(currMouse[0] - 2, currMouse[1] - 2, 4, 4);

        // Draw each filter trail
        int i = 0;
        for (KalmanFilterModule f : filters) {
            List<double[]> path = filterPaths.get(f.getName());
            if (path.size() < 2) continue;

            double[] prev = path.get(path.size() - 2);
            double[] curr = path.get(path.size() - 1);

            Color c = COLORS[i % COLORS.length];
            double offset = i * 0.5;

            // Offset stroke to make overlapping trails visible
            gc.setStroke(c);
            gc.setLineWidth(3);
            gc.strokeLine(prev[0] + offset, prev[1] + offset, curr[0] + offset, curr[1] + offset);

            // Dot
            gc.setFill(c);
            gc.fillOval(curr[0] + offset - 3, curr[1] + offset - 3, 6, 6);

            // Label
            gc.setFill(Color.BLACK);
            gc.fillText(f.getName(), curr[0] + offset + 5, curr[1] + offset - 5);

            i++;
        }

        // Draw legend
        gc.setFill(Color.BLACK);
        gc.fillText("🟩 Mouse Path", 10, 20);
        i = 0;
        for (KalmanFilterModule f : filters) {
            gc.setFill(COLORS[i % COLORS.length]);
            gc.fillText("⬤ " + f.getName(), 10, 40 + 20 * i);
            i++;
        }
    }

    public MouseTracker getMouseTracker() {
        return mouseTracker;
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
}
