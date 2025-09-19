package com.example.filtertest;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;
import java.util.ArrayList;
import java.util.List;

import com.filters.EKFWrapper;
import com.filters.SREKFWrapper;
import com.filters.SRUKFWrapper;
import com.filters.TakasuWrapper;
import com.filters.UKFWrapper;

/**
 * MainApp – JavaFX entry point. Initializes and launches the visualizer with filters.
 */
public class MainApp extends Application {

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Kalman Filter Visualizer");

        // List of active filters – add your filter modules here
        List<KalmanFilterModule> filters = new ArrayList<>();
        filters.add(new EKFWrapper());
        filters.add(new SREKFWrapper());
        filters.add(new UKFWrapper());
        filters.add(new SRUKFWrapper());
        filters.add(new TakasuWrapper());

        FilterVisualizerPane visualizer = new FilterVisualizerPane(filters);
        Scene scene = new Scene(visualizer, 1000, 800);
        MouseTracker mouseTracker = visualizer.getMouseTracker();
        mouseTracker.attachToScene(scene);
        primaryStage.setScene(scene);
        primaryStage.show();

        visualizer.start(); // Begin mouse tracking and filter updates
    }

    public static void main(String[] args) {
        launch(args);
    }
}
