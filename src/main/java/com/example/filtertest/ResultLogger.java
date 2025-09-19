package com.example.filtertest;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/**
 * ResultLogger – Logs mouse and filter results to a CSV file.
 */
public class ResultLogger {

    private final PrintWriter writer;
    private final List<KalmanFilterModule> filters;

    public ResultLogger(String filename, List<KalmanFilterModule> filters) throws IOException {
        this.writer = new PrintWriter(new FileWriter(filename, false));
        this.filters = filters;

        // Write CSV header
        StringBuilder header = new StringBuilder("Time,MouseX,MouseY");
        for (KalmanFilterModule f : filters) {
            header.append(",").append(f.getName()).append("_X,").append(f.getName()).append("_Y");
        }
        writer.println(header.toString());
    }

    public void log(long timestamp, double mouseX, double mouseY) {
        StringBuilder line = new StringBuilder();
        line.append(timestamp).append(",")
            .append(mouseX).append(",")
            .append(mouseY);

        for (KalmanFilterModule f : filters) {
            line.append(",")
                .append(f.getEstimateX()).append(",")
                .append(f.getEstimateY());
        }

        writer.println(line.toString());
    }

    public void close() {
        writer.close();
    }
}
