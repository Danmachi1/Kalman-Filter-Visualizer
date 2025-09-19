package com.filters;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;

import com.example.filtertest.KalmanFilterModule;

import static java.lang.foreign.ValueLayout.*;

/**
 * 🔴 TakasuWrapper – Native Kalman filter with predict + update using Panama FFI
 */
public class TakasuWrapper implements KalmanFilterModule {

    private static final int STATE_DIM = 4;   // x, y, vx, vy
    private static final int MEAS_DIM = 2;    // x, y
    private static final int NOISE_DIM = 2;   // process noise input

    private static final Linker linker = Linker.nativeLinker();
    private static final SymbolLookup lookup = SymbolLookup.libraryLookup("kfcore", Arena.global());

    private static MethodHandle kalmanTakasuHandle;
    private static MethodHandle kalmanPredictHandle;

    private final VarHandle floatHandle = JAVA_FLOAT.varHandle();

    // ── Shared memory buffers ──
    private MemorySegment x, P, dz, R, Ht, Phi, G, Q;

    private long lastTime = System.nanoTime();

    static {
        try {
            kalmanTakasuHandle = linker.downcallHandle(
                lookup.find("kalman_takasu").orElseThrow(),
                FunctionDescriptor.of(JAVA_INT,
                    ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
                    JAVA_INT, JAVA_INT,
                    JAVA_FLOAT, ADDRESS)
            );

            kalmanPredictHandle = linker.downcallHandle(
                lookup.find("kalman_predict").orElseThrow(),
                FunctionDescriptor.ofVoid(
                    ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
                    JAVA_INT, JAVA_INT)
            );
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    @Override
    public void init() {
        Arena arena = Arena.ofShared();

        x    = arena.allocate(JAVA_FLOAT.byteSize() * STATE_DIM);
        P    = arena.allocate(JAVA_FLOAT.byteSize() * STATE_DIM * STATE_DIM);
        dz   = arena.allocate(JAVA_FLOAT.byteSize() * MEAS_DIM);
        R    = arena.allocate(JAVA_FLOAT.byteSize() * MEAS_DIM * MEAS_DIM);
        Ht   = arena.allocate(JAVA_FLOAT.byteSize() * STATE_DIM * MEAS_DIM);
        Phi  = arena.allocate(JAVA_FLOAT.byteSize() * STATE_DIM * STATE_DIM);
        G    = arena.allocate(JAVA_FLOAT.byteSize() * STATE_DIM * NOISE_DIM);
        Q    = arena.allocate(JAVA_FLOAT.byteSize() * NOISE_DIM);

        writeFloatArray(x, new float[]{0f, 0f, 0f, 0f});
        writeFloatArray(P, identityMatrix(STATE_DIM, 500f));
        writeFloatArray(R, identityMatrix(MEAS_DIM, 0.01f)); // trust mouse
        writeFloatArray(Ht, new float[]{
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f
        });
        writeFloatArray(Q, new float[]{0.05f, 0.05f});
        writeFloatArray(G, new float[]{
            0.5f, 0.0f,
            0.0f, 0.5f,
            1.0f, 0.0f,
            0.0f, 1.0f
        });
    }

    @Override
    public void predict() {
        long now = System.nanoTime();
        float dt = (now - lastTime) / 1_000_000_000.0f;
        lastTime = now;

        writeFloatArray(Phi, new float[]{
            1f, 0f, dt, 0f,
            0f, 1f, 0f, dt,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 1f
        });

        try {
            kalmanPredictHandle.invokeExact(
                x, P, Phi, G, Q, STATE_DIM, NOISE_DIM
            );
        } catch (Throwable e) {
            System.err.println("❌ Predict step failed:");
            e.printStackTrace();
        }
    }

    @Override
    public void update(double xObs, double yObs) {
        float prevX = (float) floatHandle.get(x, 0L);
        float prevY = (float) floatHandle.get(x, 4L);

        float dx = (float) xObs - prevX;
        float dy = (float) yObs - prevY;
        writeFloatArray(dz, new float[]{dx, dy});

        MemorySegment chi2 = Arena.ofAuto().allocate(JAVA_FLOAT);

        try {
            int result = (int) kalmanTakasuHandle.invokeExact(
                x, P, dz, R, Ht,
                STATE_DIM, MEAS_DIM,
                0.0f, chi2
            );
            System.out.printf("✅ Takasu updated. Native return: %d\n", result);
        } catch (Throwable e) {
            System.err.println("❌ Takasu update failed:");
            e.printStackTrace();
        }

        float updatedX = (float) floatHandle.get(x, 0L);
        float updatedY = (float) floatHandle.get(x, 4L);
        System.out.printf("📍 Takasu Estimate: X=%.2f, Y=%.2f\n", updatedX, updatedY);
    }

    @Override
    public double getEstimateX() {
        return (float) floatHandle.get(x, 0L);
    }

    @Override
    public double getEstimateY() {
        return (float) floatHandle.get(x, 4L);
    }

    @Override
    public String getName() {
        return "Takasu KF";
    }

    // ────────────── Helpers ──────────────

    private void writeFloatArray(MemorySegment seg, float[] values) {
        for (int i = 0; i < values.length; i++) {
            floatHandle.set(seg, i * 4L, values[i]);
        }
    }

    private float[] identityMatrix(int dim, float diagValue) {
        float[] result = new float[dim * dim];
        for (int i = 0; i < dim; i++) {
            result[i * dim + i] = diagValue;
        }
        return result;
    }
}
