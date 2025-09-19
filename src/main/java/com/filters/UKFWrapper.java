package com.filters;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

/** JDK 21-compatible Panama wrapper for one Kalman filter variant. */
public final class UKFWrapper implements AutoCloseable, com.example.filtertest.KalmanFilterModule {

    /* ── customise these two for each variant ─────────────────────────── */
    private static final String LIB_NAME = "kalman_ukf";   // DLL / SO base name
    private static final String PREFIX   = "ukf";          // "srekf", "ukf", "srukf"

    /* ── Panama plumbing ──────────────────────────────────────────────── */
    private static final Linker LINKER = Linker.nativeLinker();
    private static final Arena  ARENA  = Arena.ofShared();
    static {
        System.loadLibrary("kalman_ukf"); // Looks for kalman_ekf.dll in java.library.path
    }

    private static final SymbolLookup LIB =
            SymbolLookup.libraryLookup(LIB_NAME, Arena.global());

    private static MethodHandle fnR(MemoryLayout ret, String name, MemoryLayout... args) {
        return LINKER.downcallHandle(LIB.find(PREFIX + '_' + name).orElseThrow(),
                                     FunctionDescriptor.of(ret, args));
    }
    private static MethodHandle fnV(String name, MemoryLayout... args) {
        return LINKER.downcallHandle(LIB.find(PREFIX + '_' + name).orElseThrow(),
                                     FunctionDescriptor.ofVoid(args));
    }

    /* native symbols */
    private static final MethodHandle
        CREATE  = fnR(ADDRESS,     "create"),                // ( ) -> ptr
        DESTROY = fnV("destroy",   ADDRESS),                 // (ptr)
        INIT    = fnV("init",      ADDRESS, JAVA_DOUBLE, JAVA_DOUBLE),
        SET_DT  = fnV("set_dt",    ADDRESS, JAVA_DOUBLE),
        PREDICT = fnV("predict",   ADDRESS),                 // no dt param!
        UPDATE  = fnV("update",    ADDRESS, JAVA_DOUBLE, JAVA_DOUBLE),
        STATE   = fnV("state",     ADDRESS, ADDRESS);        // (ptr, out[4])

    /* for JDK 21 array allocation */
    private static final SequenceLayout FOUR_DOUBLES = MemoryLayout.sequenceLayout(4, JAVA_DOUBLE);

    /* instance state ---------------------------------------------------- */
    private final MemorySegment handle;
    private boolean initialised;
    private double ex, ey;

    public UKFWrapper() {
        try { handle = (MemorySegment) CREATE.invokeExact(); }
        catch (Throwable t) { throw new RuntimeException(t); }
    }

    /* KalmanFilterModule interface -------------------------------------- */
    @Override public void init() {/* first update sets the initial state */ }

    @Override public void predict() {
        try { PREDICT.invokeExact(handle); pull(); }
        catch(Throwable t){ throw new RuntimeException(t); }
    }

    @Override public void update(double x, double y) {
        try {
            if (!initialised) {
                INIT.invokeExact(handle, x, y);
                System.out.printf("UKFpreupdate: %.2f, %.2f%n", x, y);

                initialised = true;
            }
            UPDATE.invokeExact(handle, x, y);
            System.out.printf("UKFpostupdate: %.2f, %.2f%n", x, y);

            pull();
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    private void pull() throws Throwable {
        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment buf = scratch.allocate(FOUR_DOUBLES); // ← JDK 21-compatible
            STATE.invokeExact(handle, buf);
            ex = buf.getAtIndex(JAVA_DOUBLE, 0);
            ey = buf.getAtIndex(JAVA_DOUBLE, 1);
        }
    }

    @Override public double getEstimateX() { return ex; }
    @Override public double getEstimateY() { return ey; }
    @Override public String  getName()     { return PREFIX.toUpperCase(); }

    /* housekeeping ------------------------------------------------------ */
    public void setDt(double dtSec) {
        try { SET_DT.invokeExact(handle, dtSec); }
        catch(Throwable t){ throw new RuntimeException(t); }
    }

    @Override public void close() {
        try { DESTROY.invokeExact(handle); } catch (Throwable ignored) {}
    }
}
