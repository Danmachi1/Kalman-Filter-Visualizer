package com.filters;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

public class KFCoreLibrary {

    private static final Linker LINKER = Linker.nativeLinker();

    // Load from compiled native library: kfcore.dll (or .so/.dylib depending on platform)
    private static final SymbolLookup LOOKUP = SymbolLookup.libraryLookup("kfcore", Arena.global());

    public static final MethodHandle kalmanTakasu;
    public static final MethodHandle kalmanUDU;
    public static final MethodHandle decorrelate;
    public static final MethodHandle udu;

    static {
        try {
            kalmanTakasu = LINKER.downcallHandle(
                    LOOKUP.find("kalman_takasu").orElseThrow(),
                    FunctionDescriptor.of(JAVA_INT,
                            ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,  // x, P, dz, R, Ht
                            JAVA_INT, JAVA_INT,                           // stateDim, measDim
                            JAVA_FLOAT,                                   // threshold
                            ADDRESS                                       // outlier flag (nullable)
                    )
            );

            kalmanUDU = LINKER.downcallHandle(
                    LOOKUP.find("kalman_udu").orElseThrow(),
                    FunctionDescriptor.of(JAVA_INT,
                            ADDRESS, ADDRESS, ADDRESS,  // x, U, d
                            ADDRESS, ADDRESS, ADDRESS,  // z, R, Ht
                            JAVA_INT, JAVA_INT,         // stateDim, measDim
                            JAVA_FLOAT, JAVA_INT        // threshold, flags
                    )
            );

            decorrelate = LINKER.downcallHandle(
                    LOOKUP.find("decorrelate").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            ADDRESS,  // z
                            ADDRESS,  // Ht
                            ADDRESS,  // R
                            JAVA_INT, // stateDim
                            JAVA_INT  // measDim
                    )
            );

            udu = LINKER.downcallHandle(
                    LOOKUP.find("udu").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            ADDRESS, // input P
                            ADDRESS, // output U
                            ADDRESS, // output d
                            JAVA_INT // dim
                    )
            );

        } catch (Exception e) {
            throw new RuntimeException("❌ Failed to bind native methods", e);
        }
    }

    private KFCoreLibrary() {
        // Prevent instantiation
    }
    static MethodHandle kalmanUDUPredict = LINKER.downcallHandle(
            LOOKUP.find("kalman_udu_predict").orElseThrow(),
            FunctionDescriptor.ofVoid( // void return
                ADDRESS, // float* x
                ADDRESS, // float* U
                ADDRESS, // float* d
                ADDRESS, // const float* Phi
                ADDRESS, // const float* G
                ADDRESS, // const float* Q
                JAVA_INT, // int n
                JAVA_INT  // int r
            )
        );

     
}
