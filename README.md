# Kalman Filter Visualizer

A JavaFX desktop comparison project for 2D position tracking. Mouse input
provides a reference trajectory, and Java wrappers call native filtering
libraries through the Foreign Function and Memory API.

## Filter comparison

[MainApp](src/main/java/com/example/filtertest/MainApp.java) instantiates
five wrappers:

| Wrapper | Native library base name | Export prefix |
| --- | --- | --- |
| EKFWrapper | kalman_ekf | ekf |
| SREKFWrapper | kalman_srekf | srekf |
| UKFWrapper | kalman_ukf | ukf |
| SRUKFWrapper | kalman_ukf | srukf |
| TakasuWrapper | see KFCoreLibrary | see KFCoreLibrary |

EKF/UKF and their square-root variants are different estimation methods,
not automatically a ranking from worse to better. The square-root UKF
exports are in `kalman_ukf.dll`; a separate `kalman_srukf.dll` is not
required by the current wrapper.

The UI draws the mouse path and colored filter trails, with labels and a
legend. [FilterRunner](src/main/java/com/example/filtertest/FilterRunner.java)
initializes each module and calls predict/update for each observation.
The canvas is 1000 by 800 pixels.

## Build and runtime status

[pom.xml](pom.xml) targets **JDK 22**, JavaFX 21.0.1 and Windows JavaFX
artifacts. A Java 21 runtime alone does not satisfy that compiler target.

The repository includes Windows DLLs. Native loading also depends on
architecture, dependent libraries, search paths and the exported ABI.
A clean cross-platform build of all native components is not supplied.

With a matching JDK 22, Maven, Windows JavaFX environment and verified
native-library setup, the configured JavaFX launcher is:

```sh
mvn javafx:run
```

The entry point is `com.example.filtertest.MainApp`. If native loading
fails, check the actual library paths and exports rather than renaming
unrelated binaries. Native access may require
`--enable-native-access=ALL-UNNAMED` in the launched JVM.

These are prerequisites and the configured launcher, not a claim that
the complete graphical/native application has been tested on a fresh
Windows installation.

## Logging and measurement caveats

[ResultLogger](src/main/java/com/example/filtertest/ResultLogger.java)
can write timestamped mouse/filter coordinates to CSV, but the current
MainApp/visualizer does **not** instantiate it. Starting the application
does not automatically produce `results.csv`.

Mouse input is an interactive reference, not an independently measured
ground-truth trajectory. No reproducible benchmark in this README
establishes a 5–10x speedup, superior numerical accuracy, fixed frame
rate or sub-millisecond timing. Such claims need a controlled input,
timing method, configuration and measured results.

Current limitations include unbounded path-history lists, per-frame
console output, no explicit native-handle shutdown in MainApp, and no
frame-time propagation from FilterRunner into the wrappers' setDt methods.
Keep sessions short and treat this as a development/research project.

## Source regression check

Python 3 is sufficient for the narrow library-name consistency check:

```sh
python3 -m unittest discover -s tests -v
```

It verifies that four wrappers load the same library name they use for
symbol lookup, and that the corresponding DLL files exist. The original
SREKF wrapper loaded the EKF library despite looking up SREKF symbols;
the regression detects that mismatch.

Validation: one test with four wrapper subcases passes after the fix;
the SREKF subcase fails on the original source. Static inspection of the
bundled PE export tables confirms the srekf and shared ukf/srukf symbol
names. **This does not execute the binaries or validate their numerical
behavior.** Full JavaFX/native execution remains unverified in this pass.

## Attribution and licensing

The JavaFX visualization and native wrappers use bundled third-party
filter implementations and examples. Each component retains its own
attribution and license:

- [KFCore](KFCore/LICENSE), Jan Zwiener. Its included terms require:
  “This product includes KFCore, developed by Jan Zwiener.”
- [kalman-master](kalman-master/LICENSE.txt), Markus Herb, MIT license
- [TinyEKF](TinyEKF/LICENSE.md), Simon D. Levy, MIT license
- [iekf](iekf/README.md), credited there to Easton Potokar and Kalin Norman
- [uNavINS](uNavINS/README.md), with upstream references in its own README

Keep all existing attribution and license files. Check each component's
terms before redistributing source or binaries. No new blanket license
is granted by this README.

The project focuses on visualizing and comparing these libraries through
a common Java interface. Numerical accuracy and performance comparisons
still need reproducible benchmarks.
