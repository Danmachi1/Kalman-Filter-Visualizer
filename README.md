# Kalman-Filter-Visualizer
Comparing EKF, UKF, SR-EKF, SR-UKF, and Custom Takasu Variants

Kalman Filter Visualizer: Comparing EKF, UKF, SR-EKF, SR-UKF, and Custom Takasu Variants
Overview
This project is a real-time visualization and benchmarking tool designed to test and compare multiple Kalman filter implementations, including both standard variants and custom enhancements I've developed or optimized over time. Kalman filters are essential for state estimation in noisy environments, such as tracking object positions in robotics, navigation systems, or sensor fusion applications. Here, we focus on 2D position-velocity tracking (x, y, vx, vy states) using mouse movements as the "ground truth" input—a simple yet effective way to simulate dynamic, real-world trajectories with inherent noise from human input.
The core goal is to visually and quantitatively evaluate how different filters handle prediction, update, and smoothing under varying conditions. I've included:

Extended Kalman Filter (EKF): A classic linearization-based approach for non-linear systems.
Square-Root Extended Kalman Filter (SR-EKF): An enhanced, numerically stable variant using square-root Cholesky factorization to reduce error propagation.
Unscented Kalman Filter (UKF): A sigma-point sampling method for better handling of non-linearities without Jacobian computations.
Square-Root Unscented Kalman Filter (SR-UKF): My optimized version combining UKF's sampling with square-root decomposition for improved stability and efficiency.
Takasu Kalman Filter: A custom implementation inspired by Takasu's robust filtering techniques, incorporating outlier rejection and decorrelation for high-noise scenarios.

These filters represent a mix of established algorithms and my personal contributions—such as numerical stability tweaks in the square-root variants and custom process/measurement models in Takasu—to push the boundaries of performance in real-time applications. By running them side-by-side, you can observe divergences in estimation accuracy, lag, and smoothness, making it ideal for research, education, or fine-tuning your own filters.
How It Works
The application is built as a JavaFX desktop app, providing an interactive canvas where you control the "true" trajectory via mouse movements (dragged or moved). This simulates a noisy observation stream, as mouse input isn't perfectly linear or noise-free. The filters process this input in real-time, estimating the state (position and velocity) while accounting for process noise (e.g., acceleration changes) and measurement noise (e.g., cursor jitter).
Core Components

Input Handling (MouseTracker):

Tracks the mouse position (x, y) within the 1000x800 canvas.
Updates occur on mouseMoved and mouseDragged events, clamped to canvas bounds to prevent edge artifacts.
Serves as the "ground truth" reference for visualization and logging.


Filter Orchestration (FilterRunner and KalmanFilterModule Interface):

FilterRunner manages a list of filter instances, initializing them and stepping through the predict-update cycle for each.
Each filter implements the KalmanFilterModule interface:

init(): Optional one-time setup (e.g., initial covariance).
predict(): Propagates the state forward using a process model (e.g., constant velocity with dt-based integration).
update(x, y): Incorporates the new observation, computing Kalman gain and correcting the state/covariance.
getEstimateX/Y(): Retrieves the current position estimate for rendering.
getName(): For labeling in the UI and logs.


The cycle runs at ~60 FPS via JavaFX's AnimationTimer, ensuring smooth real-time updates.


Visualization (FilterVisualizerPane):

A custom Pane with a Canvas for rendering.
Draws:

Mouse Path: A green trail (line segments + dot) showing the actual input trajectory.
Filter Paths: Colored trails (with slight offsets for visibility) for each filter's estimates, plus dots and labels (e.g., "EKF", "SRUKF").
Legend: Bottom-left key mapping colors to filters.


Uses an extended color palette (10+ colors) to support adding more filters without clashes.
Trails are incrementally drawn (only the latest segment per frame) to avoid performance hits from full path redrawing.


Logging (ResultLogger):

Outputs to a CSV file (results.csv by default) with columns: Time, MouseX, MouseY, [FilterName]_X, [FilterName]_Y.
Timestamped at each step for post-analysis (e.g., RMSE computation in tools like Python/MATLAB).
Automatically flushes on close.


Main Entry Point (MainApp):

Launches the JavaFX stage, instantiates the filter list, wires up the visualizer, and starts the animation loop.
Easily extensible: Add/remove filters in the filters.add() list.



Workflow in a Single Frame

Capture current mouse (x, y).
Append to mouse path history.
For each filter:

Call predict(): Advance state using dt (time since last frame) and process model (e.g., Φ transition matrix for constant velocity).
Call update(x, y): Compute innovation, Kalman gain, and posterior state.
Store estimate in path history.


Render incremental lines/dots/labels.
Log the step.

This setup allows immediate visual feedback: Smoother, closer-following paths indicate better tuning, while divergences highlight weaknesses (e.g., EKF struggling with high non-linearity).
Why Native C DLLs via Panama FFI?
For computationally intensive tasks like matrix inversions, Cholesky decompositions, and sigma-point propagations in Kalman filters, pure Java can introduce overhead from garbage collection, interpreted loops, and less-optimized linear algebra. To achieve sub-millisecond predict/update times suitable for real-time viz (and scalable to embedded systems), I've offloaded the core math to hand-optimized C libraries, exposed as DLLs/SOs.
Performance Rationale

Speed: C's direct memory access and compiler optimizations (e.g., SIMD via intrinsics) yield 5-10x faster matrix ops than Java's equivalents (e.g., via JAMA or custom loops). In benchmarks, this keeps frame rates stable even with 10+ filters.
Numerical Stability: Custom C impls (like square-root filters) use proven algorithms (e.g., UDU decomposition) to avoid covariance blow-up from floating-point errors—harder to guarantee in high-level langs.
Modularity: Separate concerns—Java handles UI/logic, C does math—easing porting (e.g., to Android via JNI) or testing isolated filter units.

Integration Details

Panama FFI (JDK 21+): Modern, zero-overhead alternative to JNI. Uses Linker and MethodHandle for downcalls to C symbols (e.g., ekf_predict(handle)).

Each wrapper (e.g., EKFWrapper) loads its DLL (kalman_ekf.dll), binds functions like create(), predict(), update(), and state().
Memory segments (e.g., for state vectors) are allocated via Arena for safe, scoped native access—no manual pointers!
Handles like MemorySegment abstract raw pointers, with auto-closing for cleanup.


C Libraries (not included here, but compiled separately):

kalman_ekf.dll: Standard EKF with Jacobian-based linearization.
kalman_srekf.dll: SR-EKF using Cholesky for sqrt(P) propagation.
kalman_ukf.dll: UKF with sigma points and weighted stats.
kalman_srukf.dll: SR-UKF, my enhancement with UDU for positive-definiteness.
kfcore.dll: Shared core for Takasu (e.g., kalman_takasu() with outlier flagging via Mahalanobis distance).


Build Notes: Place DLLs in java.library.path (or extract to temp dir). Compile C with MSVC/GCC, linking math libs (e.g., -lm). Cross-platform via CMake if needed.

This hybrid approach maximizes Java's ecosystem strengths (UI, ease) while harnessing C's raw power—perfect for prototyping filters before full deployment.
Getting Started

Prerequisites: JDK 21+, JavaFX (via Maven/Gradle), native libs compiled.
Build/Run: mvn compile exec:java or IDE run MainApp.
Extend: Implement KalmanFilterModule for new filters; add to MainApp list.
Analyze: Run, move mouse wildly, check results.csv for metrics.

License & Contributions
MIT License. Pull requests for new filters or optimizations welcome!
