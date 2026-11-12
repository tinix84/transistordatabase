# Semiconductor gap audit, full reference-family census

Codex · 2026-11-11 · Baseline `e41cc8a` · Supplement to the seven-operation audit

The frozen inventory has **2,131 SC and ENCO definition records**. This supplement assigns an explicit disposition to every reference ID. Of these records, 674 are duplicate definitions and 87 are stubs under the original ledger rules. The remaining 1,370 records span 16 families. Their original roles identify 1,234 source definitions and 136 test references. Some source-role records also sit in test paths or represent test setup methods.

The wider review identified five more semiconductor operations. They cover inverse harmonic reconstruction, paired waveform statistics, coupled two-phase Boost waveforms, signal conditioning and continuous-temperature diode fitting. Entries 8 through 12 supplement the seven operations in the [first audit](semiconductor-next-gap-audit-2026-11-11.md). The [ID census](semiconductor-reference-family-census-2026-11-11.json) identifies definitions whose individual behavior still needs review. Complete accounting does not establish per-function numerical equivalence.

## 1. How to read the census

The census joins definition IDs to the November 10 verification record and the ten later workflow receipts. It also links the seven initial entries, five additional operations and seven broader domain boundaries. A function can support several workflows. The JSON retains these links separately from its primary disposition.

The primary disposition distinguishes a selected operation, an earlier qualified workflow, an earlier scoped capability, a domain boundary, a test, a duplicate, a stub and a residual review item. An earlier result applies only to the input domain and outputs in its receipt. The category for unreviewed context explicitly preserves uncertainty about the individual definition.

The family table records inspected examples and shared application boundaries. Reading a representative function supports that family observation. It does not justify declaring every function in that family equivalent.

## 2. All 16 reference families have an explicit boundary

| Family | Existing capability or reference role | Boundary at the baseline |
|---|---|---|
| orchestration-context | Project and simulator integration contains real circuit-simulation and EMC functions alongside configuration helpers. | An application runner qualifies one generated double-pulse job. PLECS remote execution, QSPICE analyses, project component sizing, and standards-specific masks need separate models or licensed tools. |
| data-model | Native device records, source evidence and finite configuration lists have application counterparts. | Generic circuit objects, indicator metadata and historical importer classes do not imply device-level equivalence. The bounded Boost grid extends one application solver. |
| time-series | Sample statistics, forward spectra, export, and gate allocation have qualified public operations. | Hysteresis, frequency masks, inverse harmonic reconstruction and paired-trace statistics exceed the baseline. Group delay requires a transfer response. Generic dataframe arithmetic remains a separate scope. |
| converter | The application models specified converters and conduction and switching losses in supported domains. | The scalar Boost feedback calculation has a bounded grid extension. Named CLLC, phase-shifted bridge and multiport project models require their own circuits and inputs. No general topology solver equivalence was established. |
| import-export | Native JSON, PLECS device XML, selected MATLAB inputs, measurement tables and three-column energy maps have qualified paths. | Generic XML helpers, simulation of whole PLECS circuits, remote project databases and unsupported source variants have distinct semantics. |
| circuit-indicators | The grid merit and per-device thermal rollup qualify selected sourced operating points and losses. | General circuit orchestration, arbitrary component indicators and derived project metrics exceed these contracts. |
| regression-lut | Conditioned device fitting supplies measured bounded channel, diode and energy fits. | Generic feature maps, trained project coefficients, dataset learners and lookup-table extrapolation require separate data and verification. |
| plot | Web views show measured curves, device fits, supported operating points and spectra. | The web does not promise every reference plotting method or graphical backend. A missing plot method alone is insufficient evidence of a scientific gap. |
| analytical | The registered gate and Biela analytical estimates have a limited verified domain. | The archive includes syntactically defective analytical code. Its formulas and physical drain waveforms require independent evidence. It cannot serve as a runnable acceptance oracle. |
| dpt | Double-pulse scheduling, netlist generation, selected RAW and log import, pulse windows and energy integration have scoped evidence. | Starting a local simulator job and linking its fresh artifacts to web measurement import extends those operations. Vendor device accuracy needs separate qualification. |
| waveform-loss | Signed conduction and switching energy, explicit event models and per-device attribution have scoped evidence. | Command-state assumptions differ from a physical current-path measurement. Hysteretic event detection and feedback from device loss at varying temperature have separate inputs. |
| gate-timing | The lumped RC/Miller estimate calculates charge intervals at declared drive conditions. | Turn-on settling after the plateau is separate. A drain-voltage or loss waveform would require device and parasitic evidence. |
| device-parameters | Conditioned static fits, capacitance decompositions, energy queries and selected slew estimates have scoped scientific evidence. | Data accessors, defective historical formulas and unsupported conditions cannot establish measured semiconductor behavior. |
| parallel-thermal | The application supports selected parallel paths, fixed-sink temperatures, thermal transients and source-conditioned reverse sharing. | Simultaneous sourced contacts and feedback from native temperature-conditioned curves require separate operations. |
| catalog-fom | Workbook figures of merit, electrical proxies and catalog scatter apply to known entries. | A linked catalog view does not establish physical loss or price. Comparable switching-balanced Boost ranking requires a common source population. |
| digitization | The browser digitizer calibrates images and records digitized curve coordinates. | All six former missing ledger symbols have a capability correspondence. Accuracy depends on the supplied calibration and image. GUI method-by-method equivalence remains unverified. |

## 3. The spectral check includes frequency bands

Entry 2 also covers a user-declared frequency band's root-mean-square (RMS) current. ENCO `attach_fs_indicators` summarizes a thresholded harmonic subset above 150 kHz. Its cumulative squared-magnitude expression and sparse frequency steps do not establish energy in joules. The baseline provides normalized mean-square contributions for every Fourier bin. An explicit band sum would connect these values to a user-selected range.

Use closed frequency bounds and all bins in the band. Report the frequency coordinates, counts, window, detrending and sampling interval. Keep a small finite RMS amplitude representable even when its square would underflow. A band without sampled frequencies needs a coverage indication.

**Independent check.** Use eight rectangular, undetrended samples at 8 Hz. For the signal `1 + 2 cos(2πn/8) + cos(4πn/8)` A, the 0, 1 and 2 Hz mean-square contributions are 1, 2 and 0.5 A². The 1–2 Hz band gives √2.5 A RMS. All bins give √3.5 A RMS. An even-length Nyquist bin has unit multiplicity.

## 4. Five further semiconductor operations have independent checks

### 8. Inverse harmonic reconstruction

transistordatabase/waveform_losses.py::series_spectrum returns normalized complex bins, N, sample interval and phase reference. Baseline has no inverse waveform operation. Use every one-sided peak complex coefficient or an explicit subset. Require original N, sample interval and first timestamp. DC and the even-length Nyquist coefficient are undoubled. Interior coefficients use one-sided normalization. Supply N explicitly to the inverse transform for odd lengths. Return retained and residual traces. With a Hann window or mean removal, identify the actually reconstructed windowed or detrended projection.

**Independent check.** For rectangular, undetrended N=7 and N=8 inputs, recover every original sample at the original times. Test DC, interior sine phase, even Nyquist, empty subset, and selected-bin plus residual closure.

**Reference limit.** The reference's ascending frequency-bound predicate does not select the intended interval. It infers a time grid from rfftfreq and omits N from irfft. Those behaviors cannot validate the result.

### 9. Paired waveform statistics

DPTManager.vue::comparisonWaveformPlots overlays selected acquisition records. Existing model-fit RMSE measures residuals for a fitted model. Neither operation computes statistics for an arbitrary pair of stored traces. Select two acquisitions, one quantity, compatible units, comparable recorded conditions and an identical original time grid. Report sample RMSE in the signal unit and Pearson correlation without units. State the sample count and window. Return an undefined reason for zero variance. An unmatched grid needs an explicit policy before comparison.

**Independent check.** For times [0,1,2] s and currents [0,1,2] and [2,1,0] A, discrete RMSE is sqrt(8/3) A and correlation is -1. Piecewise-linear time-weighted RMS error would be 2/sqrt(3) A. Identical nonconstant traces give zero error and correlation +1.

**Reference limit.** The reference operates on same-index columns. Its correlation divides by the product of deviations without a constant-trace guard.

### 10. Coupled two-phase Boost waveforms

The baseline scalar Boost has one inductor. The reference solves a two-winding inductance matrix under phase-shifted switching. A device-loss calculation needs the current in each leg.

The bounded case uses two ideal Boost legs in continuous conduction. Specify the inductance matrix, winding polarity, voltage intervals, phase shift and periodic or initial current conditions. Declare the direct-current sharing rule. Reuse existing conditioned loss calculations for each device only after its signed current and switching events are known.

**Independent checks.** For two 1 mH inductors, 0.5 mH mutual inductance and +1 V on each winding, each current slope is 2/3 kA/s. With zero mutual inductance, each slope is 1 kA/s. At zero coupling, equal inductors, 50 % duty and half-period phase shift, equal opposing ripple slopes cancel in the input current. Check volt-second balance, positive definiteness and the continuous-conduction boundary.

**Reference limit.** The specific winding solution does not establish a general state-space simulator. Winding-current derivatives have units of A/s. The reference labels its derivative traces in A.

### 11. Signal conditioning and additional spectral windows

The reference contains a median, a circular sample shift, subtraction of a rolling mean, and Hamming and Blackman windows. The corresponding source wrappers do not all select the requested signal correctly. A usable operation needs a named input series, sample alignment and source provenance.

**Independent checks.** For samples [1, 5, 2, 4], the median is 3.5. A one-sample circular shift to the right gives [4, 1, 5, 2]. A trailing two-sample mean leaves residuals [unknown, 2, −1.5, 1]. The initial sample has no complete window. For spectral windows, compare the sampled cosine coefficients with a direct discrete Fourier transform. Record the window convention, coherent gain and window power gain.

**Reference limit.** A shift in samples requires no interpolation. An angle-to-sample conversion needs a declared rounding rule. Missing initial window values require explicit coverage. Stored raw measurements keep their original values.

The original bridgeless power-factor-correction function accepts precalculated waveforms. The application already includes the `sc-bridgeless` converter preset with source-specific device roles, duty, signs and multiplicity. Its tests exercise reverse-path activation. The unavailable source waveform inputs limit circuit-accuracy claims. They do not establish an additional missing preset or solver.

### 12. Bounded continuous-temperature diode fitting

ENCO `Diode.train` fits a forward-voltage model to measured current and junction temperature. Its `get_vf` method accepts these two quantities at prediction time. The baseline diode fitting service requires an exact temperature and gate slice. Its forward and inverse operations cannot evaluate a temperature between slices.

A separate empirical model can fit voltage as a bilinear function of current and temperature at fixed gate and test conditions. Require four observed training corners and a rank-four design. Use the same covered rectangle for prediction. Require positive voltage slope versus current at both temperature limits. Require nonnegative forward voltage at all four corners.

**Independent check.** Use training currents of 0 and 10 A and temperatures of 25 and 125 °C. Supply forward voltages of 0.40 and 0.60 V at 25 °C. At 125 °C, supply 0.30 and 0.60 V. The bilinear fit gives 0.475 V at 5 A and 75 °C. The inverse gives 5 A at 0.475 V and 75 °C. At 75 °C, only 0.35–0.60 V supports inversion. A held-out corner cannot establish the training rectangle.

**Reference limit.** The ENCO feature map and regressor do not establish this bilinear law. Compare predictions with independent held-out observations before reporting accuracy for a named device. Keep the existing exact-slice method as a distinct query contract.

## 5. Seven broader domains require separate evidence

### Group delay of a transfer response

A single measured current spectrum has no input/output transfer function. A transfer-characterization workflow needs related input/output signals, common time origin, unwrapped phase, excitation coverage and a finite angular-frequency interval.

**Independent check.** A pure transfer delay tau has phase -2πf*tau. Its group delay is tau seconds.

**Reference limit.** The reference differences wrapped phase versus linear frequency without the factor 1/(2π). Citing it does not establish measured group delay.

### Standard-specific emissions limits

A current-bin mask operates in peak amperes. A receiver measurement and a standard-specific voltage limit need a different physical quantity and acquisition method. Separate synthetic user masks from standard-specific detector characteristics, units, bandwidths and test setup. Reference interpolation uses logarithmic frequency and level.

**Independent check.** A standards-specific implementation would need authorized edition-specific limit definitions and independently verified receiver conditions.

**Reference limit.** No standard tables or compliance results were validated by this audit.

### PLECS and QSPICE simulation

PLECS semiconductor XML interchange and the proposed LTspice double-pulse job do not execute arbitrary PLECS or QSPICE circuits. PLECS whole-model RPC simulation and QSPICE AC or transient runs need their own executable, model and result contract.

**Independent check.** A real supported installation and a fixture with known AC or transient response would qualify each backend.

**Reference limit.** No PLECS or QSPICE run was executed in this audit.

### Project-specific converter optimization

The grid merit workflow combines comparable metrics. The bounded Boost solver only represents its declared topology. The project CLLC and current-fed bridge optimizers depend on their own soft-switching indicators, device models, operating spaces and model validation.

**Independent check.** Model-specific circuit, operating points and independent measurements or simulator evidence are necessary for a new topology.

**Reference limit.** A generic finite candidate list or merit score does not establish converter physics.

### Passive component screening

The reference reports electrical proxies for capacitor, inductor and resistor designs. For example, its inductor loss index uses inductance times squared RMS current. A physical loss in watts also needs a dissipative model. Passive-component material, temperature and magnetic assumptions fall outside the audited semiconductor loss domain.

### Signal conditioning in a general series container

Entry 11 covers selected conditioning operations. The general series container also implements data storage, arithmetic and other transformations. Its rolling-mean method fills incomplete initial windows with zero. A sum of squared samples requires units and time weights before it can represent energy.

### Generic lookup and extrapolation

The reference can extrapolate using a gradient from an end segment of a lookup table. The application bounds measured queries to their condition and current domains. Validating a particular extrapolation would require a named physical model, an extrapolation range and independent data.

## 6. Interpretation and records

A missing source method can identify framework plumbing, an unavailable model, a different physical quantity or a missing scientific operation. The census keeps those cases separate. For example, fixture definitions help identify expected behavior. A fixture alone does not establish measured device behavior.

The wider pass reviewed the source modules and symbols of 560 initially unresolved records. The earlier parser flagged 19 definitions. They include 18 Biela definitions and one invalid bridgeless Python translation. The Biela group contains analytical operations, constructors and a zero-return placeholder. The application already implements scoped Biela calculations.

The other 16 historical records include storage wrappers, model-training methods and signal conditioning. The total of 35 identifies a review cohort. Each member has its own qualification. The machine index preserves the original IDs separately from the twelve operation assignments. Other representation and subsystem classifications describe scope, without claiming per-function test results.

The per-definition review identifies the source body, its application counterpart and the remaining qualification for each record. [Qualification report](semiconductor-reference-qualification-2026-11-11.md) · [35-record crosswalk](semiconductor-reference-qualification-2026-11-11.json)

The source ID is the primary key. The [original inventory](../sprints/semiconductor-function-gap-ledger.json) provides its source path, symbol and file hash. The [November 10 capability record](semiconductor-parity-verification-2026-11-10.json) provides earlier classifications and domains. The [November 11 verification record](semiconductor-gap-closure-2026-11-11.json) identifies the ten earlier scoped workflows. The issue tracker owns subsequent implementation status.
