# Boost switching-loss frequency grids

The opt-in mode is `switching-loss-balanced`, for Boost CCM only. It reuses the existing scalar calculation. The result is a **power-balanced included-loss estimate**. Averaged inductor voltage closure is not enforced or proved. Signed voltage and power residuals remain visible. This feature does not infer DCM, interpolate missing device conditions, or establish SOA or thermal safety.

## Frequency line

POST `/api/topology/switching-frequency-grid` with the existing scalar request and a `sweep` object. Set `x_axis` to `f_sw`, supply finite positive `x_min < x_max`, `x_points` from 2 to 32, and an existing metric. Omit the second axis. Frequencies use binary64 convex coordinates; collapsed coordinates are rejected.

Each row retains its frequency, `qualified` or `unsupported` status, diagnostic, scalar operating point if available, selected metric, and both residuals. Unsupported metrics and points use null values rather than fabricated zeros. The chart keeps gaps. The summary uses the existing requested-frequency row if qualified, otherwise the first qualified row; `summary_frequency_hz` and `summary_basis` identify that choice. No extra base calculation occurs. All-unsupported grids have no accepted summary.

## Same-frequency candidate populations

POST `/api/topology/switching-frequency-pareto` with the existing boundary, candidates, and sourced catalog. Add `boundary.mode: "switching-loss-balanced"`. Each candidate supplies the existing on-gate condition plus `v_g_off`, `r_g_on`, and `r_g_off`. Both recorded gate fields must meet the scalar exact-condition contract. Older records with an unknown gate-pair condition do not become qualified by this feature.

An optional root `frequencies_hz` list declares 2 to 32 strictly increasing finite positive Hz coordinates. The complete candidate population is screened independently at each frequency. Candidate count times frequency count must not exceed 32. With no list, one shared fixed-frequency population is screened. No best-frequency selection, cross-frequency dominance, or aggregate objective is inferred.

The new loss objective is `included_loss_w`, alongside package-body volume and sourced cost. Missing curves, source lookup, ratings, or catalog metadata keep a candidate in the response as ineligible; unknown cost is not zero. Exact objective ties remain non-dominated. Optional `boundary.voltage_residual_tolerance_v` filters eligibility by the absolute signed voltage residual. It has no default, never removes rows, and does not prove physical closure.

Each distinct TDB identity is resolved once per opt-in API request, including missing-source diagnostics, and reused across candidates and frequencies. Pure calculations accept resolved core objects and do not invent raw-file hashes. Source conditions and scalar provenance remain in each operating point.

## Bounds and transport

The two new endpoints stream-bound request bodies to4MiB before parsing and use original numeric-token precision validation before typed float conversion. Calls with new grid fields on the legacy endpoints return422 and identify the bounded endpoint. Legacy scalar, sweep, and Pareto defaults keep their previous transport and result behavior. New frequency requests reject nonzero underflow, overflow, unsupported precision, and isolated Unicode surrogates.

Before scalar work, relevant source families are limited to 128 records each, each graph to 10000 points, and the unique input devices to a combined 20000 sample-coordinate pairs. Source text is limited to 2048 characters per field. Catalog metadata is limited to 65536 encoded bytes. New results are limited to 4 MiB. File-backed source snapshots read at most16MiB per identity and reserve at most32MiB per request, including read sentinels, before parsing. The adapter reuses the existing strict source reader and reports the actual raw-file SHA separately from scalar curve provenance. These bounds qualify support; evidence is not truncated.

The existing topology GUI exposes frequency-only lines and explicit same-frequency candidate populations. JSON exports retain the entire grid, including ineligible rows; scalar CSV includes status, diagnostics, and both signed residuals. Changing inputs invalidates old results and exports.
