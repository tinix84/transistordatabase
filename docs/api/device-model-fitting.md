# Fit and evaluate a device model

Open the [device model fitting panel](http://localhost:8002/index.html?view=analytical&model=device-fitting). Select on-resistance, diode voltage/current, or switching energy.

1. Supply one identifier, source, series, split assignment, coordinate set and observed value for each row.
2. Assign rows to `train` or `holdout` before fitting. Choose polynomial degree 1, 2 or 3.
3. Fit the model. Inspect each support interval and the separate training and holdout diagnostics.
4. Enter a query within the recorded support and supply its fixed conditions.
5. Download the exact model JSON for later evaluation. The evidence exports retain the review data and hashes.

Each residual is the observed value minus the prediction. R² requires sufficient variation in the observed values. A single holdout observation returns a null R².

## Training and evaluation API

Send this illustrative resistance fixture to `POST /api/models/train`. Resistance is in ohms and temperature is in degrees Celsius.

```json
{
  "family": "ron", "degree": 1, "fixed_conditions": {"v_g_v": 12, "current_a": null},
  "observations": [
    {"id":"r25","source":"Hand fixture","series":"A","split":"train","coordinates":{"t_j_c":25},"value":0.010},
    {"id":"r75","source":"Hand fixture","series":"A","split":"train","coordinates":{"t_j_c":75},"value":0.0125},
    {"id":"r125","source":"Hand fixture","series":"A","split":"train","coordinates":{"t_j_c":125},"value":0.015},
    {"id":"h50","source":"Hand fixture","series":"A","split":"holdout","coordinates":{"t_j_c":50},"value":0.01150}
  ]
}
```

The holdout prediction is 0.01125 Ω. The residual is +0.00025 Ω. The response includes `model_json` and its UTF-8 SHA-256 digest in `evidence.model_sha256`. Preserve the supplied numeric text when importing a saved model.

For `POST /api/models/evaluate`, supply the saved model object under `model`. Supply one or more queries under `queries`. For this example, use the following query array.

```json
[
  {"id":"q50","output":"resistance","coordinates":{"t_j_c":50,"v_g_v":12,"current_a":null}}
]
```

## Conditions and interpretation

Diode fits support forward voltage and inverse current within the applicable slice. An explicitly unknown fixed diode temperature is distinct from 0 °C. Temperature must be finite when it is a fitted coordinate.

Train turn-on and turn-off energy separately. Supply a measured or declared nominal anchor and isolated sweeps along current, voltage, temperature and gate resistance. The estimator multiplies a fitted current baseline by normalized voltage, temperature and resistance factors. A holdout that changes several coordinates tests the separability assumption at that point.

The supplied browser energy fixture uses a 20 A, 400 V, 25 °C, 4 Ω anchor. Its 15 A, 500 V, 75 °C, 6 Ω query gives 0.002578125 J. These hand fixtures verify arithmetic. Device accuracy requires independent observations at the intended conditions. Saved models support replay with this application’s fitting method.
