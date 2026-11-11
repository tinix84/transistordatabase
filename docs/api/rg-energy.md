# Switching energy and gate resistance

Use the local calculator to estimate energy at a gate resistance or find the
gate resistance for a target energy. Open the [gate-resistance energy panel](http://localhost:8002/index.html?view=analytical&model=parameters#rg-energy).

1. Select the stored device and the turn-on, turn-off or reverse-recovery branch.
2. Select the recorded representation and a source context from the inventory.
3. Enter current in amperes and junction temperature in degrees Celsius.
4. For a forward query, enter gate resistance in ohms. For an inverse query, enter target energy in joules.
5. Calculate the result. Download the evidence JSON or query and source CSV.

## Repository inventory

`GET /api/transistors/{transistor_id}/rg-energy` returns source locators, fixed
conditions and bounded previews. An unavailable row retains a diagnostic.
`graph_r_e` gives energy versus gate resistance at a fixed current.
`graph_i_e` gives energy versus current at a recorded gate resistance.
Keep all nine condition fields from the selected source, including explicit
`null` values.

## Read-only queries

Send a JSON request to `POST /api/transistors/{transistor_id}/rg-energy`.
The following example uses the stored device `Fuji_2MBI600XEE065-50`.

```json
{
  "operation": "energy-at-rg",
  "branch": "e_on",
  "representation": "graph_i_e",
  "i_channel": 600,
  "t_j": 25,
  "r_g": 3.3,
  "conditions": {
    "v_supply": 300, "v_g": 15, "v_g_off": null,
    "load_inductance": null, "commutation_inductance": null,
    "commutation_device": null, "convention": null, "event": null, "method": null
  }
}
```

The example returned approximately **0.0154173 J** from the November 10 device
inventory. For an inverse query, set `operation` to `rg-at-energy` and replace
`r_g` with `e_target` in joules. The inverse calculation requires one bounded
crossing. Ambiguous or unsupported queries return a diagnostic.

The server returns the numeric result, unit, query and considered source rows
for the selected branch and representation. Each row retains its actual
operating conditions. The row set includes records considered for the query.
The available source snapshots identify normalized calculation inputs.
