<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { analysisApi } from '../services/api.js'
import CurvePlotter from './CurvePlotter.vue'
import EnergyConditionsEditor from './EnergyConditionsEditor.vue'
import XyzEnergyImportPanel from './XyzEnergyImportPanel.vue'
import EnergyScene3D from './EnergyScene3D.vue'
import EnergySourceSliceEvidence from './EnergySourceSliceEvidence.vue'
import { POINT_LIMIT } from './energyScene.js'
import { finite, parsePairs } from './waveformModels.js'
import { analysisCsv, analysisError, displayNumber as number, downloadAnalysis } from './analysisFiles.js'
function conditions() { return { t_j_c: 25, v_g_on_v: null, v_g_off_v: null, r_g_on_ohm: null, r_g_off_ohm: null, commutation_device: null, commutation_source: null, energy_convention: 'Independent hand fixture E_on; no device measurement identity' } }
function slice(id = 'slice-1', voltage = 400, energy = .001) { return { id, source: 'Editable independent hand fixture; not measured device data', source_kind: 'manual', energy_kind: 'on', voltage_v: voltage, pairs: `0,0\n10,${energy}`, conditions: conditions() } }
const input = reactive({ kind: 'on', conditions: conditions(), slices: [slice(), slice('slice-2', 800, .002)], currents: '0,5,10', voltages: '400,800', estimate: false, current: 5, voltage: 600, advanced: false, json: '' })
const result = ref(null), acceptedRequest = ref(null), error = ref(''), busy = ref(false), stale = ref(false), selectedCell = ref(null)
const inputMode = ref('prepared')
const fullEvidence = ref(false), currentPage = ref(0), voltagePage = ref(0)
watch(result, () => { fullEvidence.value = false; currentPage.value = 0; voltagePage.value = 0 })
watch(inputMode, () => { revision++; result.value = null; acceptedRequest.value = null; selectedCell.value = null; busy.value = false; error.value = ''; stale.value = false })
let revision = 0, disposed = false
watch(input, () => { revision++; stale.value = !!result.value || busy.value; result.value = null; acceptedRequest.value = null; selectedCell.value = null; error.value = ''; busy.value = false }, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
function vector(text, label) { return text.trim().split(/[,;\s]+/).map(value => finite(value, label)) }
function request() {
  if (input.advanced) return JSON.parse(input.json)
  return { energy_kind: input.kind, conditions: input.conditions, currents_a: vector(input.currents, 'Current axis'), voltages_v: vector(input.voltages, 'Voltage axis'),
    slices: input.slices.map(value => { const pairs = parsePairs(value.pairs); return { id: value.id, source: value.source, source_kind: value.source_kind, energy_kind: value.energy_kind,
      voltage_v: finite(value.voltage_v, 'Slice voltage'), current_a: pairs.map(pair => pair[0]), energy_j: pairs.map(pair => pair[1]), conditions: value.conditions } }),
    ...(input.estimate ? { estimate: { selected: true, current_a: finite(input.current, 'Estimate current'), voltage_v: finite(input.voltage, 'Estimate voltage') } } : {}) }
}
async function calculate() {
  const token = ++revision; error.value = ''; result.value = null; busy.value = true; stale.value = false; selectedCell.value = null
  try {
    const payload = JSON.parse(JSON.stringify(request())), response = await analysisApi.energyMap(payload)
    if (disposed || token !== revision) return
    acceptedRequest.value = payload; result.value = response
  } catch (failure) { if (!disposed && token === revision) error.value = analysisError(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
function prepareJson() { try { input.json = JSON.stringify(request(), null, 2) } catch (failure) { error.value = failure.message } }
const maximum = computed(() => Math.max(0, ...(result.value?.cells.flat().filter(cell => cell.energy_j !== null).map(cell => cell.energy_j) ?? [])))
function cellColor(cell) { return cell.energy_j === null ? {} : { backgroundColor: `rgba(13, 148, 180, ${maximum.value ? .12 + .65 * cell.energy_j / maximum.value : .12})` } }
const sourcePointCount = computed(() => (result.value?.source_slices ?? []).reduce((count, value) => count + value.current_a.length, 0))
const curves = computed(() => sourcePointCount.value > POINT_LIMIT ? [] : (result.value?.source_slices ?? []).map(value => ({ label: `${value.id} · ${value.voltage_v} V · ${value.source_kind}`,
  data: value.current_a.map((current, index) => ({ x: current, y: value.energy_j[index] })), pointRadius: 4 })))
const matrixCurrents = computed(() => result.value?.currents_a.slice(currentPage.value * 16, (currentPage.value + 1) * 16) ?? [])
const matrixRows = computed(() => result.value?.cells.slice(voltagePage.value * 16, (voltagePage.value + 1) * 16) ?? [])
function exportCells() { downloadAnalysis(analysisCsv(result.value.cells.flat().map(cell => ({ current_a: cell.current_a, voltage_v: cell.voltage_v,
  energy_j: cell.energy_j, status: cell.status, mode: cell.mode, candidates: cell.candidates, unsupported_slices: cell.unsupported_slices }))), 'energy-map-cells.csv', 'text/csv') }
</script>

<template>
  <section class="energy-workbench"><header><p>Source-conditioned switching energy</p><h2>Energy map</h2><p>Inspect original voltage slices with exact fixed conditions. Unknown conditions match unknown values only. They are not wildcards.</p></header>
    <label>Energy map input<select v-model="inputMode" aria-label="Energy map input"><option value="prepared">Prepared energy map and source slices</option><option value="xyz">Import XYZ file or pasted triples</option></select></label>
    <XyzEnergyImportPanel v-if="inputMode === 'xyz'"/>
    <div v-else><form @submit.prevent="calculate"><section class="panel"><h3>Selected conditions and axes</h3><label>Energy kind<select v-model="input.kind" aria-label="Map energy kind"><option value="on">Turn-on E_on</option><option value="off">Turn-off E_off</option><option value="rr">Reverse recovery E_rr</option></select></label><EnergyConditionsEditor v-model="input.conditions" prefix="Map"/><p>Blank condition fields mean unknown. Source slices retain their own conditions when these filters change.</p><div class="fields"><label>Displayed currents (A), comma separated<input v-model="input.currents" aria-label="Map currents"></label><label>Displayed voltages (V), comma separated<input v-model="input.voltages" aria-label="Map voltages"></label></div></section>
      <section class="panel"><h3>Original source slices</h3><p>Each slice retains its source, energy kind, conditions and original points. The defaults are editable manual fixtures.</p>
        <details v-for="(value, index) in input.slices" :key="index" :open="index === 0" class="slice"><summary>{{ value.id }} · {{ value.voltage_v }} V · {{ value.source_kind }}</summary><div class="fields"><label>Slice ID<input v-model="value.id" required :aria-label="`Slice ${index + 1} ID`"></label><label>Source identifier<input v-model="value.source" required></label><label>Source kind<select v-model="value.source_kind"><option value="manual">Manual</option><option value="measured">Measured</option><option value="datasheet">Datasheet</option><option value="simulated">Simulated</option><option value="unknown">Unknown</option></select></label><label>Slice energy kind<select v-model="value.energy_kind"><option value="on">Turn-on</option><option value="off">Turn-off</option><option value="rr">Reverse recovery</option></select></label><label>Slice voltage (V)<input v-model.number="value.voltage_v" type="number" step="any"></label><label>Original magnitude current (A), energy (J)<textarea v-model="value.pairs" rows="4" :aria-label="`Slice ${index + 1} points`"></textarea></label></div><details><summary>Original slice conditions</summary><EnergyConditionsEditor v-model="value.conditions" :prefix="`Slice ${index + 1}`"/></details></details>
        <button type="button" @click="input.slices.push(slice(`slice-${input.slices.length + 1}`))">Add source slice</button>
      </section>
      <section class="panel"><h3>Optional bounded estimate</h3><label class="check"><input v-model="input.estimate" type="checkbox">I select bounded bilinear I/V interpolation as an estimate</label><div v-if="input.estimate" class="fields"><label>Estimate current (A)<input v-model.number="input.current" type="number" min="0" step="any" aria-label="Estimate current"></label><label>Estimate voltage (V)<input v-model.number="input.voltage" type="number" min="0" step="any" aria-label="Estimate voltage"></label></div><p>Python rejects an estimate outside the displayed rectangle or with a missing or ambiguous corner. No extrapolation or automatic voltage scaling applies.</p><details><summary>Advanced canonical map request</summary><button type="button" @click="prepareJson">Copy form into map request editor</button><label class="check"><input v-model="input.advanced" type="checkbox">Use map JSON as input</label><label>Map request JSON<textarea v-model="input.json" rows="10"></textarea></label></details></section>
      <button type="submit" :disabled="busy">{{ busy ? 'Building map…' : 'Build energy map' }}</button>
    </form><p v-if="error" role="alert">{{ error }}</p><p v-if="stale" role="status">Inputs changed. Build the map again to restore results and exports.</p>
    <section v-if="result" data-testid="energy-map-results"><h3>{{ result.energy_kind }} energy map (J)</h3><p class="hash">Input SHA-256: {{ result.input_sha256 }} · {{ result.method_version }}</p><div class="actions"><button @click="downloadAnalysis({ request: acceptedRequest, response: result }, 'energy-map.json')">Download map request + result JSON</button><button @click="exportCells">Download map cell CSV</button></div>
      <EnergyScene3D :result="result" mode="prepared"/>
      <h4>Conditioned energy cells</h4><p>Color represents available energy only. Missing and ambiguous cells retain their status.</p>
      <p v-if="result.currents_a.length > 16 || result.voltages_v.length > 16">The matrix shows 16 current coordinates and 16 voltage coordinates per page. All cells remain in the downloads.</p>
      <div class="actions"><template v-if="result.currents_a.length > 16"><button :disabled="currentPage === 0" @click="currentPage--">Previous current columns</button><span>Current page {{ currentPage + 1 }} / {{ Math.ceil(result.currents_a.length / 16) }}</span><button :disabled="(currentPage + 1) * 16 >= result.currents_a.length" @click="currentPage++">Next current columns</button></template><template v-if="result.voltages_v.length > 16"><button :disabled="voltagePage === 0" @click="voltagePage--">Previous voltage rows</button><span>Voltage page {{ voltagePage + 1 }} / {{ Math.ceil(result.voltages_v.length / 16) }}</span><button :disabled="(voltagePage + 1) * 16 >= result.voltages_v.length" @click="voltagePage++">Next voltage rows</button></template></div>
      <div class="table-scroll"><table class="heatmap"><thead><tr><th>Voltage (V) / current (A)</th><th v-for="current in matrixCurrents" :key="current">{{ number(current) }} A</th></tr></thead><tbody><tr v-for="(row, index) in matrixRows" :key="index"><th>{{ number(result.voltages_v[voltagePage * 16 + index]) }} V</th><td v-for="cell in row.slice(currentPage * 16, (currentPage + 1) * 16)" :key="cell.current_a" :style="cellColor(cell)"><button @click="selectedCell = cell">{{ number(cell.energy_j) }} J<br>{{ cell.status }}<br>{{ cell.mode ?? '' }}</button></td></tr></tbody></table></div>
      <section v-if="result.estimate" class="estimate" data-testid="energy-estimate"><h4>Selected estimate: {{ result.estimate.query.current_a }} A, {{ result.estimate.query.voltage_v }} V</h4><p>{{ result.estimate.method }} · {{ result.estimate.status }} · <strong>{{ number(result.estimate.energy_j) }} J</strong></p><p v-if="result.estimate.error" role="alert">{{ result.estimate.error }}</p><details><summary>Exact interpolation corners</summary><pre>{{ JSON.stringify(result.estimate.corners, null, 2) }}</pre></details></section>
      <section v-if="selectedCell" class="panel"><h4>Cell evidence: {{ selectedCell.current_a }} A, {{ selectedCell.voltage_v }} V</h4><p>{{ selectedCell.status }} · {{ number(selectedCell.energy_j) }} J</p><ul><li v-for="candidate in selectedCell.candidates" :key="candidate.slice_id">{{ candidate.slice_id }} · {{ candidate.source }} · {{ candidate.source_kind }} · {{ candidate.mode }} · {{ number(candidate.energy_j) }} J</li><li v-for="item in selectedCell.unsupported_slices" :key="item.slice_id">{{ item.slice_id }}: {{ item.reason }}</li></ul></section>
      <h4>Source inclusion audit</h4><ul><li v-for="item in result.slice_diagnostics" :key="item.slice_id">{{ item.slice_id }} · {{ item.status }} · {{ item.reason ?? 'Exact selected conditions' }}</li></ul>
      <CurvePlotter v-if="curves.length" title="Original source slices (all sources, including excluded slices)" :datasets="curves" x-label="Magnitude current (A)" y-label="Original energy (J)" :allow-zoom="false"/>
      <p v-if="sourcePointCount > POINT_LIMIT">The original-slice chart exceeds the {{ POINT_LIMIT }} point display limit. Use the paged source tables and downloads.</p>
      <EnergySourceSliceEvidence v-for="source in result.source_slices" :key="source.id" :source="source" class="slice"/>
      <p v-for="assumption in result.assumptions" :key="assumption">{{ assumption }}</p><details @toggle="fullEvidence = $event.target.open"><summary>Full accepted map evidence</summary><pre v-if="fullEvidence">{{ JSON.stringify(result, null, 2) }}</pre></details>
    </section></div>
  </section>
</template>

<style scoped>
.energy-workbench { max-width:1200px; margin:auto; padding:1.2rem; min-width:0; color:var(--text-primary); }header,h3,h4 { margin:.7rem 0; }p { line-height:1.6; margin:.6rem 0; }.panel,.slice,.estimate { border:1px solid var(--border-color,#aaa); border-radius:8px; padding:1rem; margin:1rem 0; min-width:0; background:var(--bg-secondary); }.fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:1rem; }label { display:grid; gap:.35rem; margin:.6rem 0; }.check { display:flex; align-items:baseline; gap:.5rem; }input:not([type=checkbox]),select,textarea { width:100%; min-width:0; padding:.5rem; border:1px solid var(--border-color,#aaa); color:inherit; background:var(--bg-secondary); font:inherit; }textarea { font-family:monospace; }button { padding:.6rem; margin:.25rem; cursor:pointer; color:inherit; background:var(--bg-secondary); border:1px solid #7b97ab; border-radius:5px; }button:disabled { opacity:.6; cursor:wait; }summary { cursor:pointer; font-weight:600; }.table-scroll { max-width:100%; overflow-x:auto; }table { width:100%; border-collapse:collapse; }th,td { text-align:left; padding:.6rem; border-bottom:1px solid var(--border-color,#aaa); vertical-align:top; overflow-wrap:anywhere; }.heatmap { min-width:540px; }.heatmap button { background:transparent; font:inherit; }pre { white-space:pre-wrap; overflow-wrap:anywhere; }.hash { overflow-wrap:anywhere; }ul { padding-left:1.3rem; }[role=alert] { color:#b02828; overflow-wrap:anywhere; }
@media(max-width:600px) { .energy-workbench { padding:.5rem; }.panel,.slice,.estimate { padding:.6rem; } }
</style>
