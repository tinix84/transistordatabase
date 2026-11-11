<script setup>
import { computed, ref, watch } from 'vue'
import { displayNumber as number } from './analysisFiles.js'

const props = defineProps({ projection: { type: Object, required: true } })
const timePage = ref(0), frequencyPage = ref(0)
const pageSize = 25
watch(() => props.projection, () => { timePage.value = 0; frequencyPage.value = 0 })
const timeRows = computed(() => props.projection.time_rows.slice(timePage.value * pageSize, (timePage.value + 1) * pageSize))
const frequencyRows = computed(() => props.projection.frequency_rows.slice(frequencyPage.value * pageSize, (frequencyPage.value + 1) * pageSize))
const timeFields = [
  ['sample_count', 'Samples'], ['minimum', 'Minimum'], ['maximum', 'Maximum'], ['arithmetic_mean', 'Arithmetic sample mean'],
  ['sample_standard_deviation', 'Sample SD (ddof=1)'], ['discrete_rms', 'Discrete sample RMS'],
  ['time_weighted_mean', 'Piecewise-linear time mean'], ['time_weighted_rms', 'Piecewise-linear time RMS'],
]
const frequencyFields = [
  ['retained_bin_count', 'Retained bins'], ['bin_index', 'Bin'], ['frequency_hz', 'Frequency (Hz)'], ['magnitude', 'Magnitude'],
  ['real', 'Real'], ['imaginary', 'Imaginary'], ['phase_rad', 'Phase (rad)'],
]
function downloadRaw(value, filename, type = 'text/csv') {
  const url = URL.createObjectURL(new Blob([value], { type }))
  const link = document.createElement('a'); link.href = url; link.download = filename; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
</script>

<template>
  <section class="grouped-evidence" data-testid="grouped-series-evidence">
    <h4>Grouped physical series statistics and harmonics</h4>
    <p>{{ projection.scope }} · {{ projection.method_version }}. Every row keeps its configuration, operating point, source, unit, status, and input hash.</p>
    <p>Statistics describe supplied samples. Sample standard deviation uses ddof=1. Piecewise-linear time statistics are separate quantities. Values use each row's series unit.</p>
    <p>Harmonics keep each pair and physical series separate. Phase is in radians from the first supplied sample. Zero-magnitude phase is unavailable.</p>
    <p>Downloads contain all server-projected rows, including unsupported and unrequested spectra. Table pages and pair filters do not change downloads. CSV uses {{ projection.null_csv }} for null.</p>
    <div class="actions">
      <button @click="downloadRaw(projection.time_json, 'grouped-time-statistics.json', 'application/json')">Download grouped time JSON</button>
      <button @click="downloadRaw(projection.time_csv, 'grouped-time-statistics.csv')">Download grouped time CSV</button>
      <button @click="downloadRaw(projection.frequency_json, 'grouped-frequency-harmonics.json', 'application/json')">Download grouped frequency JSON</button>
      <button @click="downloadRaw(projection.frequency_csv, 'grouped-frequency-harmonics.csv')">Download grouped frequency CSV</button>
    </div>
    <section v-for="kind in ['time', 'frequency']" :key="kind" :data-testid="`grouped-${kind}-table`">
      <h5>{{ kind === 'time' ? 'Sample and time statistics' : 'Selected harmonic bins' }}</h5>
      <p>{{ kind === 'time' ? projection.time_rows.length : projection.frequency_rows.length }} rows · page {{ (kind === 'time' ? timePage : frequencyPage) + 1 }} · at most {{ pageSize }} rows per page</p>
      <div class="actions" v-if="kind === 'time'"><button :disabled="timePage === 0" @click="timePage--">Previous time rows</button><button :disabled="(timePage + 1) * pageSize >= projection.time_rows.length" @click="timePage++">Next time rows</button></div>
      <div class="actions" v-else><button :disabled="frequencyPage === 0" @click="frequencyPage--">Previous frequency rows</button><button :disabled="(frequencyPage + 1) * pageSize >= projection.frequency_rows.length" @click="frequencyPage++">Next frequency rows</button></div>
      <div class="table-scroll"><table><thead><tr><th>Configuration / source</th><th>Operating point / source / weight</th><th>Series / unit</th><th>Status / error</th><th v-for="[key, label] in kind === 'time' ? timeFields : frequencyFields" :key="key">{{ label }}</th><th>FFT policy</th><th>Input SHA-256</th></tr></thead>
        <tbody><tr v-for="(row, index) in kind === 'time' ? timeRows : frequencyRows" :key="index"><td>{{ row.configuration_id }}<br>{{ row.configuration_source ?? 'Unknown' }}</td><td>{{ row.operating_point_id }}<br>{{ row.operating_point_source ?? 'Unknown' }}<br>Weight {{ number(row.weight) }}</td><td>{{ row.series ?? 'Unavailable' }}<br>{{ row.unit ?? 'Unavailable' }}</td><td>Series: {{ row.status }}<br><template v-if="row.pair_status">Pair: {{ row.pair_status }}<br></template><template v-if="row.retained_bin_count === 0">No bins exceed threshold.<br></template>{{ row.error ?? '' }}</td><td v-for="[key] in kind === 'time' ? timeFields : frequencyFields" :key="key">{{ number(row[key]) }}</td><td>{{ row.window ?? 'Unavailable' }} / {{ row.detrend ?? 'Unavailable' }}<br>Threshold {{ row.relative_threshold ?? 'Unavailable' }}</td><td class="hash">{{ row.input_sha256 }}<br>Grid {{ row.grid_input_sha256 }}<br>{{ row.method_version }}</td></tr></tbody>
      </table></div>
    </section>
  </section>
</template>

<style scoped>
.grouped-evidence { min-width:0; border:1px solid var(--border-color,#aaa); border-radius:8px; padding:1rem; margin:1rem 0; }
p { line-height:1.6; overflow-wrap:anywhere; }.actions { display:flex; flex-wrap:wrap; gap:.4rem; }button { padding:.55rem .8rem; border:1px solid #7c98ae; border-radius:5px; color:inherit; background:var(--bg-secondary); cursor:pointer; }button:disabled { opacity:.5; cursor:default; }
.table-scroll { max-width:100%; overflow-x:auto; }table { border-collapse:collapse; min-width:1200px; width:100%; }th,td { padding:.6rem; text-align:left; vertical-align:top; border-bottom:1px solid var(--border-color,#aaa); }td { min-width:100px; max-width:240px; overflow-wrap:anywhere; }.hash { min-width:180px; font-family:monospace; }
@media(max-width:600px) { .grouped-evidence { padding:.6rem; } }
</style>
