<script setup>
import { computed } from 'vue'
import CurvePlotter from './CurvePlotter.vue'
import { analysisCsv, downloadAnalysis } from './analysisFiles.js'
const props = defineProps({ spectrum: { type: Object, required: true } })
const mask = computed(() => props.spectrum.mask_screening)
const bands = computed(() => props.spectrum.band_rms)
const basis = computed(() => props.spectrum.analysis_basis)
const datasets = computed(() => mask.value ? [
  { label: `Observed peak (${mask.value.unit})`, showLine: false, pointRadius: 3, borderColor: '#0369a1', data: mask.value.bins.map(row => ({ x: row.frequency_hz, y: row.amplitude_peak })) },
  { label: 'Declared limit at covered bins', showLine: false, pointRadius: 3, borderColor: '#b45309', data: mask.value.bins.map(row => ({ x: row.frequency_hz, y: row.limit_peak })) },
] : [])
function number(value) { return value === null || value === undefined ? 'Unavailable' : Number(value).toPrecision(6) }
function maskCsv() {
  const context = { unit: mask.value.unit, source: mask.value.source, interpolation: mask.value.interpolation,
    margin_convention: mask.value.margin_convention, window: basis.value.window, detrend: basis.value.detrend,
    coherent_gain: basis.value.coherent_gain, quantity_basis: basis.value.mask_basis }
  downloadAnalysis(analysisCsv(mask.value.bins.map(row => ({ ...row, ...context }))), 'spectrum-mask-bins.csv', 'text/csv')
}
function bandsCsv() {
  const context = { window: basis.value.window, detrend: basis.value.detrend, coherent_gain: basis.value.coherent_gain,
    window_power_gain: basis.value.window_power_gain, quantity_basis: basis.value.band_basis }
  downloadAnalysis(analysisCsv(bands.value.map(row => ({ ...row, ...context }))), 'spectrum-band-rms.csv', 'text/csv')
}
</script>

<template>
  <section v-if="basis" class="spectral-results" data-testid="spectrum-analysis-results">
    <h4>All-bin spectrum analysis</h4>
    <p>{{ basis.observed_bin_count }} observed bins. Window: {{ basis.window }}. Detrending: {{ basis.detrend }}. Coherent gain {{ number(basis.coherent_gain) }}. Window power gain {{ number(basis.window_power_gain) }}.</p>
    <section v-if="mask" data-testid="spectrum-mask-results">
      <h5>Declared mask: {{ mask.status }}</h5>
      <p>{{ mask.covered_bin_count }} covered of {{ mask.observed_bin_count }} bins. {{ mask.violating_bin_indices.length }} violations. {{ mask.uncovered_bin_indices.length }} uncovered. Uncovered bins are unknown.</p>
      <p>Source: {{ mask.source }}. {{ mask.comparison }}. {{ mask.margin_convention }}. {{ mask.scope }}</p>
      <p data-testid="spectrum-mask-failing-bins">Violating bin indices: {{ mask.violating_bin_indices.length ? mask.violating_bin_indices.join(', ') : 'None among covered bins' }}.</p>
      <CurvePlotter title="All observed peak bins and covered mask limits" :datasets="datasets" x-label="Frequency (Hz)" :y-label="`Peak amplitude (${mask.unit})`" :y-min="0" :allow-zoom="false" :compact-scientific-y-ticks="true"/>
      <p>The table previews the first 100 bins. Downloads retain all bins, including zeros, DC, Nyquist, and unknown coverage.</p>
      <div class="table-scroll"><table><thead><tr><th>Bin</th><th>Frequency (Hz)</th><th>Observed ({{ mask.unit }})</th><th>Limit</th><th>Margin</th><th>Coverage / result</th></tr></thead><tbody><tr v-for="row in mask.bins.slice(0, 100)" :key="row.bin_index"><td>{{ row.bin_index }}</td><td>{{ number(row.frequency_hz) }}</td><td>{{ number(row.amplitude_peak) }}</td><td>{{ number(row.limit_peak) }}</td><td>{{ number(row.margin_peak) }}</td><td>{{ row.coverage }} / {{ row.status }}</td></tr></tbody></table></div>
      <button type="button" @click="maskCsv" data-testid="spectrum-mask-download">Download every mask bin CSV</button>
    </section>
    <section v-if="bands" data-testid="spectrum-band-results">
      <h5>Raw windowed discrete band RMS</h5>
      <p>{{ basis.band_basis }}. {{ basis.band_overlap_policy }}</p>
      <div class="table-scroll"><table><thead><tr><th>Band</th><th>Range (Hz)</th><th>Endpoints</th><th>Bins</th><th>RMS</th><th>Result</th></tr></thead><tbody><tr v-for="row in bands" :key="row.name"><td>{{ row.name }}</td><td>{{ number(row.lower_hz) }} to {{ number(row.upper_hz) }}</td><td>{{ row.endpoints }}</td><td>{{ row.bin_count }}</td><td>{{ number(row.rms) }} {{ row.unit }}</td><td>{{ row.status }}</td></tr></tbody></table></div>
      <button type="button" @click="bandsCsv" data-testid="spectrum-bands-download">Download band RMS CSV</button>
    </section>
    <button type="button" @click="downloadAnalysis(JSON.stringify(spectrum), 'spectrum-analysis.json', 'application/octet-stream')" data-testid="spectrum-analysis-download">Download exact spectrum analysis JSON</button>
  </section>
</template>

<style scoped>
.spectral-results { min-width: 0; margin: 1rem 0; padding: .9rem; border: 1px solid #94a3b8; border-radius: .5rem; }
p { line-height: 1.5; overflow-wrap: anywhere; } .table-scroll { overflow-x: auto; max-width: 100%; }
table { border-collapse: collapse; min-width: 600px; width: 100%; } th, td { padding: .5rem; text-align: left; border-bottom: 1px solid #cbd5e1; }
button { padding: .6rem; margin: .5rem .5rem .5rem 0; font: inherit; cursor: pointer; }
</style>
