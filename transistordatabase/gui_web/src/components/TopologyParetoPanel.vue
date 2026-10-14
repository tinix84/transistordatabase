<template>
  <details class="pareto-panel" open>
    <summary>Fixed-boundary Boost semiconductor Pareto screen</summary>
    <p class="intro">
      Every candidate is evaluated at the same Boost CCM boundary. Loss uses the
      loss-balanced channel-conduction model. Cost and dimensions are request
      metadata and are never written into TDB records.
    </p>
    <div class="limits" role="note">
      <strong>Screening limits:</strong> switching, reverse-recovery, gate-drive,
      passive and magnetic losses are excluded. The rating screen checks only
      absolute blocking voltage and peak current, not SOA or thermal safety.
      Temperature remains unavailable when the TDB lacks usable thermal
      resistances. Package volume is a length × width × height body estimate.
    </div>

    <section class="pareto-section">
      <h4>Shared electrical boundary</h4>
      <div class="pareto-grid boundary-grid">
        <label>Input voltage (V)<input v-model.number="boundary.v_in" type="number" min="0.001" step="any"></label>
        <label>Output voltage (V)<input v-model.number="boundary.v_out" type="number" min="0.001" step="any"></label>
        <label>Output power (W)<input v-model.number="boundary.p_out" type="number" min="0.001" step="any"></label>
        <label>Switching frequency (kHz)<input v-model.number="boundary.f_sw_khz" type="number" min="0.001" step="any"></label>
        <label>Inductance (µH)<input v-model.number="boundary.l_value_uh" type="number" min="0.001" step="any"></label>
        <label>Output capacitance (µF)<input v-model.number="boundary.c_out_uf" type="number" min="0.001" step="any"></label>
        <label>Junction temperature (°C)<input v-model.number="boundary.t_j" type="number" step="any"></label>
        <label>Fixed case/sink temperature (°C)<input v-model.number="boundary.t_case" type="number" step="any"></label>
      </div>
    </section>

    <section class="pareto-section">
      <div class="section-heading">
        <h4>Candidate configurations</h4>
        <button type="button" class="btn btn-secondary" :disabled="candidates.length >= maxCandidates" @click="addCandidate">
          Add candidate
        </button>
      </div>
      <div class="candidate-list">
        <article v-for="(candidate, index) in candidates" :key="candidate.candidate_id" class="candidate-card">
          <div class="section-heading">
            <label>Candidate label<input v-model.trim="candidate.label" type="text" maxlength="120"></label>
            <button type="button" class="btn btn-secondary remove-button" :disabled="candidates.length <= 2" :aria-label="`Remove ${candidate.label || `candidate ${index + 1}`}`" @click="removeCandidate(index)">
              Remove
            </button>
          </div>
          <div class="pareto-grid candidate-grid">
            <label>Switch TDB device
              <select v-model="candidate.switch_transistor_id">
                <option value="">Select a device</option>
                <option v-for="device in transistors" :key="deviceId(device)" :value="deviceId(device)">{{ deviceName(device) }}</option>
              </select>
            </label>
            <label>Rectifier TDB device
              <select v-model="candidate.diode_transistor_id">
                <option value="">Select a device</option>
                <option v-for="device in transistors" :key="deviceId(device)" :value="deviceId(device)">{{ deviceName(device) }}</option>
              </select>
            </label>
            <label>Parallel switches<input v-model.number="candidate.parallel_switches" type="number" min="1" max="100" step="1"></label>
            <label>Parallel rectifiers<input v-model.number="candidate.parallel_diodes" type="number" min="1" max="100" step="1"></label>
            <label>Switch gate voltage (V)<input v-model.number="candidate.v_g" type="number" step="any"></label>
            <label>Rectifier curve gate voltage (V, optional)<input v-model="candidate.diode_v_g" type="number" step="any"></label>
          </div>
        </article>
      </div>
    </section>

    <section class="pareto-section">
      <h4>External package and price catalog</h4>
      <p class="hint">Enter one-unit USD list prices and source dates. Dimensions are maximum package-body values in mm; each source is preserved with the result.</p>
      <article v-for="partId in partIds" :key="partId" class="catalog-card">
        <h5>{{ partId }}</h5>
        <div class="pareto-grid catalog-grid">
          <label>Orderable part<input v-model.trim="catalog[partId].orderable_part" type="text"></label>
          <label>One-unit price (USD)<input v-model.number="catalog[partId].unit_cost_usd" type="number" min="0.001" step="any"></label>
          <label>Price accessed on<input v-model="catalog[partId].cost_accessed_on" type="date"></label>
          <label>Price source URL<input v-model.trim="catalog[partId].price_source_url" type="url"></label>
          <label>Length (mm)<input v-model.number="catalog[partId].length_mm" type="number" min="0.001" step="any"></label>
          <label>Width (mm)<input v-model.number="catalog[partId].width_mm" type="number" min="0.001" step="any"></label>
          <label>Height (mm)<input v-model.number="catalog[partId].height_mm" type="number" min="0.001" step="any"></label>
          <label>Dimensions source URL<input v-model.trim="catalog[partId].dimensions_source_url" type="url"></label>
        </div>
      </article>
    </section>

    <div class="pareto-actions">
      <button type="button" class="btn btn-primary" :disabled="loading || candidates.length < 2" @click="calculate">
        {{ loading ? 'Screening…' : 'Calculate Pareto screen' }}
      </button>
      <button type="button" class="btn btn-info" :disabled="!screenResult" @click="downloadResults">Download JSON</button>
    </div>
    <p v-if="errorMessage" class="error-message" role="alert">{{ errorMessage }}</p>

    <section v-if="screenResult" class="pareto-section results" aria-live="polite">
      <h4>Pareto results</h4>
      <p class="hint">
        {{ screenResult.eligible_candidate_count }} of {{ screenResult.candidates.length }} candidates are eligible.
        All listed objectives are minimized. {{ screenResult.dominance_rule }}
      </p>
      <p class="hint">Cost basis: {{ screenResult.cost_basis }}. Pareto membership is calculated in Python from channel-conduction loss, package-body volume, and component cost.</p>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Candidate</th><th>Status</th><th>Channel loss (W)</th><th>Volt-second residual (V)</th><th>Body volume (mm³)</th><th>Cost (USD)</th><th>Ratings</th></tr></thead>
          <tbody>
            <tr v-for="candidate in screenResult.candidates" :key="candidate.candidate_id" :class="{ frontier: candidate.pareto, excluded: !candidate.eligible }">
              <th scope="row">{{ candidate.label }}</th>
              <td>{{ candidate.pareto ? 'Pareto' : candidate.eligible ? 'Dominated' : 'Ineligible' }}</td>
              <td>{{ format(candidate.objectives?.channel_conduction_loss_w, 3) }}</td>
              <td>{{ format(candidate.operating_point?.calculated?.inductor_volt_second_residual_v, 6) }}</td>
              <td>{{ format(candidate.objectives?.package_body_volume_mm3, 1) }}</td>
              <td>{{ format(candidate.objectives?.component_cost_usd, 2) }}</td>
              <td>{{ candidate.ratings?.pass == null ? 'Unavailable' : candidate.ratings.pass ? 'Pass' : 'Fail' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-for="candidate in screenResult.candidates" :key="`${candidate.candidate_id}-detail`" class="candidate-result">
        <h5>{{ candidate.label }} — {{ candidate.pareto ? 'Pareto' : candidate.eligible ? 'Dominated' : 'Ineligible' }}</h5>
        <ul v-if="candidate.ineligible_reasons.length" class="error-list">
          <li v-for="reason in candidate.ineligible_reasons" :key="reason">{{ reason }}</li>
        </ul>
        <p v-if="candidate.operating_point?.losses">{{ candidate.operating_point.losses.scope }}.</p>
        <div v-if="candidate.operating_point?.model?.assumptions?.length" class="candidate-assumptions">
          <strong>Operating-point assumptions</strong>
          <ul>
            <li v-for="assumption in candidate.operating_point.model.assumptions" :key="assumption">{{ assumption }}</li>
          </ul>
        </div>
        <p v-if="candidate.operating_point?.temperatures">
          Switch temperature: {{ temperature(candidate.operating_point.temperatures.switch_c, candidate.operating_point.temperatures.switch_unavailable_reason) }};
          rectifier temperature: {{ temperature(candidate.operating_point.temperatures.diode_c, candidate.operating_point.temperatures.diode_unavailable_reason) }}.
        </p>
        <ul v-if="candidate.catalog.switch || candidate.catalog.diode" class="source-list">
          <li v-for="(part, role) in candidate.catalog" :key="role">
            {{ role }}: {{ part.orderable_part }}, ${{ format(part.unit_cost_usd, 2) }} one-unit price accessed {{ part.cost_accessed_on }}.
            <a :href="part.price_source_url" target="_blank" rel="noreferrer">Price source</a> ·
            <a :href="part.dimensions_source_url" target="_blank" rel="noreferrer">Dimensions source</a>
          </li>
        </ul>
      </div>
      <div v-if="eligibleCandidates.length" class="plots">
        <div v-for="plot in plots" :key="plot.x + plot.y" class="plot-card">
          <h5>{{ plot.title }}</h5>
          <svg class="pareto-plot" viewBox="0 0 400 240" role="img" :aria-label="plot.title">
            <line x1="42" y1="15" x2="42" y2="205" />
            <line x1="42" y1="205" x2="385" y2="205" />
            <text x="210" y="234" text-anchor="middle">{{ plot.xLabel }}</text>
            <text x="14" y="105" text-anchor="middle" transform="rotate(-90 14 105)">{{ plot.yLabel }}</text>
            <g v-for="point in project(plot)" :key="point.candidate.candidate_id">
              <circle :cx="point.x" :cy="point.y" r="6" :class="{ 'frontier-point': point.candidate.pareto }">
                <title>{{ point.candidate.label }}: {{ plot.xLabel }} {{ format(point.xValue, 3) }}, {{ plot.yLabel }} {{ format(point.yValue, 3) }}; {{ point.candidate.pareto ? 'Pareto' : 'dominated' }}</title>
              </circle>
            </g>
          </svg>
        </div>
      </div>
      <ul class="limitations"><li v-for="limit in screenResult.model_limitations" :key="limit">{{ limit }}</li></ul>
    </section>
  </details>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue'
import { transistorApi } from '../services/api.js'

const props = defineProps({ transistors: { type: Array, default: () => [] } })
const maxCandidates = 20
const transistors = computed(() => props.transistors || [])
const boundary = reactive({
  v_in: 100, v_out: 200, p_out: 1000, f_sw_khz: 20,
  l_value_uh: 1000, c_out_uf: 1000, t_j: 25, t_case: 60
})
const candidates = ref([
  {
    candidate_id: 'cree-one', label: 'CREE ×1 + CREE rectifier ×1',
    switch_transistor_id: 'CREE_C3M0060065J', diode_transistor_id: 'CREE_C3M0060065J',
    parallel_switches: 1, parallel_diodes: 1, v_g: 15, diode_v_g: 0
  },
  {
    candidate_id: 'infineon-one', label: 'Infineon ×1 + CREE rectifier ×1',
    switch_transistor_id: 'Infineon_IPBE65R050CFD7A', diode_transistor_id: 'CREE_C3M0060065J',
    parallel_switches: 1, parallel_diodes: 1, v_g: 10, diode_v_g: 0
  },
  {
    candidate_id: 'infineon-two', label: 'Infineon ×2 + CREE rectifier ×1',
    switch_transistor_id: 'Infineon_IPBE65R050CFD7A', diode_transistor_id: 'CREE_C3M0060065J',
    parallel_switches: 2, parallel_diodes: 1, v_g: 10, diode_v_g: 0
  }
])
const catalog = reactive({
  CREE_C3M0060065J: {
    orderable_part: 'C3M0060065J', unit_cost_usd: 9.64,
    length_mm: 9.125, width_mm: 10.23, height_mm: 4.57,
    price_source_url: 'https://www.digikey.com/en/products/detail/wolfspeed-inc/C3M0060065J/12525328',
    cost_accessed_on: '2026-10-14',
    dimensions_source_url: 'https://assets.wolfspeed.com/uploads/2024/01/Wolfspeed_C3M0060065J_data_sheet.pdf'
  },
  Infineon_IPBE65R050CFD7A: {
    orderable_part: 'IPBE65R050CFD7AATMA1', unit_cost_usd: 10.78,
    length_mm: 9.45, width_mm: 10.2, height_mm: 4.5,
    price_source_url: 'https://www.digikey.com/en/products/detail/infineon-technologies/IPBE65R050CFD7AATMA1/13212427',
    cost_accessed_on: '2026-10-14',
    dimensions_source_url: 'https://www.infineon.com/assets/row/public/documents/24/49/infineon-ipbe65r050cfd7a-datasheet-en.pdf'
  }
})
const screenResult = ref(null)
const errorMessage = ref('')
const loading = ref(false)
let requestSequence = 0
let candidateSerial = 3

const partIds = computed(() => [...new Set(candidates.value.flatMap(candidate => [
  candidate.switch_transistor_id,
  candidate.diode_transistor_id
]).filter(Boolean))])
const eligibleCandidates = computed(() => screenResult.value?.candidates?.filter(candidate => candidate.eligible) || [])
const plots = [
  { x: 'channel_conduction_loss_w', y: 'package_body_volume_mm3', title: 'Channel loss vs package volume', xLabel: 'Channel loss (W)', yLabel: 'Body volume (mm³)' },
  { x: 'channel_conduction_loss_w', y: 'component_cost_usd', title: 'Channel loss vs one-unit cost', xLabel: 'Channel loss (W)', yLabel: 'Cost (USD)' },
  { x: 'package_body_volume_mm3', y: 'component_cost_usd', title: 'Package volume vs one-unit cost', xLabel: 'Body volume (mm³)', yLabel: 'Cost (USD)' }
]

function deviceId(device) {
  return device?.id || device?.metadata?.name || device?.name || ''
}
function deviceName(device) {
  return device?.metadata?.name || device?.name || deviceId(device)
}
function emptyCatalog() {
  return {
    orderable_part: '', unit_cost_usd: null, length_mm: null, width_mm: null,
    height_mm: null, price_source_url: '', cost_accessed_on: '',
    dimensions_source_url: ''
  }
}
function ensureCatalog(partId) {
  if (partId && !catalog[partId]) catalog[partId] = emptyCatalog()
}
function invalidateResults() {
  requestSequence += 1
  screenResult.value = null
  errorMessage.value = ''
  loading.value = false
}
watch(candidates, () => {
  partIds.value.forEach(ensureCatalog)
  invalidateResults()
}, { deep: true })
watch([boundary, catalog], invalidateResults, { deep: true })

function addCandidate() {
  if (candidates.value.length >= maxCandidates) return
  candidateSerial += 1
  const firstPart = partIds.value[0] || deviceId(transistors.value[0])
  candidates.value.push({
    candidate_id: `candidate-${candidateSerial}`,
    label: `Candidate ${candidateSerial}`,
    switch_transistor_id: firstPart,
    diode_transistor_id: firstPart,
    parallel_switches: 1,
    parallel_diodes: 1,
    v_g: 15,
    diode_v_g: 0
  })
  ensureCatalog(firstPart)
}
function removeCandidate(index) {
  if (candidates.value.length > 2) candidates.value.splice(index, 1)
}
function optionalNumber(value) {
  if (value == null || value === '') return undefined
  const number = Number(value)
  return Number.isFinite(number) ? number : undefined
}
function buildRequest() {
  const partCatalog = Object.fromEntries(partIds.value.map(partId => {
    const item = catalog[partId]
    const entry = {
      orderable_part: item.orderable_part || undefined,
      unit_cost_usd: optionalNumber(item.unit_cost_usd),
      length_mm: optionalNumber(item.length_mm),
      width_mm: optionalNumber(item.width_mm),
      height_mm: optionalNumber(item.height_mm),
      price_source_url: item.price_source_url || undefined,
      cost_accessed_on: item.cost_accessed_on || undefined,
      dimensions_source_url: item.dimensions_source_url || undefined
    }
    return [partId, Object.fromEntries(Object.entries(entry).filter(([, value]) => value !== undefined))]
  }))
  return {
    boundary: {
      v_in: Number(boundary.v_in), v_out: Number(boundary.v_out),
      p_out: Number(boundary.p_out), f_sw: Number(boundary.f_sw_khz) * 1000,
      l_value: Number(boundary.l_value_uh) * 1e-6,
      c_out: Number(boundary.c_out_uf) * 1e-6,
      t_j: Number(boundary.t_j), t_case: optionalNumber(boundary.t_case)
    },
    candidates: candidates.value.map(candidate => ({
      ...candidate,
      v_g: optionalNumber(candidate.v_g),
      diode_v_g: optionalNumber(candidate.diode_v_g)
    })),
    catalog: partCatalog
  }
}
async function calculate() {
  const sequence = ++requestSequence
  loading.value = true
  errorMessage.value = ''
  try {
    const request = buildRequest()
    const response = await transistorApi.calculateTopologyPareto(request)
    if (sequence !== requestSequence) return
    screenResult.value = { request, ...response }
  } catch (error) {
    if (sequence !== requestSequence) return
    screenResult.value = null
    errorMessage.value = error.response?.data?.detail || error.message || 'Pareto calculation failed'
  } finally {
    if (sequence === requestSequence) loading.value = false
  }
}
function format(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : '—'
}
function temperature(value, reason) {
  return Number.isFinite(value) ? `${format(value, 1)} °C` : `unavailable${reason ? ` (${reason})` : ''}`
}
function project(plot) {
  const points = eligibleCandidates.value
  const xValues = points.map(candidate => candidate.objectives[plot.x])
  const yValues = points.map(candidate => candidate.objectives[plot.y])
  const xMin = Math.min(...xValues)
  const xMax = Math.max(...xValues)
  const yMin = Math.min(...yValues)
  const yMax = Math.max(...yValues)
  const xRange = xMax - xMin || 1
  const yRange = yMax - yMin || 1
  return points.map(candidate => {
    const xValue = candidate.objectives[plot.x]
    const yValue = candidate.objectives[plot.y]
    return {
      candidate,
      xValue,
      yValue,
      x: 48 + ((xValue - xMin) / xRange) * 326,
      y: 194 - ((yValue - yMin) / yRange) * 166
    }
  })
}
function downloadResults() {
  if (!screenResult.value) return
  const exportData = {
    workflow: 'fixed-boundary-boost-semiconductor-pareto',
    request: screenResult.value.request,
    result: screenResult.value,
    exported_at: new Date().toISOString()
  }
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `boost-semiconductor-pareto-${new Date().toISOString().slice(0, 10)}.json`
  link.click()
  URL.revokeObjectURL(url)
}

defineExpose({ buildRequest, calculate })
</script>

<style scoped>
.pareto-panel {
  margin: 2rem auto 0;
  max-width: 1600px;
  padding: 1.25rem;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, .1);
}
.pareto-panel > summary { cursor: pointer; font-size: 1.2rem; font-weight: 700; color: #243746; }
.intro, .hint { color: #5b6670; line-height: 1.45; }
.limits { margin: 1rem 0; padding: .8rem 1rem; background: #fff7e5; border-left: 4px solid #d58b00; line-height: 1.45; }
.pareto-section { margin-top: 1.5rem; }
.pareto-section h4 { margin: 0 0 .8rem; color: #2c3e50; }
.pareto-grid { display: grid; gap: .8rem; }
.boundary-grid { grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
.candidate-grid, .catalog-grid { grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
.pareto-grid label, .section-heading label { display: flex; flex-direction: column; gap: .3rem; color: #47535c; font-size: .9rem; }
.pareto-grid input, .pareto-grid select, .section-heading input { min-width: 0; padding: .45rem .55rem; border: 1px solid #c8d0d5; border-radius: 4px; font: inherit; }
.section-heading { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin-bottom: .75rem; }
.section-heading h4 { margin: 0; }
.candidate-list { display: grid; gap: .8rem; }
.candidate-card, .catalog-card, .plot-card { padding: 1rem; border: 1px solid #dbe0e3; border-radius: 6px; background: #fafbfc; }
.candidate-card .section-heading input { max-width: 340px; }
.catalog-card { margin-top: .75rem; }
.catalog-card h5 { margin: 0 0 .8rem; color: #334b5a; }
.btn { cursor: pointer; padding: .5rem .8rem; border: 0; border-radius: 4px; font: inherit; }
.btn:disabled { opacity: .55; cursor: not-allowed; }
.btn-primary { background: #126a91; color: #fff; }
.btn-secondary { background: #e8edf0; color: #243746; }
.btn-info { background: #167c80; color: #fff; }
.remove-button { flex: 0 0 auto; }
.pareto-actions { display: flex; gap: .75rem; margin-top: 1.5rem; }
.error-message, .error-list { color: #9c2525; }
.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-variant-numeric: tabular-nums; }
th, td { padding: .55rem .6rem; border-bottom: 1px solid #dbe0e3; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead { background: #edf2f4; }
tr.frontier { background: #eaf5ef; }
tr.excluded { color: #69757d; }
.candidate-result { padding: .8rem 0; border-top: 1px solid #e3e7e9; }
.candidate-result h5 { margin: 0 0 .4rem; }
.candidate-result p { margin: .4rem 0; color: #4c5860; }
.candidate-assumptions ul { margin: .35rem 0; padding-left: 1.2rem; color: #4c5860; }
.source-list { margin: .35rem 0; padding-left: 1.2rem; }
.source-list a { color: #145d80; }
.plots { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: .8rem; margin-top: 1.2rem; }
.plot-card h5 { margin: 0 0 .5rem; text-align: center; }
.pareto-plot { display: block; width: 100%; height: auto; overflow: visible; }
.pareto-plot line { stroke: #75838c; stroke-width: 1; }
.pareto-plot text { fill: #4e5d66; font-size: 11px; }
.pareto-plot circle { fill: #a6afb4; stroke: #35454f; stroke-width: 1; }
.pareto-plot circle.frontier-point { fill: #167d60; }
.limitations { margin-top: 1.2rem; padding-left: 1.2rem; color: #5b6670; }
@media (max-width: 700px) {
  .pareto-panel { padding: .8rem; }
  .section-heading { align-items: stretch; flex-direction: column; }
  th, td { padding: .45rem; font-size: .85rem; }
}
</style>
