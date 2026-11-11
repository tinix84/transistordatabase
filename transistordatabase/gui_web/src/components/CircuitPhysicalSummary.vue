<script setup>
import { computed, nextTick, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { summarizeCircuit } from '../services/circuitSummaryApi.js'
import { finiteDecimal } from './analysisFiles.js'
import { currentDirections, lossComponents, serializeMembers, stageMembers } from './circuitSummaryForm.js'

const props = defineProps({ converterInput: Object, bindings: Array, converterRequest: Function })
const emit = defineEmits(['load-fixture'])
const enabled = ref(false), result = ref(null), error = ref(''), busy = ref(false), stale = ref(false)
const selectedMember = ref(''), selectedEvidence = ref('')
const form = reactive({ configuration_id: '', scenario_id: '', temperature_c: 25, boundarySource: '', boundaryConditions: '',
  members: [], acknowledge_known_only: false, provenance: '', advanced: false, json: '' })
let revision = 0, disposed = false
function invalidate() { revision++; stale.value = result.value !== null || busy.value; result.value = null; error.value = '' }
watch(() => props.converterInput, invalidate, { deep: true, flush: 'sync' })
watch(form, invalidate, { deep: true, flush: 'sync' })
watch(enabled, invalidate, { flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
const activeMember = computed(() => form.members.find(member => member.id === selectedMember.value))
function stage() {
  try {
    const converter = props.converterRequest()
    form.members = stageMembers(converter.bindings ?? props.bindings)
    selectedMember.value = form.members[0]?.id ?? ''
  } catch (failure) { error.value = failure.message }
}
function addJunction() {
  const member = activeMember.value
  member.junctions.push({ id: `${member.id}:junction-${member.junctions.length + 1}`, rtheta_k_w: null, source: '', conditions: { description: null } })
}
async function fixture() {
  emit('load-fixture'); await nextTick(); stage()
  form.configuration_id = 'independent-two-device-hand-fixture'
  form.scenario_id = '1A-1V-1s-fixed-sink'
  form.temperature_c = 25; form.boundarySource = 'Independent algebraic hand fixture; not measured device data'
  form.boundaryConditions = 'Ideal fixed sink. Constant heat. No thermal feedback.'
  form.provenance = 'Independent acceptance fixture. No named-device accuracy claim.'
  form.acknowledge_known_only = false; form.advanced = false
  for (const member of form.members) {
    const id = `${member.id}:shared-junction`
    member.layout_complete = true
    member.junctions = [{ id, rtheta_k_w: 10, source: 'Declared hand-fixture resistance', conditions: { description: 'Constant 10 K/W junction to ideal sink' } }]
    for (const component of lossComponents) for (const direction of currentDirections) member.loss_owners[component][direction] = { die: 'channel', junction_id: id }
  }
}
function request(fromForm = false) {
  if (form.advanced && !fromForm) return form.json
  return { configuration_id: form.configuration_id, scenario_id: form.scenario_id, converter: props.converterRequest(),
    boundary: { kind: 'ideal-fixed-sink', temperature_c: finiteDecimal(form.temperature_c, 'Sink temperature'), source: form.boundarySource.trim() || null, conditions: { description: form.boundaryConditions.trim() || null } },
    members: serializeMembers(form.members), acknowledge_known_only: form.acknowledge_known_only,
    provenance: { source: form.provenance.trim() || null } }
}
function showRequest() { try { form.json = JSON.stringify(request(true), null, 2) } catch (failure) { error.value = failure.message } }
async function calculate() {
  result.value = null; error.value = ''; stale.value = false; const token = ++revision; busy.value = true
  try {
    const data = await summarizeCircuit(request())
    if (disposed || token !== revision) return
    result.value = data; selectedEvidence.value = Object.keys(data.evidence_by_role ?? {})[0] ?? ''
  } catch (failure) {
    if (disposed || token !== revision) return
    const detail = failure.response?.data?.detail
    error.value = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : failure.message
  } finally { if (!disposed) busy.value = false }
}
function number(value) { return value === null || value === undefined ? 'Unavailable' : Number(value).toPrecision(6) }
function download(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type })), link = document.createElement('a')
  link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 0)
}
</script>

<template>
  <section class="circuit-summary panel">
    <h3>Optional physical circuit summary</h3>
    <label class="check"><input v-model="enabled" type="checkbox" data-testid="circuit-enable"> Configure physical loss ownership and steady junction temperatures</label>
    <template v-if="enabled">
      <p class="notice">This action uses the supplied converter signals and signed models above. Python calculates electrical losses once. Declare each member's heat owners and junction network. No current split or diode ownership is inferred.</p>
      <p>Only an ideal fixed sink with constant average heat is supported here. This is a steady calculation without temperature feedback. Use the existing Thermal tools for transients and Foster models.</p>
      <form @submit.prevent="calculate">
        <div class="actions"><button type="button" @click="stage">Stage members from current converter mapping</button><button type="button" @click="fixture" data-testid="circuit-fixture">Load independent two-device circuit fixture</button></div>
        <p>The independent fixture declares 1 A, 1 V and 1 s, with no gate edges. It does not claim named-device accuracy. Staging starts with unknown layout, resistance and ownership.</p>
        <div class="columns"><label>Configuration ID<input v-model="form.configuration_id" data-testid="circuit-configuration"></label><label>Scenario ID<input v-model="form.scenario_id" data-testid="circuit-scenario"></label></div>
        <div class="columns"><label>Ideal fixed sink temperature (°C)<input :value="form.temperature_c" @input="form.temperature_c = $event.target.value" type="number" step="any" data-testid="circuit-sink"></label><label>Boundary source (blank means unknown)<input v-model="form.boundarySource" data-testid="circuit-boundary-source"></label></div>
        <label>Boundary conditions<input v-model="form.boundaryConditions"></label>
        <label>Physical member to edit<select v-model="selectedMember" data-testid="circuit-member"><option v-for="member in form.members" :key="member.id" :value="member.id">{{ member.id }} — {{ member.role }}</option></select></label>
        <fieldset v-if="activeMember" :key="activeMember.id"><legend>{{ activeMember.id }} — explicit physical layout</legend>
          <label class="check"><input v-model="activeMember.layout_complete" type="checkbox" data-testid="circuit-layout"> I declare this member's complete junction layout</label>
          <p>Channel and body diode share a junction only when you select the same junction ID below. Junction IDs must be unique across members.</p>
          <div v-for="(junction, index) in activeMember.junctions" :key="index" class="junction">
            <div class="columns"><label>Junction ID<input v-model="junction.id" :data-testid="`junction-id-${index}`"></label><label>Junction to sink resistance (K/W, blank means unknown)<input :value="junction.rtheta_k_w" @input="junction.rtheta_k_w = $event.target.value" type="number" step="any" min="0" :data-testid="`junction-rtheta-${index}`"></label></div>
            <p>Zero resistance declares an ideal junction-to-sink path. Give its source and conditions explicitly.</p>
            <label>Resistance source (blank means unknown)<input v-model="junction.source"></label><label>Resistance conditions<input v-model="junction.conditions.description"></label>
            <button type="button" @click="activeMember.junctions.splice(index, 1)">Remove junction declaration</button>
          </div>
          <button type="button" @click="addJunction">Add declared junction</button>
          <h4>Explicit component and current-direction ownership</h4>
          <div v-for="component in lossComponents" :key="component" class="owner-component"><strong>{{ component.replaceAll('_', ' ') }}</strong>
            <div v-for="direction in currentDirections" :key="direction" class="owner-row"><span>{{ direction }} current</span><label>Die<select v-model="activeMember.loss_owners[component][direction].die" :data-testid="`owner-${component}-${direction}-die`"><option :value="null">Unknown owner</option><option value="channel">Channel</option><option value="diode">Diode</option></select></label><label>Junction<select v-model="activeMember.loss_owners[component][direction].junction_id" :data-testid="`owner-${component}-${direction}-junction`"><option :value="null">Unknown junction</option><option v-for="junction in activeMember.junctions" :key="junction.id" :value="junction.id">{{ junction.id }}</option></select></label></div>
          </div>
        </fieldset>
        <label class="check"><input v-model="form.acknowledge_known_only" type="checkbox" data-testid="circuit-known-only"> I acknowledge a known-heat-only temperature estimate when losses are omitted. This is not the full junction temperature.</label>
        <label>Summary provenance<input v-model="form.provenance" data-testid="circuit-provenance"></label>
        <details><summary>Advanced circuit request import / inspection</summary><button type="button" @click="showRequest">Copy circuit form into JSON editor</button><label class="check"><input v-model="form.advanced" type="checkbox" data-testid="circuit-advanced"> Use circuit JSON as calculation input</label><label>Canonical circuit request<textarea v-model="form.json" rows="12" spellcheck="false" data-testid="circuit-json"></textarea></label></details>
        <button type="submit" class="primary" :disabled="busy" data-testid="circuit-calculate">{{ busy ? 'Calculating circuit…' : 'Calculate physical circuit summary' }}</button>
        <p v-if="error" role="alert" class="error">{{ error }}</p><p v-if="stale && !result" role="status">Circuit inputs changed. Calculate again to show current results.</p>
      </form>
      <section v-if="result" data-testid="circuit-results">
        <h4>Physical circuit summary</h4>
        <p>Configuration: {{ result.configuration_id }} · Scenario: {{ result.scenario_id }} · Input SHA: {{ result.input_sha256 }}</p>
        <p>Electrical full total: <strong>{{ number(result.electrical.total_power_w) }} W</strong> · {{ result.electrical.total_status }}. Known component subtotal: {{ number(result.electrical.known_component_power_w) }} W.</p>
        <p>Full maximum junction temperature: <strong>{{ number(result.maximum.tj_c) }} °C</strong>. All tied hottest junction IDs: {{ result.maximum.junction_ids.join(', ') || 'Unavailable' }} · {{ result.maximum.status }}.</p>
        <p v-if="result.maximum.known_heat_tj_c != null" class="notice">Acknowledged known-heat-only maximum: {{ number(result.maximum.known_heat_tj_c) }} °C. Known-heat hottest IDs: {{ result.maximum.known_heat_junction_ids.join(', ') }}. This excludes omitted heat and is not a full temperature.</p>
        <p>Layout coverage: {{ result.coverage.layout_complete_members }} / {{ result.coverage.member_count }} members. Unallocated known heat: {{ number(result.coverage.unallocated_known_heat_w) }} W.</p>
        <ul v-if="result.coverage.omissions.length"><li v-for="(omission, index) in result.coverage.omissions" :key="index">{{ omission }}</li></ul>
        <p>Boundary: {{ number(result.boundary.temperature_c) }} °C · {{ result.boundary.source ?? 'Unknown source' }} · {{ JSON.stringify(result.boundary.conditions) }}. Iteration: {{ result.iteration_status }}.</p>
        <div class="actions"><button @click="download(result.export_json, 'circuit-summary.json', 'application/json')">Download circuit audit JSON</button><button @click="download(result.export_csv, 'circuit-summary.csv', 'text/csv')">Download circuit audit CSV</button></div>
        <div class="table-scroll"><table><thead><tr><th>Physical member / role</th><th>Full electrical W</th><th>Known subtotal W</th><th>Channel full / known W</th><th>Diode full / known W</th><th>Unallocated known W</th></tr></thead><tbody><tr v-for="member in result.members" :key="member.id"><th>{{ member.id }} / {{ member.role }}</th><td>{{ number(member.electrical.total_power_w) }}</td><td>{{ number(member.electrical.known_component_power_w) }}</td><td>{{ number(member.die_losses.channel.power_w) }} / {{ number(member.die_losses.channel.known_power_w) }}</td><td>{{ number(member.die_losses.diode.power_w) }} / {{ number(member.die_losses.diode.known_power_w) }}</td><td>{{ number(member.die_losses.unallocated.known_power_w) }}</td></tr></tbody></table></div>
        <div class="table-scroll"><table><thead><tr><th>Member / junction</th><th>Resistance K/W</th><th>Full / known heat W</th><th>Full Tj °C</th><th>Acknowledged known-heat-only Tj °C</th><th>Status / omissions / source</th></tr></thead><tbody><template v-for="member in result.members" :key="member.id"><tr v-for="junction in member.junctions" :key="junction.id"><th>{{ member.id }} / {{ junction.id }}</th><td>{{ number(junction.rtheta_k_w) }}</td><td>{{ number(junction.heat_w) }} / {{ number(junction.known_heat_w) }}</td><td>{{ number(junction.tj_c) }}</td><td>{{ number(junction.known_heat_tj_c) }}</td><td>{{ junction.status }}<small>{{ junction.omissions.join('; ') }}</small><small>{{ junction.source ?? 'Unknown source' }} · {{ JSON.stringify(junction.conditions) }}</small></td></tr></template></tbody></table></div>
        <details><summary>Python evidence by role and request identity</summary><label>Evidence role<select v-model="selectedEvidence"><option v-for="(_, role) in result.evidence_by_role" :key="role" :value="role">{{ role }}</option></select></label><pre>{{ JSON.stringify(result.evidence_by_role[selectedEvidence], null, 2) }}</pre><pre>{{ JSON.stringify(result.provenance, null, 2) }}</pre></details>
      </section>
    </template>
  </section>
</template>

<style scoped>
.circuit-summary { min-width:0; border:1px solid #d1dbe6; background:#f8fafc; border-radius:10px; padding:1rem; margin:1rem 0; overflow-wrap:anywhere; }
label { display:grid; gap:.4rem; margin:.8rem 0; }.check { display:flex; align-items:baseline; gap:.5rem; }
input:not([type=checkbox]),select,textarea { width:100%; min-width:0; box-sizing:border-box; font:inherit; padding:.6rem; border:1px solid #94a3b8; border-radius:5px; }
button { padding:.6rem .9rem; margin:.3rem 0; border:1px solid #94a3b8; background:white; color:#172033; border-radius:5px; cursor:pointer; }button:disabled { opacity:.5; }.primary { background:#155e75; color:white; }
.actions { display:flex; flex-wrap:wrap; gap:.6rem; }.columns { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }.owner-row { display:grid; grid-template-columns:140px minmax(0,1fr) minmax(0,1fr); gap:.8rem; align-items:center; }.owner-component,.junction { border-bottom:1px solid #cbd5e1; padding:.6rem 0; }
fieldset { min-width:0; }summary { cursor:pointer; }pre { white-space:pre-wrap; overflow-wrap:anywhere; }.notice { padding:1rem; background:#eef8fb; border-left:4px solid #31829c; }.error { color:#991b1b; background:#fef2f2; padding:1rem; }
.table-scroll { max-width:100%; overflow:auto; }table { width:100%; min-width:760px; border-collapse:collapse; text-align:left; font-size:.9rem; }th,td { padding:.6rem; border-bottom:1px solid #cbd5e1; vertical-align:top; }small { display:block; color:#475569; }
@media(max-width:700px) { .columns,.owner-row { grid-template-columns:1fr; }.circuit-summary { padding:.8rem; } }
</style>
