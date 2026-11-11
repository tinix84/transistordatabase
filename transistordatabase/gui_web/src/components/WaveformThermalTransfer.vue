<script setup>
import { onBeforeUnmount, reactive, ref, watch } from 'vue'
import { analysisApi } from '../services/api.js'
import { analysisError, sameAnalysisEvidence } from './analysisFiles.js'
const props = defineProps({ analysis: { type: Object, required: true }, acceptedRequest: { type: Object, required: true } })
const emit = defineEmits(['thermal-handoff'])
const selection = reactive({ positive: '', negative: '', zero: '', network: '', partial: false })
const error = ref(''), busy = ref(false)
let revision = 0, disposed = false
watch([() => props.analysis, selection], () => { revision++; error.value = ''; busy.value = false }, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
async function transfer() {
  const token = ++revision; error.value = ''; busy.value = true
  try {
    if (!selection.positive || !selection.negative || !selection.zero || !selection.network) throw new Error('Select each direction owner and the physical network owner.')
    const response = await analysisApi.thermalHandoff({ waveform_request: props.acceptedRequest,
      direction_owners: { positive: selection.positive, negative: selection.negative, zero: selection.zero },
      network_heat_owner: selection.network, event_mode: 'impulse', allow_partial: selection.partial })
    if (!sameAnalysisEvidence(response.source_request, props.acceptedRequest) || !sameAnalysisEvidence(response.waveform_result, props.analysis)) throw new Error('Recomputed waveform evidence differs from the displayed result. Calculate the waveform again before thermal transfer.')
    if (response.method_version !== 'waveform-thermal-handoff-v1' || !/^[a-f0-9]{64}$/.test(response.source_input_sha256) || !/^[a-f0-9]{64}$/.test(response.input_sha256)) throw new Error('The thermal handoff has an unsupported method or missing source hashes.')
    if (!disposed && token === revision) emit('thermal-handoff', JSON.parse(JSON.stringify(response)))
  } catch (failure) { if (!disposed && token === revision) error.value = analysisError(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
</script>

<template>
  <details class="thermal-transfer">
    <summary>Send exact waveform heat to the thermal workbench</summary>
    <p>Declare physical ownership for all current directions, including zero-current switching events. Current direction does not identify a physical junction.</p>
    <div class="fields">
      <label v-for="direction in ['positive', 'negative', 'zero']" :key="direction">{{ direction }}-current heat owner
        <select v-model="selection[direction]" :aria-label="`${direction}-current heat owner`"><option value="">Select physical junction</option><option value="forward">Forward junction</option><option value="reverse">Reverse junction</option></select>
      </label>
      <label>Physical thermal network<select v-model="selection.network" aria-label="Handoff thermal network"><option value="">Select network ownership</option><option value="forward">Forward junction only</option><option value="reverse">Reverse junction only</option><option value="shared-junction">Both branches explicitly share one junction</option></select></label>
    </div>
    <label class="check"><input v-model="selection.partial" type="checkbox"> I accept known components only if a component is omitted. Missing heat remains unknown.</label>
    <p>Python preserves exact cubic heat segments and pre-edge event times. Switching heat uses ideal instantaneous energy deposition.</p>
    <p>The transfer requires an exact match between the displayed request/result and the recomputed Python evidence, including source identity.</p>
    <button :disabled="busy" type="button" @click="transfer">{{ busy ? 'Preparing exact heat…' : 'Prepare thermal handoff' }}</button>
    <p v-if="error" role="alert">{{ error }}</p>
  </details>
</template>

<style scoped>
.thermal-transfer { padding:1rem; border:1px solid var(--border-color,#aaa); border-radius:8px; margin:1rem 0; }
summary { cursor:pointer; font-weight:600; }p { margin:.6rem 0; line-height:1.5; }
.fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:.7rem; }
label { display:grid; gap:.3rem; }select { width:100%; min-width:0; padding:.5rem; background:var(--bg-secondary); color:inherit; border:1px solid var(--border-color,#aaa); }
.check { display:flex; align-items:baseline; gap:.5rem; margin:1rem 0; }
button { padding:.6rem; cursor:pointer; }[role=alert] { color:#b02828; overflow-wrap:anywhere; }
</style>
