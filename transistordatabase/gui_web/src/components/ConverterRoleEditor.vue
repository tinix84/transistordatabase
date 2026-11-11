<script setup>
import WaveformModelEditor from './WaveformModelEditor.vue'
defineProps({ modelValue: Object, role: String, sourceOmission: Boolean, envelope: Boolean })
const emit = defineEmits(['update:modelValue'])
function update(value, key, next) { emit('update:modelValue', { ...value, [key]: next }) }
</script>

<template>
  <details class="role-editor">
    <summary>{{ role }} — device models and independent conditions</summary>
    <label>Role / device source<input :value="modelValue.source" @input="update(modelValue, 'source', $event.target.value)"></label>
    <label class="check"><input type="checkbox" :checked="modelValue.conductionOmit" @change="update(modelValue, 'conductionOmit', $event.target.checked)"> Explicitly omit conduction</label>
    <label v-if="modelValue.conductionOmit">Conduction omission reason<input :value="modelValue.conductionReason" @input="update(modelValue, 'conductionReason', $event.target.value)"></label>
    <div v-else class="columns"><WaveformModelEditor :model-value="modelValue.positive" title="Positive-current conduction" @update:model-value="update(modelValue, 'positive', $event)"/><WaveformModelEditor :model-value="modelValue.negative" title="Negative-current conduction" @update:model-value="update(modelValue, 'negative', $event)"/></div>
    <p v-if="sourceOmission" class="notice">The selected archive source omits switching in this role. Select and explain that scope above; the complete converter total will remain unknown.</p>
    <template v-else>
      <template v-if="!envelope"><label>Event evidence<select :value="modelValue.eventMode" @change="update(modelValue, 'eventMode', $event.target.value)"><option value="gate">Supplied gate above explicit threshold</option><option value="masks">Explicit N−1 interval masks</option><option value="omitted">Explicitly omitted event evidence</option></select></label><label v-if="modelValue.eventMode === 'gate'">Gate threshold (V)<input type="number" step="any" :value="modelValue.threshold" @input="update(modelValue, 'threshold', $event.target.valueAsNumber)"></label><template v-if="modelValue.eventMode === 'masks'"><label>Turn-on interval mask (0 or 1)<input :value="modelValue.onMask" @input="update(modelValue, 'onMask', $event.target.value)"></label><label>Turn-off interval mask (0 or 1)<input :value="modelValue.offMask" @input="update(modelValue, 'offMask', $event.target.value)"></label></template><label v-if="modelValue.eventMode === 'omitted'">Event omission reason<input :value="modelValue.eventReason" @input="update(modelValue, 'eventReason', $event.target.value)"></label></template>
      <template v-for="side in ['on', 'off']" :key="side">
        <h4>{{ side === 'on' ? 'Turn-on energy' : 'Turn-off energy' }}</h4>
        <label class="check"><input type="checkbox" :checked="modelValue[side + 'Omit']" @change="update(modelValue, side + 'Omit', $event.target.checked)"> Explicitly omit turn-{{ side }}</label>
        <label v-if="modelValue[side + 'Omit']">Turn-{{ side }} omission reason<input :value="modelValue[side + 'Reason']" @input="update(modelValue, side + 'Reason', $event.target.value)"></label>
        <div v-else class="columns"><WaveformModelEditor :model-value="modelValue[side + 'Positive']" :title="`Positive-current turn-${side}`" energy @update:model-value="update(modelValue, side + 'Positive', $event)"/><WaveformModelEditor :model-value="modelValue[side + 'Negative']" :title="`Negative-current turn-${side}`" energy @update:model-value="update(modelValue, side + 'Negative', $event)"/></div>
      </template>
    </template>
  </details>
</template>

<style scoped>
.role-editor { border:1px solid #cbd5e1; border-radius:8px; padding:1rem; margin:1rem 0; min-width:0; background:white; }
summary { font-weight:700; cursor:pointer; overflow-wrap:anywhere; }
label { display:grid; gap:.35rem; margin:.8rem 0; }
.check { display:flex; align-items:baseline; }
input:not([type=checkbox]), select { width:100%; box-sizing:border-box; min-width:0; font:inherit; padding:.55rem; }
.columns { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }
.notice { background:#fff7ed; padding:.8rem; }
@media(max-width:700px) { .columns { grid-template-columns:1fr; } }
</style>
