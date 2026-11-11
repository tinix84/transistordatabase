<script setup>
const props = defineProps({ modelValue: { type: Object, required: true }, title: String, energy: Boolean })
const emit = defineEmits(['update:modelValue'])
const fields = [
  ['t_j_c', 'Junction temperature (°C)'], ['v_g_v', 'Gate voltage (V)'],
  ['r_g_ohm', 'Gate resistance (Ω)'], ['v_supply_v', 'Switching supply (V)'],
]
function update(key, value) { emit('update:modelValue', { ...props.modelValue, [key]: value }) }
function condition(group, key, value) {
  update(group, { ...props.modelValue[group], [key]: value })
}
</script>

<template>
  <fieldset class="model-editor">
    <legend>{{ title }}</legend>
    <label><input type="checkbox" :checked="modelValue.enabled" @change="update('enabled', $event.target.checked)"> Supply this direction's model</label>
    <template v-if="modelValue.enabled">
      <label>Model
        <select :value="modelValue.mode" @change="update('mode', $event.target.value)">
          <option v-if="!energy" value="linear-drop-estimate">Threshold + resistance estimate</option>
          <option value="recorded">Recorded bounded curve</option>
          <option v-if="energy" value="voltage-scaling-estimate">Energy proportional to bus voltage estimate</option>
        </select>
      </label>
      <label>Source identifier<input :value="modelValue.source" @input="update('source', $event.target.value)" placeholder="File, measurement or explicit assumption"></label>
      <template v-if="modelValue.mode === 'linear-drop-estimate'">
        <div class="columns">
          <label>Threshold voltage (V)<input type="number" step="any" :value="modelValue.offset_v" @input="update('offset_v', $event.target.valueAsNumber)"></label>
          <label>Resistance (Ω)<input type="number" step="any" :value="modelValue.resistance_ohm" @input="update('resistance_ohm', $event.target.valueAsNumber)"></label>
        </div>
      </template>
      <label v-else>{{ energy ? 'Magnitude current (A), energy (J)' : 'Magnitude current (A), voltage drop (V)' }} — one pair per line
        <textarea :value="modelValue.pairs" @input="update('pairs', $event.target.value)" rows="4" spellcheck="false"></textarea>
      </label>
      <label v-if="modelValue.mode !== 'recorded'" class="selection"><input type="checkbox" :checked="modelValue.selected" @change="update('selected', $event.target.checked)"> I select this named estimate and its assumptions</label>
      <details>
        <summary>Recorded and requested conditions (unknown stays unknown)</summary>
        <p v-if="modelValue.mode === 'voltage-scaling-estimate'">Only supply voltage is scaled. All other conditions must match. A waveform bus column overrides the requested supply at each event.</p>
        <div v-for="group in (modelValue.mode === 'linear-drop-estimate' ? ['recorded'] : ['recorded', 'requested'])" :key="group">
          <strong>{{ group === 'recorded' ? 'Recorded / assumed conditions' : 'Requested conditions' }}</strong>
          <div class="conditions">
            <label v-for="[key, name] in fields" :key="key">{{ name }}
              <span><input type="checkbox" :checked="modelValue[group][key] !== null" @change="condition(group, key, $event.target.checked ? 0 : null)"> Known</span>
              <input v-if="modelValue[group][key] !== null" type="number" step="any" :value="modelValue[group][key]" :aria-invalid="!Number.isFinite(modelValue[group][key])" @input="condition(group, key, $event.target.valueAsNumber)">
              <span v-if="modelValue[group][key] !== null && !Number.isFinite(modelValue[group][key])" role="alert">Known condition needs a finite number; choose Unknown explicitly if unrecorded.</span>
              <span v-else class="unknown">Unknown</span>
            </label>
          </div>
        </div>
      </details>
    </template>
  </fieldset>
</template>

<style scoped>
.model-editor { min-width: 0; border: 1px solid var(--border-color,#cbd5e1); border-radius: 10px; padding: 1rem; background: var(--bg-secondary,#fff); color:var(--text-primary,#1e293b); }
legend { font-weight: 700; padding: 0 .4rem; }
label { display: grid; gap: .35rem; margin: .65rem 0; }
label:has(> input[type=checkbox]), label.selection { display: flex; align-items: baseline; }
input:not([type=checkbox]), select, textarea { width: 100%; box-sizing: border-box; min-width: 0; border: 1px solid #94a3b8; border-radius: 5px; padding: .55rem; font: inherit; background:var(--bg-secondary,#fff); color:inherit; }
textarea { font-family: monospace; }
.columns, .conditions { display: grid; gap: .8rem; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.unknown { color: #64748b; padding: .55rem; }
details { margin-top: .8rem; font-size: .9rem; }
summary { cursor: pointer; }
@media (max-width: 600px) { .conditions, .columns { grid-template-columns: 1fr; } }
</style>
