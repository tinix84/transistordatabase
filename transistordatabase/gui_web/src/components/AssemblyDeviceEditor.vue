<template>
  <fieldset class="assembly-device">
    <legend>{{ title }}</legend>
    <label>Device source <input :value="modelValue.source" :aria-label="title + ' device source'" @input="set('source', $event.target.value)" /></label>
    <div class="fields">
      <label v-for="(label, key) in fields" :key="key">{{ label }}
        <input :value="modelValue[key]" type="number" step="any" :aria-label="title + ' ' + key" @input="set(key, $event.target.value === '' ? null : Number($event.target.value))" />
        <span>Leave blank if unknown.</span>
      </label>
      <label>Cost currency <input :value="modelValue.currency" :aria-label="title + ' currency'" @input="set('currency', $event.target.value || null)" /></label>
    </div>
    <div v-for="(label, key) in curves" :key="key">
      <label><input type="checkbox" :checked="!!modelValue[key]" :aria-label="title + ' include ' + key" @change="enable(key, $event.target.checked)" /> {{ label }} available</label>
      <AssemblyTableEditor v-if="modelValue[key]" :title="title + ' ' + label" :energy="key !== 'conduction'" :model-value="modelValue[key]" @update:model-value="set(key, $event)" @invalid="emit('invalid', $event)" />
    </div>
    <label><input type="checkbox" :checked="!!modelValue.foster" :aria-label="title + ' include Foster'" @change="toggleFoster($event.target.checked)" /> Internal Foster poles available</label>
    <div v-if="modelValue.foster" class="fields">
      <label v-for="(label, key) in fosterFields" :key="key">{{ label }}
        <input :value="modelValue.foster[key]?.join(', ') ?? ''" :aria-label="title + ' ' + key" @input="pole(key, $event.target.value)" />
      </label>
    </div>
    <p v-for="(message, key) in poleErrors" :key="key" role="alert">{{ message }}</p>
  </fieldset>
</template>
<script setup>
import { onBeforeUnmount, ref } from 'vue'
import AssemblyTableEditor from './AssemblyTableEditor.vue'
const props = defineProps({ modelValue: { type: Object, required: true }, title: String })
const emit = defineEmits(['update:modelValue', 'invalid'])
const fields = { blocking_voltage_v: 'Blocking rating (V)', current_rating_a: 'Current rating (A)', area_m2: 'Independent contact area (m²)', cost: 'Cost in stated currency', weight_kg: 'Weight (kg)', internal_resistance_k_per_w: 'Internal thermal resistance (K/W)', shared_sink_resistance_k_per_w: 'Shared external sink resistance (K/W)' }
const curves = { conduction: 'Conduction curve', turn_on: 'Turn-on energy', turn_off: 'Turn-off energy' }
const fosterFields = { resistance_k_per_w: 'Pole resistances (K/W), comma separated', tau_s: 'Time constants (s), comma separated', capacity_j_per_k: 'Optional pole capacities (J/K), blank means unknown' }
const poleErrors = ref({})
function clearPoleErrors() { for (const key of Object.keys(fosterFields)) emit('invalid', { key: props.title + key, invalid: false }); poleErrors.value = {} }
onBeforeUnmount(clearPoleErrors)
function toggleFoster(selected) { if (!selected) clearPoleErrors(); set('foster', selected ? { resistance_k_per_w: [.2, .3], tau_s: [.01, .1], capacity_j_per_k: null } : null) }
function set(key, value) { emit('update:modelValue', { ...props.modelValue, [key]: value }) }
function enable(key, selected) {
  set(key, selected ? { kind: key === 'conduction' ? 'channel' : 'energy', x: [0, 10], y: key === 'conduction' ? [0, 2] : [0.0001, 0.001], conditions: key === 'conduction' ? { t_j_c: 25, v_g_v: 15 } : { t_j_c: 25, v_g_v: 15, v_supply_v: 400, r_g_ohm: null }, source: 'Illustrative curve — replace with a recorded source' } : null)
}
function pole(key, text) {
  const values = text.trim() ? text.trim().split(/[\s,;]+/).map(Number) : null
  if (values?.some(v => !Number.isFinite(v)) || (values === null && key !== 'capacity_j_per_k')) { poleErrors.value[key] = `${fosterFields[key]}: enter finite numeric values.`; emit('invalid', { key: props.title + key, invalid: true }); return }
  const remaining = { ...poleErrors.value }; delete remaining[key]; poleErrors.value = remaining
  set('foster', { ...props.modelValue.foster, [key]: values }); emit('invalid', { key: props.title + key, invalid: false })
}
</script>
<style scoped>
fieldset { min-width: 0; border: 1px solid #899bad; border-radius: 8px; padding: 1rem; }
.fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .6rem; }
label { display: block; margin: .5rem 0; } input:not([type=checkbox]) { box-sizing: border-box; width: 100%; padding: .45rem; } span { font-size: .85em; }
@media(max-width: 600px) { .fields { grid-template-columns: 1fr; } }
</style>
