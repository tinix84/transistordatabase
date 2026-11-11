<template>
  <fieldset class="assembly-table">
    <legend>{{ title }}</legend>
    <label>Recorded source <input :value="modelValue.source" :aria-label="title + ' source'" @input="set('source', $event.target.value)" /></label>
    <div class="columns">
      <label>Current magnitude (A)<textarea :value="modelValue.x.join('\n')" :aria-label="title + ' current A'" rows="4" @input="samples('x', $event.target.value)" /></label>
      <label>{{ energy ? 'Event energy (J)' : 'Voltage-drop magnitude (V)' }}<textarea :value="modelValue.y.join('\n')" :aria-label="title + ' values'" rows="4" @input="samples('y', $event.target.value)" /></label>
    </div>
    <div class="conditions">
      <div v-for="(value, key) in modelValue.conditions" :key="key">
        <label>{{ labels[key] || key }} <input :value="value" type="number" step="any" :disabled="value === null" :aria-label="title + ' ' + key" @input="condition(key, $event.target.value === '' ? null : Number($event.target.value))" /></label>
        <label><input type="checkbox" :checked="value === null" :aria-label="title + ' ' + key + ' unknown'" @change="condition(key, $event.target.checked ? null : 0)" /> Unknown</label>
      </div>
    </div>
    <p v-if="error" role="alert">{{ error }}</p>
  </fieldset>
</template>
<script setup>
import { onBeforeUnmount, ref } from 'vue'
const props = defineProps({ modelValue: { type: Object, required: true }, title: String, energy: Boolean })
const emit = defineEmits(['update:modelValue', 'invalid'])
const error = ref('')
onBeforeUnmount(() => { for (const key of ['x', 'y']) emit('invalid', { key: props.title + key, invalid: false }) })
const labels = { t_j_c: 'Junction temperature (°C)', v_g_v: 'Gate bias (V)', v_supply_v: 'Supply test voltage (V)', r_g_ohm: 'Gate resistance (Ω)' }
function set(key, value) { emit('update:modelValue', { ...props.modelValue, [key]: value }) }
function condition(key, value) { set('conditions', { ...props.modelValue.conditions, [key]: value }) }
function samples(key, text) {
  emit('invalid', { key: props.title + key, invalid: true })
  const tokens = text.trim().split(/[\s,;]+/)
  const values = tokens.map(Number)
  if (!text.trim() || values.length < 2 || values.some(value => !Number.isFinite(value))) {
    error.value = 'Enter at least two finite numeric samples. Both columns must have matching lengths.'
    return
  }
  error.value = ''; set(key, values); emit('invalid', { key: props.title + key, invalid: false })
}
</script>
<style scoped>
fieldset { min-width: 0; margin: 1rem 0; border: 1px solid #8e9eab; border-radius: 6px; padding: .8rem; }
label { display: block; margin: .4rem 0; } input:not([type=checkbox]), textarea { box-sizing: border-box; width: 100%; min-width: 0; padding: .45rem; }
.columns, .conditions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .7rem; }
@media(max-width: 600px) { .columns, .conditions { grid-template-columns: 1fr; } }
</style>
