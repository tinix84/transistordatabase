<script setup>
const props = defineProps({ modelValue: { type: Object, required: true }, prefix: { type: String, default: 'Fixed' }, preserveNumericText: { type: Boolean, default: false } })
const emit = defineEmits(['update:modelValue'])
const numbers = [['t_j_c', 'Junction temperature (°C)'], ['v_g_on_v', 'Turn-on gate voltage (V)'], ['v_g_off_v', 'Turn-off gate voltage (V)'], ['r_g_on_ohm', 'Turn-on gate resistance (Ω)'], ['r_g_off_ohm', 'Turn-off gate resistance (Ω)']]
function update(key, value) { emit('update:modelValue', { ...props.modelValue, [key]: value }) }
function updateNumber(key, event) {
  const field = event.target
  const value = props.preserveNumericText ? (field.value !== '' ? field.value : field.validity.badInput ? 'Invalid numeric input' : null) : field.value === '' ? null : field.valueAsNumber
  update(key, value)
}
</script>
<template>
  <div class="conditions">
    <label v-for="[key, title] in numbers" :key="key">{{ title }}<input type="number" step="any" :aria-label="`${prefix} ${title}`" :value="modelValue[key]" placeholder="Unknown" @input="updateNumber(key, $event)"></label>
    <label v-for="[key, title] in [['commutation_device', 'Commutation device'], ['commutation_source', 'Commutation source']]" :key="key">{{ title }}<input :aria-label="`${prefix} ${title}`" :value="modelValue[key]" placeholder="Unknown" @input="update(key, $event.target.value === '' ? null : $event.target.value)"></label>
    <label>Energy convention<input :aria-label="`${prefix} energy convention`" :value="modelValue.energy_convention" required @input="update('energy_convention', $event.target.value)"></label>
  </div>
</template>
<style scoped>
.conditions { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:.7rem; }label { display:grid; gap:.35rem; margin:.4rem 0; }input { width:100%; min-width:0; padding:.5rem; font:inherit; color:inherit; background:var(--bg-secondary); border:1px solid var(--border-color,#aaa); border-radius:4px; }
</style>
