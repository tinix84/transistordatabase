<script setup>
const props = defineProps({ modelValue: { type: Object, required: true }, label: { type: String, default: 'Gate' } })
const emit = defineEmits(['update:modelValue'])
function update(key, value) {
  const changed = { ...props.modelValue, [key]: value }
  if (key === 'scale') {
    changed.low = value === 'relative' ? '0.25' : '2.5'
    changed.high = value === 'relative' ? '0.75' : '7.5'
  }
  emit('update:modelValue', changed)
}
</script>

<template>
  <fieldset class="hysteresis-settings" data-testid="hysteresis-settings">
    <legend>{{ label }} hysteresis</legend>
    <label>Threshold reference<select :value="modelValue.scale" @change="update('scale', $event.target.value)" data-testid="hysteresis-scale"><option value="absolute">Declared voltages (V)</option><option value="relative">Fractions of a declared calibrated voltage range</option></select></label>
    <div class="hysteresis-columns">
      <label>{{ modelValue.scale === 'absolute' ? 'Low threshold (V)' : 'Low fraction (0 to 1)' }}<input :value="modelValue.low" @input="update('low', $event.target.value)" type="number" step="any" data-testid="hysteresis-low"></label>
      <label>{{ modelValue.scale === 'absolute' ? 'High threshold (V)' : 'High fraction (0 to 1)' }}<input :value="modelValue.high" @input="update('high', $event.target.value)" type="number" step="any" data-testid="hysteresis-high"></label>
    </div>
    <div v-if="modelValue.scale === 'relative'" class="hysteresis-columns">
      <label>Calibrated minimum gate voltage (V)<input :value="modelValue.minimum" @input="update('minimum', $event.target.value)" type="number" step="any" data-testid="hysteresis-minimum"></label>
      <label>Calibrated maximum gate voltage (V)<input :value="modelValue.maximum" @input="update('maximum', $event.target.value)" type="number" step="any" data-testid="hysteresis-maximum"></label>
    </div>
    <label>Initial state at sample zero<select :value="modelValue.initial" @change="update('initial', $event.target.value)" data-testid="hysteresis-initial"><option value="auto">Infer only when the first sample is outside the band</option><option value="off">Off</option><option value="on">On</option></select></label>
    <p>Strict crossings change state. Equality and values inside the band retain the previous sampled state. Choose an initial state when the first sample is inside or on the band. A declared state must agree with a first sample outside the band.</p>
  </fieldset>
</template>

<style scoped>
.hysteresis-settings { border: 1px solid #94a3b8; border-radius: .5rem; padding: .8rem; min-width: 0; }
label { display: grid; gap: .4rem; margin: .6rem 0; }
input, select { min-width: 0; width: 100%; box-sizing: border-box; padding: .6rem; font: inherit; }
.hysteresis-columns { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: .8rem; }
p { line-height: 1.5; }
@media(max-width: 780px) { .hysteresis-columns { grid-template-columns: 1fr; } }
</style>
