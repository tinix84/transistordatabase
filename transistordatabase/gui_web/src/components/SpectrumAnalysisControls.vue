<script setup>
const props = defineProps({ modelValue: { type: Object, required: true } })
const emit = defineEmits(['update:modelValue'])
function update(key, value) { emit('update:modelValue', { ...props.modelValue, [key]: value }) }
</script>

<template>
  <section class="spectrum-controls" data-testid="spectrum-analysis-controls">
    <label class="check"><input type="checkbox" :checked="modelValue.maskEnabled" @change="update('maskEnabled', $event.target.checked)" data-testid="spectrum-mask-enabled"> Screen every current FFT bin against a declared peak mask</label>
    <fieldset v-if="modelValue.maskEnabled">
      <legend>Declared current mask (A_peak)</legend>
      <label>Mask source<input :value="modelValue.maskSource" @input="update('maskSource', $event.target.value)" data-testid="spectrum-mask-source"></label>
      <label>Frequency mask CSV (Hz, A_peak)<textarea :value="modelValue.maskCsv" @input="update('maskCsv', $event.target.value)" rows="6" data-testid="spectrum-mask-csv"></textarea></label>
      <label>Mask interpolation<select :value="modelValue.interpolation" @change="update('interpolation', $event.target.value)" data-testid="spectrum-mask-interpolation"><option value="exact">Exact exported FFT frequency coordinates only</option><option value="linear-frequency">Linear frequency and peak amplitude within declared segments</option><option value="log-frequency">Log frequency and linear peak amplitude within positive-frequency segments</option></select></label>
      <p>Use the exact frequencies from the spectrum JSON or CSV. A blank or null limit marks an unknown gap. Interpolation does not cross gaps or extrapolate. Equality passes. Values are peak amperes, without dB, RMS, or detector conversion.</p>
    </fieldset>
    <label class="check"><input type="checkbox" :checked="modelValue.bandsEnabled" @change="update('bandsEnabled', $event.target.checked)" data-testid="spectrum-bands-enabled"> Calculate raw windowed discrete RMS for declared bands</label>
    <fieldset v-if="modelValue.bandsEnabled">
      <legend>Current band RMS (A_rms)</legend>
      <label>Frequency bands CSV<textarea :value="modelValue.bandsCsv" @input="update('bandsCsv', $event.target.value)" rows="5" data-testid="spectrum-bands-csv"></textarea></label>
      <p>Endpoint choices are closed, open, left-closed, and right-closed. Bands outside the physical Nyquist range are rejected. Empty bands remain unavailable. Overlapping bands are independent and must not be added.</p>
      <p>RMS uses the raw windowed, detrended discrete FFT. Window power gain is reported without an implicit correction.</p>
    </fieldset>
  </section>
</template>

<style scoped>
.spectrum-controls { min-width: 0; margin: 1rem 0; }
fieldset { min-width: 0; border: 1px solid #94a3b8; border-radius: .5rem; padding: .8rem; }
label { display: grid; gap: .4rem; margin: .7rem 0; } .check { display: flex; align-items: baseline; gap: .5rem; }
input:not([type=checkbox]), select, textarea { width: 100%; min-width: 0; box-sizing: border-box; padding: .6rem; font: inherit; }
textarea { font-family: monospace; } p { line-height: 1.5; overflow-wrap: anywhere; }
</style>
