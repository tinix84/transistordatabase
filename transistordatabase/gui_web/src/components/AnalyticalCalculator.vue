<template>
  <div class="analytical-calculator">
    <h2>Analytical Loss Models</h2>
    <p>Reusable scientific estimates and bounded device-parameter calculations with explicit conditions.</p>
    <div class="model-tabs">
      <button v-for="tab in tabs" :key="tab.key" :class="{ active: activeModel === tab.key }"
        @click="activeModel = tab.key">{{ tab.label }}</button>
    </div>
    <BielaCalculator v-show="activeModel === 'biela'" :extraction="bielaExtraction" />
    <GateChargeCalculator v-if="activeModel === 'gate_charge'" />
    <DeviceParameterCalculator v-show="activeModel === 'parameters'" @apply-biela="stageBielaExtraction" @withdraw-biela="bielaExtraction = null" />
    <DeviceModelFitting v-if="activeModel === 'device-fitting'" />
  </div>
</template>
<script setup>
import { ref } from 'vue'
import BielaCalculator from './BielaCalculator.vue'
import GateChargeCalculator from './GateChargeCalculator.vue'
import DeviceParameterCalculator from './DeviceParameterCalculator.vue'
import DeviceModelFitting from './DeviceModelFitting.vue'
const initialModel = new URLSearchParams(window.location.search).get('model')
const activeModel = ref(['biela', 'gate_charge', 'parameters', 'device-fitting'].includes(initialModel) ? initialModel : 'biela')
const bielaExtraction = ref(null)
function stageBielaExtraction(value) { bielaExtraction.value = value; activeModel.value = 'biela' }
const tabs = [{ key: 'biela', label: 'Biela Switching Loss Model' },
  { key: 'gate_charge', label: 'Gate Charge Timing Model' },
  { key: 'parameters', label: 'Device Parameter Operations' },
  { key: 'device-fitting', label: 'Device model fitting' }]
</script>
<style scoped>
.analytical-calculator { padding: 20px; max-width: 1400px; margin: auto; }
.model-tabs { display: flex; flex-wrap: wrap; gap: 10px; margin: 24px 0; }
.model-tabs button { padding: 12px; border: 1px solid #cbd3dc; background: white; cursor: pointer; }
.model-tabs button.active { border-bottom: 3px solid #2875b8; font-weight: bold; }
@media (max-width: 600px) { .analytical-calculator { padding: 8px; } }
</style>
