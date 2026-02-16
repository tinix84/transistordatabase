<template>
  <div class="transistor-form">
    <div class="form-header">
      <h2>{{ isEditing ? 'Edit Transistor' : 'Create New Transistor' }}</h2>
      <button @click="$emit('cancel')" class="btn btn-secondary">Cancel</button>
    </div>

    <!-- Section Tabs -->
    <div class="section-tabs">
      <button
        v-for="tab in availableTabs"
        :key="tab.key"
        @click="activeTab = tab.key"
        :class="['tab-btn', { active: activeTab === tab.key }]"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- General Info Tab -->
    <form v-show="activeTab === 'general'" @submit.prevent="handleSubmit" class="form-content">
      <div class="form-section">
        <h3>Metadata</h3>
        <div class="form-grid">
          <div class="form-group">
            <label for="name">Name *</label>
            <input
              id="name"
              v-model="formData.metadata.name"
              type="text"
              required
              placeholder="e.g. IGBT_1200V_100A"
            />
          </div>

          <div class="form-group">
            <label for="type">Type *</label>
            <select id="type" v-model="formData.metadata.type" required>
              <option value="">Select type</option>
              <option value="IGBT">IGBT</option>
              <option value="MOSFET">MOSFET</option>
              <option value="SiC-MOSFET">SiC-MOSFET</option>
              <option value="GaN">GaN</option>
              <option value="BJT">BJT</option>
              <option value="Diode">Diode</option>
            </select>
          </div>

          <div class="form-group">
            <label for="manufacturer">Manufacturer *</label>
            <input
              id="manufacturer"
              v-model="formData.metadata.manufacturer"
              type="text"
              required
              placeholder="e.g. Infineon, Rohm, etc."
            />
          </div>

          <div class="form-group">
            <label for="housing">Housing Type *</label>
            <select id="housing" v-model="formData.metadata.housing_type" required>
              <option value="">Select housing</option>
              <option value="TO-247">TO-247</option>
              <option value="TO-220">TO-220</option>
              <option value="TO-252">TO-252</option>
              <option value="SO-8">SO-8</option>
              <option value="SOT-23">SOT-23</option>
              <option value="Other">Other</option>
            </select>
          </div>

          <div class="form-group">
            <label for="author">Author</label>
            <input
              id="author"
              v-model="formData.metadata.author"
              type="text"
              placeholder="Optional"
            />
          </div>

          <div class="form-group form-group-wide">
            <label for="comment">Comment</label>
            <textarea
              id="comment"
              v-model="formData.metadata.comment"
              placeholder="Optional comments or notes"
              rows="3"
            ></textarea>
          </div>
        </div>
      </div>

      <div class="form-section">
        <h3>Electrical Ratings</h3>
        <div class="form-grid">
          <div class="form-group">
            <label for="v_abs_max">V<sub>abs,max</sub> (V) *</label>
            <input
              id="v_abs_max"
              v-model.number="formData.electrical.v_abs_max"
              type="number"
              step="0.1"
              min="0"
              required
              placeholder="e.g. 1200"
            />
          </div>

          <div class="form-group">
            <label for="i_abs_max">I<sub>abs,max</sub> (A) *</label>
            <input
              id="i_abs_max"
              v-model.number="formData.electrical.i_abs_max"
              type="number"
              step="0.1"
              min="0"
              required
              placeholder="e.g. 100"
            />
          </div>

          <div class="form-group">
            <label for="i_cont">I<sub>cont</sub> (A) *</label>
            <input
              id="i_cont"
              v-model.number="formData.electrical.i_cont"
              type="number"
              step="0.1"
              min="0"
              required
              placeholder="e.g. 80"
            />
          </div>

          <div class="form-group">
            <label for="t_j_max">T<sub>j,max</sub> (°C) *</label>
            <input
              id="t_j_max"
              v-model.number="formData.electrical.t_j_max"
              type="number"
              step="0.1"
              required
              placeholder="e.g. 175"
            />
          </div>
        </div>
      </div>

      <div class="form-section">
        <h3>Thermal Properties</h3>
        <div class="form-grid">
          <div class="form-group">
            <label for="r_th_cs">R<sub>th,cs</sub> (K/W) *</label>
            <input
              id="r_th_cs"
              v-model.number="formData.thermal.r_th_cs"
              type="number"
              step="0.001"
              min="0"
              required
              placeholder="e.g. 0.5"
            />
          </div>

          <div class="form-group">
            <label for="housing_area">Housing Area (cm²) *</label>
            <input
              id="housing_area"
              v-model.number="formData.thermal.housing_area"
              type="number"
              step="0.1"
              min="0"
              required
              placeholder="e.g. 1.0"
            />
          </div>

          <div class="form-group">
            <label for="cooling_area">Cooling Area (cm²) *</label>
            <input
              id="cooling_area"
              v-model.number="formData.thermal.cooling_area"
              type="number"
              step="0.1"
              min="0"
              required
              placeholder="e.g. 1.0"
            />
          </div>
        </div>
      </div>

      <div class="form-actions">
        <button type="button" @click="$emit('cancel')" class="btn btn-secondary">
          Cancel
        </button>
        <button type="button" @click="validateForm" class="btn btn-secondary">
          Validate
        </button>
        <button type="submit" class="btn btn-primary" :disabled="isSubmitting">
          {{ isSubmitting ? 'Saving...' : (isEditing ? 'Update' : 'Create') }}
        </button>
      </div>
    </form>

    <!-- Channel Curves Tab -->
    <div v-show="activeTab === 'channel'" class="curve-tab-content">
      <div class="curve-tab-header">
        <h3>Channel Characteristic Curves</h3>
        <div class="component-toggle">
          <button
            @click="curveComponent = 'switch'"
            :class="['toggle-btn', { active: curveComponent === 'switch' }]"
          >Switch</button>
          <button
            @click="curveComponent = 'diode'"
            :class="['toggle-btn', { active: curveComponent === 'diode' }]"
          >Diode</button>
        </div>
      </div>

      <CurveList
        :curves="curveComponent === 'switch' ? curves.switch_channel : curves.diode_channel"
        :title="curveComponent === 'switch' ? 'Switch Channel Curves' : 'Diode Channel Curves'"
        curveType="Channel"
        @add="openCurveEditor('channel')"
        @view="viewCurve('channel', $event)"
        @delete="deleteCurve('channel', $event)"
      />

      <CurveEditor
        v-if="showEditor && editorCurveType === 'channel'"
        curveType="channel"
        :component="curveComponent"
        @submit="handleCurveSubmit"
        @cancel="closeEditor"
      />
    </div>

    <!-- Switching Losses Tab -->
    <div v-show="activeTab === 'switching'" class="curve-tab-content">
      <h3>Switching Loss Data</h3>
      <div class="sub-tabs">
        <button
          v-for="lt in ['e_on', 'e_off', 'e_rr']"
          :key="lt"
          @click="activeLossType = lt"
          :class="['sub-tab-btn', { active: activeLossType === lt }]"
        >{{ lossTypeLabels[lt] }}</button>
      </div>

      <CurveList
        :curves="curves['switching_' + activeLossType] || []"
        :title="lossTypeLabels[activeLossType]"
        curveType="Switching Loss"
        @add="openCurveEditor('switching_loss')"
        @view="viewCurve('switching_loss', $event)"
        @delete="deleteCurve('switching_loss', $event)"
      />

      <CurveEditor
        v-if="showEditor && editorCurveType === 'switching_loss'"
        curveType="switching_loss"
        :lossType="activeLossType"
        @submit="handleCurveSubmit"
        @cancel="closeEditor"
      />
    </div>

    <!-- Gate Charge Tab -->
    <div v-show="activeTab === 'gate_charge'" class="curve-tab-content">
      <h3>Gate Charge Curves</h3>

      <CurveList
        :curves="curves.gate_charge || []"
        title="Gate Charge Curves"
        curveType="Gate Charge"
        @add="openCurveEditor('gate_charge')"
        @view="viewCurve('gate_charge', $event)"
        @delete="deleteCurve('gate_charge', $event)"
      />

      <CurveEditor
        v-if="showEditor && editorCurveType === 'gate_charge'"
        curveType="gate_charge"
        @submit="handleCurveSubmit"
        @cancel="closeEditor"
      />
    </div>

    <!-- SOA Tab -->
    <div v-show="activeTab === 'soa'" class="curve-tab-content">
      <h3>Safe Operating Area</h3>

      <CurveList
        :curves="curves.soa || []"
        title="SOA Curves"
        curveType="SOA"
        @add="openCurveEditor('soa')"
        @view="viewCurve('soa', $event)"
        @delete="deleteCurve('soa', $event)"
      />

      <CurveEditor
        v-if="showEditor && editorCurveType === 'soa'"
        curveType="soa"
        @submit="handleCurveSubmit"
        @cancel="closeEditor"
      />
    </div>

    <!-- Capacitance Tab -->
    <div v-show="activeTab === 'capacitance'" class="curve-tab-content">
      <h3>Voltage-Dependent Capacitance</h3>
      <div class="sub-tabs">
        <button
          v-for="ct in ['c_oss', 'c_iss', 'c_rss']"
          :key="ct"
          @click="activeCapType = ct"
          :class="['sub-tab-btn', { active: activeCapType === ct }]"
        >{{ capTypeLabels[ct] }}</button>
      </div>

      <CurveList
        :curves="curves['capacitance_' + activeCapType] || []"
        :title="capTypeLabels[activeCapType]"
        curveType="Capacitance"
        @add="openCurveEditor('capacitance')"
        @view="viewCurve('capacitance', $event)"
        @delete="deleteCurve('capacitance', $event)"
      />

      <CurveEditor
        v-if="showEditor && editorCurveType === 'capacitance'"
        curveType="capacitance"
        :capType="activeCapType"
        @submit="handleCurveSubmit"
        @cancel="closeEditor"
      />
    </div>

    <!-- Validation Results (shown on any tab) -->
    <div v-if="validationResult" class="validation-section">
      <h3>Validation Results</h3>
      <div v-if="validationResult.errors && validationResult.errors.length > 0" class="errors">
        <h4>Errors:</h4>
        <ul>
          <li v-for="error in validationResult.errors" :key="error">{{ error }}</li>
        </ul>
      </div>
      <div v-if="validationResult.warnings && validationResult.warnings.length > 0" class="warnings">
        <h4>Warnings:</h4>
        <ul>
          <li v-for="warning in validationResult.warnings" :key="warning">{{ warning }}</li>
        </ul>
      </div>
      <div v-if="(!validationResult.errors || validationResult.errors.length === 0)" class="success">
        Validation passed successfully
      </div>
    </div>

    <!-- Curve notification -->
    <div v-if="curveMessage" :class="['curve-notification', curveMessage.type]">
      {{ curveMessage.text }}
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { transistorApi } from '../services/api.js'
import CurveEditor from './CurveEditor.vue'
import CurveList from './CurveList.vue'

const props = defineProps({
  transistor: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['saved', 'cancel'])

const isEditing = ref(false)
const isSubmitting = ref(false)
const validationResult = ref(null)
const activeTab = ref('general')
const curveComponent = ref('switch')
const activeLossType = ref('e_on')
const activeCapType = ref('c_oss')
const showEditor = ref(false)
const editorCurveType = ref(null)
const curveMessage = ref(null)

const lossTypeLabels = { e_on: 'E_on (Turn-on)', e_off: 'E_off (Turn-off)', e_rr: 'E_rr (Reverse Recovery)' }
const capTypeLabels = { c_oss: 'C_oss (Output)', c_iss: 'C_iss (Input)', c_rss: 'C_rss (Reverse Transfer)' }

const formData = reactive({
  metadata: {
    name: '',
    type: '',
    manufacturer: '',
    housing_type: '',
    author: '',
    comment: ''
  },
  electrical: {
    v_abs_max: 0,
    i_abs_max: 0,
    i_cont: 0,
    t_j_max: 0
  },
  thermal: {
    r_th_cs: 0,
    housing_area: 0,
    cooling_area: 0
  }
})

const curves = reactive({
  switch_channel: [],
  diode_channel: [],
  switching_e_on: [],
  switching_e_off: [],
  switching_e_rr: [],
  gate_charge: [],
  soa: [],
  capacitance_c_oss: [],
  capacitance_c_iss: [],
  capacitance_c_rss: [],
})

const availableTabs = computed(() => {
  const tabs = [{ key: 'general', label: 'General Info' }]
  if (isEditing.value) {
    tabs.push(
      { key: 'channel', label: 'Channel Curves' },
      { key: 'switching', label: 'Switching Losses' },
      { key: 'gate_charge', label: 'Gate Charge' },
      { key: 'soa', label: 'SOA' },
      { key: 'capacitance', label: 'Capacitance' },
    )
  }
  return tabs
})

function getTransistorId() {
  if (!props.transistor) return null
  return props.transistor.metadata.name.replace(/ /g, '_').replace(/\//g, '_')
}

onMounted(async () => {
  if (props.transistor) {
    isEditing.value = true
    Object.assign(formData.metadata, props.transistor.metadata)
    Object.assign(formData.electrical, props.transistor.electrical)
    Object.assign(formData.thermal, props.transistor.thermal)
    await loadCurves()
  }
})

async function loadCurves() {
  const id = getTransistorId()
  if (!id) return
  try {
    const data = await transistorApi.getCurves(id)
    // Map backend response to local state
    if (data.switch) {
      curves.switch_channel = data.switch.channel || []
      curves.switching_e_on = data.switch.e_on || []
      curves.switching_e_off = data.switch.e_off || []
    }
    if (data.diode) {
      curves.diode_channel = data.diode.channel || []
      curves.switching_e_rr = data.diode.e_rr || []
    }
    curves.gate_charge = data.switch?.gate_charge || []
    curves.soa = data.switch?.soa || []
    if (data.capacitance) {
      curves.capacitance_c_oss = data.capacitance.c_oss || []
      curves.capacitance_c_iss = data.capacitance.c_iss || []
      curves.capacitance_c_rss = data.capacitance.c_rss || []
    }
  } catch (error) {
    console.error('Failed to load curves:', error)
  }
}

function openCurveEditor(type) {
  editorCurveType.value = type
  showEditor.value = true
}

function closeEditor() {
  showEditor.value = false
  editorCurveType.value = null
}

function showNotification(text, type = 'success') {
  curveMessage.value = { text, type }
  setTimeout(() => { curveMessage.value = null }, 3000)
}

async function handleCurveSubmit(curveData) {
  const id = getTransistorId()
  if (!id) return

  try {
    const type = editorCurveType.value
    if (type === 'channel') {
      await transistorApi.addChannelCurve(id, curveData, curveComponent.value)
    } else if (type === 'switching_loss') {
      await transistorApi.addSwitchingLossCurve(id, activeLossType.value, curveData)
    } else if (type === 'gate_charge') {
      await transistorApi.addGateChargeCurve(id, curveData)
    } else if (type === 'soa') {
      await transistorApi.addSOACurve(id, curveData)
    } else if (type === 'capacitance') {
      await transistorApi.addCapacitanceCurve(id, activeCapType.value, curveData)
    }
    showNotification('Curve added successfully')
    closeEditor()
    await loadCurves()
  } catch (error) {
    console.error('Failed to add curve:', error)
    showNotification('Failed to add curve: ' + error.message, 'error')
  }
}

async function deleteCurve(curveType, index) {
  const id = getTransistorId()
  if (!id) return

  let component, type
  if (curveType === 'channel') {
    component = curveComponent.value
    type = 'channel'
  } else if (curveType === 'switching_loss') {
    component = activeLossType.value === 'e_rr' ? 'diode' : 'switch'
    type = activeLossType.value
  } else if (curveType === 'gate_charge') {
    component = 'switch'
    type = 'gate_charge'
  } else if (curveType === 'soa') {
    component = 'switch'
    type = 'soa'
  } else if (curveType === 'capacitance') {
    component = 'capacitance'
    type = activeCapType.value
  }

  try {
    await transistorApi.deleteCurve(id, component, type, index)
    showNotification('Curve deleted')
    await loadCurves()
  } catch (error) {
    console.error('Failed to delete curve:', error)
    showNotification('Failed to delete curve: ' + error.message, 'error')
  }
}

function viewCurve(curveType, index) {
  // For now, selecting a curve in the list highlights it
  // CurvePlotter can be integrated here for detailed view
  console.log('View curve:', curveType, index)
}

async function validateForm() {
  try {
    if (isEditing.value) {
      const transistorId = getTransistorId()
      validationResult.value = await transistorApi.validate(transistorId)
    } else {
      const errors = []
      const warnings = []

      if (!formData.metadata.name.trim()) errors.push('Name is required')
      if (!formData.metadata.type) errors.push('Type is required')
      if (!formData.metadata.manufacturer.trim()) errors.push('Manufacturer is required')
      if (!formData.metadata.housing_type) errors.push('Housing type is required')

      if (formData.electrical.v_abs_max <= 0) errors.push('V_abs_max must be positive')
      if (formData.electrical.i_abs_max <= 0) errors.push('I_abs_max must be positive')
      if (formData.electrical.i_cont <= 0) errors.push('I_cont must be positive')
      if (formData.electrical.i_cont > formData.electrical.i_abs_max) {
        warnings.push('I_cont is greater than I_abs_max')
      }

      if (formData.thermal.r_th_cs <= 0) errors.push('R_th_cs must be positive')
      if (formData.thermal.housing_area <= 0) errors.push('Housing area must be positive')
      if (formData.thermal.cooling_area <= 0) errors.push('Cooling area must be positive')

      validationResult.value = { errors, warnings }
    }
  } catch (error) {
    console.error('Validation error:', error)
    validationResult.value = {
      errors: ['Validation failed: ' + error.message],
      warnings: []
    }
  }
}

async function handleSubmit() {
  isSubmitting.value = true

  try {
    if (isEditing.value) {
      const transistorId = getTransistorId()
      await transistorApi.update(transistorId, formData)
    } else {
      await transistorApi.create(formData)
    }

    emit('saved')
  } catch (error) {
    console.error('Save error:', error)
    alert('Failed to save transistor: ' + error.message)
  } finally {
    isSubmitting.value = false
  }
}
</script>

<style scoped>
.transistor-form {
  max-width: 1000px;
  margin: 0 auto;
  background: var(--bg-secondary, white);
  border-radius: 8px;
  box-shadow: 0 2px 8px var(--shadow, rgba(0,0,0,0.1));
  overflow: hidden;
}

.form-header {
  background: #3498db;
  color: white;
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.form-header h2 {
  margin: 0;
}

/* Section Tabs */
.section-tabs {
  display: flex;
  background: var(--bg-tertiary, #f8f9fa);
  border-bottom: 1px solid var(--border-color, #e1e5e9);
  overflow-x: auto;
}

.tab-btn {
  padding: 0.75rem 1.25rem;
  border: none;
  background: transparent;
  color: var(--text-secondary, #666);
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  border-bottom: 3px solid transparent;
  white-space: nowrap;
  transition: all 0.2s;
}

.tab-btn:hover {
  background: var(--bg-secondary, #fff);
  color: var(--text-primary, #333);
}

.tab-btn.active {
  color: #3498db;
  border-bottom-color: #3498db;
  background: var(--bg-secondary, #fff);
}

.form-content {
  padding: 2rem;
}

.form-section {
  margin-bottom: 2rem;
}

.form-section h3 {
  color: var(--text-primary, #2c3e50);
  margin: 0 0 1rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--border-color, #eee);
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.form-group {
  display: flex;
  flex-direction: column;
}

.form-group-wide {
  grid-column: 1 / -1;
}

.form-group label {
  font-weight: bold;
  margin-bottom: 0.5rem;
  color: var(--text-secondary, #555);
}

.form-group input,
.form-group select,
.form-group textarea {
  padding: 0.75rem;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  font-size: 1rem;
  transition: border-color 0.3s;
  background: var(--bg-secondary, #fff);
  color: var(--text-primary, #333);
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #3498db;
  box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
}

.form-group textarea {
  resize: vertical;
  min-height: 80px;
}

.form-actions {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  padding-top: 1rem;
  border-top: 1px solid var(--border-color, #eee);
}

/* Curve Tab Content */
.curve-tab-content {
  padding: 1.5rem 2rem;
}

.curve-tab-content h3 {
  color: var(--text-primary, #2c3e50);
  margin: 0 0 1rem 0;
}

.curve-tab-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.component-toggle {
  display: flex;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  overflow: hidden;
}

.toggle-btn {
  padding: 0.5rem 1rem;
  border: none;
  background: var(--bg-tertiary, #f5f5f5);
  color: var(--text-secondary, #666);
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}

.toggle-btn.active {
  background: #3498db;
  color: white;
}

.sub-tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.sub-tab-btn {
  padding: 0.5rem 1rem;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 20px;
  background: var(--bg-tertiary, #f5f5f5);
  color: var(--text-secondary, #666);
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}

.sub-tab-btn.active {
  background: #3498db;
  color: white;
  border-color: #3498db;
}

/* Validation and notification */
.validation-section {
  padding: 1rem 2rem 2rem;
  background: var(--bg-tertiary, #f9f9f9);
  border-top: 1px solid var(--border-color, #eee);
}

.validation-section h3 {
  color: var(--text-primary, #2c3e50);
  margin: 0 0 1rem 0;
}

.errors { color: #e74c3c; margin-bottom: 1rem; }
.warnings { color: #f39c12; margin-bottom: 1rem; }
.success { color: #27ae60; font-weight: bold; }
.errors h4, .warnings h4 { margin: 0 0 0.5rem 0; }
.errors ul, .warnings ul { margin: 0; padding-left: 1.5rem; }

.curve-notification {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 500;
  z-index: 1000;
  animation: slideIn 0.3s ease;
}

.curve-notification.success {
  background: #27ae60;
  color: white;
}

.curve-notification.error {
  background: #e74c3c;
  color: white;
}

@keyframes slideIn {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

/* Button styles */
.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  transition: background-color 0.3s;
}

.btn:disabled { opacity: 0.6; cursor: not-allowed; }
.btn-primary { background: #3498db; color: white; }
.btn-primary:hover:not(:disabled) { background: #2980b9; }
.btn-secondary { background: #95a5a6; color: white; }
.btn-secondary:hover { background: #7f8c8d; }

@media (max-width: 768px) {
  .form-grid { grid-template-columns: 1fr; }
  .form-actions { flex-direction: column; }
  .form-header { flex-direction: column; gap: 1rem; }
  .curve-tab-header { flex-direction: column; gap: 0.5rem; }
  .section-tabs { flex-wrap: wrap; }
}
</style>
