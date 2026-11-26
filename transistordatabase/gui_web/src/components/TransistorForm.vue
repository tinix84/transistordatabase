<template>
  <div class="transistor-form">
    <header class="form-header">
      <div class="device-heading">
        <h2>{{ isEditing ? 'Edit Transistor' : 'Create New Transistor' }}</h2>
        <p v-if="isEditing" class="part-number">{{ transistor.metadata.name }}</p>
      </div>
      <div class="form-actions">
        <button type="button" @click="$emit('cancel')" class="btn btn-secondary">Cancel</button>
        <template v-if="activeTab === 'general'">
          <button type="button" @click="validateForm" class="btn btn-secondary">
            {{ isEditing ? 'Validate saved record' : 'Validate' }}
          </button>
          <button type="submit" form="transistor-properties" class="btn btn-primary" :disabled="isSubmitting">
            {{ isSubmitting ? 'Saving...' : (isEditing ? 'Update' : 'Create') }}
          </button>
        </template>
      </div>
    </header>

    <div class="editor-layout" :class="{ 'has-curve-nav': isEditing }">
      <nav v-if="isEditing" class="section-tabs" aria-label="Device editor sections">
        <button
          v-for="tab in availableTabs"
          :key="tab.key"
          type="button"
          @click="activeTab = tab.key"
          :class="['tab-btn', { active: activeTab === tab.key }]"
          :aria-current="activeTab === tab.key ? 'true' : undefined"
        >
          {{ tab.label }}
        </button>
      </nav>

      <div class="editor-content">
        <p v-if="saveError" class="save-error" role="alert">{{ saveError }}</p>

        <div v-if="validationResult" class="validation-section" role="status">
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

        <!-- General Info Tab -->
        <form
          v-show="activeTab === 'general'"
          id="transistor-properties"
          class="form-content"
          @submit.prevent="handleSubmit"
        >
          <section class="form-section identity-section" aria-labelledby="device-identity">
            <h3 id="device-identity">Metadata</h3>
            <div class="form-grid">
              <div class="form-group">
                <label for="name">Name *</label>
                <input
                  id="name"
                  v-model="formData.metadata.name"
                  type="text"
                  :readonly="isEditing"
                  :aria-describedby="isEditing ? 'device-name-help' : undefined"
                  required
                  placeholder="e.g. IGBT_1200V_100A"
                />
                <small v-if="isEditing" id="device-name-help">Device names cannot be changed while editing.</small>
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
                  placeholder="e.g. Infineon"
                />
              </div>

              <div class="form-group">
                <label for="housing">Housing Type *</label>
                <select id="housing" v-model="formData.metadata.housing_type" required>
                  <option value="">Select housing</option>
                  <option
                    v-if="formData.metadata.housing_type && !housingTypeOptions.includes(formData.metadata.housing_type)"
                    :value="formData.metadata.housing_type"
                  >{{ formData.metadata.housing_type }}</option>
                  <option value="TO247">TO-247</option>
                  <option value="TO220">TO-220</option>
                  <option value="TO263">TO-263</option>
                  <option value="TO92">TO-92</option>
                </select>
              </div>

              <div class="form-group">
                <label for="author">Author</label>
                <input
                  id="author"
                  v-model="formData.metadata.author"
                  type="text"
                />
              </div>

              <div class="form-group form-group-wide">
                <label for="comment">Comment</label>
                <textarea
                  id="comment"
                  v-model="formData.metadata.comment"
                  rows="3"
                ></textarea>
              </div>
            </div>
          </section>

          <section class="form-section" aria-labelledby="device-ratings">
            <h3 id="device-ratings">Electrical Ratings</h3>
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
          </section>

          <section class="form-section" aria-labelledby="device-thermal">
            <h3 id="device-thermal">Thermal Properties</h3>
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
                <label for="r_th_switch_cs">R<sub>th,switch-cs</sub> (K/W) *</label>
                <input
                  id="r_th_switch_cs"
                  v-model.number="formData.thermal.r_th_switch_cs"
                  type="number"
                  step="0.001"
                  min="0"
                  required
                />
              </div>

              <div class="form-group">
                <label for="r_th_diode_cs">R<sub>th,diode-cs</sub> (K/W) *</label>
                <input
                  id="r_th_diode_cs"
                  v-model.number="formData.thermal.r_th_diode_cs"
                  type="number"
                  step="0.001"
                  min="0"
                  required
                />
              </div>

              <div class="form-group">
                <label for="housing_area">Housing Area (cm²) *</label>
                <input
                  id="housing_area"
                  v-model.number="formData.thermal.housing_area"
                  type="number"
                  step="any"
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
                  step="any"
                  min="0"
                  required
                  placeholder="e.g. 1.0"
                />
              </div>
            </div>
          </section>

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
            @edit="editCurve('channel', $event)"
            @view="viewCurve('channel', $event)"
            @delete="deleteCurve('channel', $event)"
          />
          <CurveEditor
            v-if="showEditor && editorCurveType === 'channel'"
            :key="curveEditorKey('channel')"
            curveType="channel"
            :component="editingCurve?.route.component || curveComponent"
            :edit-mode="Boolean(editingCurve)"
            :initial-data="editingCurve?.initialData || null"
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
            @edit="editCurve('switching_loss', $event)"
            @view="viewCurve('switching_loss', $event)"
            @delete="deleteCurve('switching_loss', $event)"
          />
          <CurveEditor
            v-if="showEditor && editorCurveType === 'switching_loss'"
            :key="curveEditorKey('switching_loss')"
            curveType="switching_loss"
            :lossType="editingCurve?.route.type || activeLossType"
            :edit-mode="Boolean(editingCurve)"
            :initial-data="editingCurve?.initialData || null"
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
            @edit="editCurve('gate_charge', $event)"
            @view="viewCurve('gate_charge', $event)"
            @delete="deleteCurve('gate_charge', $event)"
          />
          <CurveEditor
            v-if="showEditor && editorCurveType === 'gate_charge'"
            :key="curveEditorKey('gate_charge')"
            curveType="gate_charge"
            :edit-mode="Boolean(editingCurve)"
            :initial-data="editingCurve?.initialData || null"
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
            @edit="editCurve('soa', $event)"
            @view="viewCurve('soa', $event)"
            @delete="deleteCurve('soa', $event)"
          />
          <CurveEditor
            v-if="showEditor && editorCurveType === 'soa'"
            :key="curveEditorKey('soa')"
            curveType="soa"
            :edit-mode="Boolean(editingCurve)"
            :initial-data="editingCurve?.initialData || null"
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
            @edit="editCurve('capacitance', $event)"
            @view="viewCurve('capacitance', $event)"
            @delete="deleteCurve('capacitance', $event)"
          />
          <CurveEditor
            v-if="showEditor && editorCurveType === 'capacitance'"
            :key="curveEditorKey('capacitance')"
            curveType="capacitance"
            :capType="editingCurve?.route.type || activeCapType"
            :edit-mode="Boolean(editingCurve)"
            :initial-data="editingCurve?.initialData || null"
            @submit="handleCurveSubmit"
            @cancel="closeEditor"
          />
        </div>

        <div v-show="activeTab === 'thermal_curves'" class="curve-tab-content" data-testid="thermal-curve-tab">
          <h3>Transient Junction-to-Case Thermal Impedance</h3>
          <p>Stored samples are shown unchanged. Time is seconds; Zth(j-c) is K/W.</p>
          <div class="component-toggle">
            <button type="button" :disabled="Boolean(thermalDraft || thermalDeleteComponent)" @click="thermalComponent = 'switch'" :class="['toggle-btn', { active: thermalComponent === 'switch' }]">Switch</button>
            <button type="button" :disabled="Boolean(thermalDraft || thermalDeleteComponent)" @click="thermalComponent = 'diode'" :class="['toggle-btn', { active: thermalComponent === 'diode' }]">Diode</button>
          </div>

          <p v-if="curveLoadError" class="curve-viewer-empty" role="alert">{{ curveLoadError }}</p>
          <section v-else class="thermal-curve-card" :aria-label="`${thermalComponent} thermal curve`">
            <template v-if="thermalCurves[thermalComponent].graph_t_rthjc">
              <p>{{ thermalCurves[thermalComponent].graph_t_rthjc[0].length }} stored points</p>
              <div class="card-actions">
                <button type="button" class="btn btn-secondary" @click="viewThermalCurve">View</button>
                <button type="button" class="btn btn-secondary" @click="editThermalCurve">Edit</button>
                <button type="button" class="btn btn-danger" @click="startThermalDelete">Delete</button>
              </div>
            </template>
            <p v-else role="status">No transient thermal curve is stored for this {{ thermalComponent }}.</p>
            <button v-if="!thermalCurves[thermalComponent].graph_t_rthjc && !thermalDraft" type="button" class="btn btn-primary" @click="startThermalCurve">Add thermal curve</button>
          </section>

          <form v-if="thermalDraft" class="thermal-curve-editor" @submit.prevent="saveThermalCurve">
            <h4>{{ thermalCurves[thermalDraftComponent].graph_t_rthjc ? 'Edit' : 'Add' }} {{ thermalDraftComponent }} thermal curve</h4>
            <p>Enter time in seconds and thermal impedance in K/W. Time samples must increase.</p>
            <table>
              <thead><tr><th>Time (s)</th><th>Zth(j-c) (K/W)</th><th>Action</th></tr></thead>
              <tbody>
                <tr v-for="(time, index) in thermalDraft[0]" :key="index">
                  <td><input v-model.number="thermalDraft[0][index]" type="number" step="any" :aria-label="`Time seconds point ${index + 1}`" /></td>
                  <td><input v-model.number="thermalDraft[1][index]" type="number" step="any" :aria-label="`Zth K per W point ${index + 1}`" /></td>
                  <td><button type="button" class="btn btn-secondary" :disabled="thermalDraft[0].length <= 2" :aria-label="`Remove point ${index + 1}`" @click="removeThermalPoint(index)">Remove</button></td>
                </tr>
              </tbody>
            </table>
            <div class="form-actions">
              <button type="button" class="btn btn-secondary" @click="cancelThermalEdit">Cancel</button>
              <button type="button" class="btn btn-secondary" @click="addThermalPoint">Add point</button>
              <button type="submit" class="btn btn-primary">Save thermal curve</button>
            </div>
          </form>
          <div v-if="thermalDeleteComponent" class="modal-overlay" @click="cancelThermalDelete">
            <div class="modal-content" role="dialog" aria-modal="true" aria-labelledby="thermal-delete-title" @click.stop>
              <h4 id="thermal-delete-title">Delete {{ thermalDeleteComponent }} thermal curve?</h4>
              <p>This clears only the stored graph_t_rthjc samples.</p>
              <div class="modal-actions">
                <button type="button" class="btn btn-secondary" @click="cancelThermalDelete">Cancel</button>
                <button type="button" class="btn btn-danger" @click="deleteThermalCurve">Delete thermal curve</button>
              </div>
            </div>
          </div>
        </div>

        <div v-if="selectedCurve" class="curve-viewer" data-testid="curve-viewer" role="region" aria-label="Curve detail">
          <div class="curve-viewer-header">
            <h4>{{ selectedCurve.title }}</h4>
            <button type="button" class="btn btn-secondary" aria-label="Close curve view" @click="selectedCurve = null">
              Close
            </button>
          </div>
          <CurvePlotter
            v-if="selectedCurvePlot"
            :title="selectedCurve.title"
            :datasets="[selectedCurvePlot.dataset]"
            :x-label="selectedCurvePlot.xLabel"
            :x-value-scientific="selectedCurvePlot.xValueScientific || false"
            :y-label="selectedCurvePlot.yLabel"
            :y-value-scientific="selectedCurvePlot.yValueScientific || false"
            :x-log-scale="selectedCurvePlot.xLogScale"
            :y-log-scale="selectedCurvePlot.yLogScale"
          />
          <p v-else class="curve-viewer-empty" role="status">This curve has no supported graph data to plot.</p>
        </div>
      </div>
    </div>

    <!-- Curve notification -->
    <div v-if="curveMessage" :class="['curve-notification', curveMessage.type]" role="status">
      {{ curveMessage.text }}
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { transistorApi } from '../services/api.js'
import CurveEditor from './CurveEditor.vue'
import CurveList from './CurveList.vue'
import CurvePlotter from './CurvePlotter.vue'

const props = defineProps({
  transistor: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['saved', 'cancel'])

const isEditing = ref(false)
const isSubmitting = ref(false)
const saveError = ref('')
const validationResult = ref(null)
const activeTab = ref('general')
const curveComponent = ref('switch')
const activeLossType = ref('e_on')
const activeCapType = ref('c_oss')
const thermalComponent = ref('switch')
const thermalDraft = ref(null)
const thermalDraftComponent = ref(null)
const thermalDeleteComponent = ref(null)
const curveLoadError = ref('')
const showEditor = ref(false)
const editorCurveType = ref(null)
const editingCurve = ref(null)
const curveMessage = ref(null)
const selectedCurve = ref(null)

const lossTypeLabels = { e_on: 'E_on (Turn-on)', e_off: 'E_off (Turn-off)', e_rr: 'E_rr (Reverse Recovery)' }
const capTypeLabels = { c_oss: 'C_oss (Output)', c_iss: 'C_iss (Input)', c_rss: 'C_rss (Reverse Transfer)' }
const housingTypeOptions = ['TO247', 'TO220', 'TO263', 'TO92']

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
    r_th_switch_cs: 0,
    r_th_diode_cs: 0,
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
const thermalCurves = reactive({
  switch: { graph_t_rthjc: null, _edit_token: '' },
  diode: { graph_t_rthjc: null, _edit_token: '' }
})

const curveWorkflow = {
  channel: {
    series: () => curveComponent.value === 'switch' ? curves.switch_channel : curves.diode_channel,
    title: () => `${curveComponent.value === 'switch' ? 'Switch' : 'Diode'} channel`,
    plot: curve => ({ graph: curve?.graph_v_i, xLabel: 'Voltage (V)', yLabel: 'Current (A)' }),
    add: (id, curve) => transistorApi.addChannelCurve(id, curve, curveComponent.value),
    curveRoute: () => ({ component: curveComponent.value, type: 'channel' })
  },
  switching_loss: {
    series: () => curves[`switching_${activeLossType.value}`],
    title: () => lossTypeLabels[activeLossType.value],
    plot: curve => {
      if (curve?.graph_i_e) return { graph: curve.graph_i_e, xLabel: 'Current (A)', yLabel: 'Switching energy (J)' }
      if (curve?.graph_r_e) return { graph: curve.graph_r_e, xLabel: 'Gate resistance (Ω)', yLabel: 'Switching energy (J)' }
      return { graph: null, xLabel: 'Current (A)', yLabel: 'Switching energy (J)' }
    },
    add: (id, curve) => transistorApi.addSwitchingLossCurve(id, activeLossType.value, curve),
    curveRoute: () => ({
      component: activeLossType.value === 'e_rr' ? 'diode' : 'switch',
      type: activeLossType.value
    })
  },
  gate_charge: {
    series: () => curves.gate_charge,
    title: () => 'Gate charge',
    plot: curve => ({
      graph: curve?.graph_q_v,
      // Stored graph_q_v charges are in C; present them in nC for the plot axis.
      xScale: 1e9,
      xLabel: 'Gate charge (nC)',
      yLabel: 'Gate voltage (V)'
    }),
    add: (id, curve) => transistorApi.addGateChargeCurve(id, curve),
    curveRoute: () => ({ component: 'switch', type: 'gate_charge' })
  },
  soa: {
    series: () => curves.soa,
    title: () => 'SOA',
    plot: curve => ({ graph: curve?.graph_i_v, xLabel: 'Voltage (V)', yLabel: 'Current (A)' }),
    add: (id, curve) => transistorApi.addSOACurve(id, curve),
    curveRoute: () => ({ component: 'switch', type: 'soa' })
  },
  capacitance: {
    series: () => curves[`capacitance_${activeCapType.value}`],
    title: () => capTypeLabels[activeCapType.value],
    plot: curve => ({ graph: curve?.graph_v_c, xLabel: 'Voltage (V)', yLabel: 'Capacitance (F)' }),
    add: (id, curve) => transistorApi.addCapacitanceCurve(id, activeCapType.value, curve),
    curveRoute: () => ({ component: 'capacitance', type: activeCapType.value })
  }
}

const availableTabs = computed(() => {
  const tabs = [{ key: 'general', label: 'General Info' }]
  if (isEditing.value) {
    tabs.push(
      { key: 'channel', label: 'Channel Curves' },
      { key: 'switching', label: 'Switching Losses' },
      { key: 'gate_charge', label: 'Gate Charge' },
      { key: 'soa', label: 'SOA' },
      { key: 'capacitance', label: 'Capacitance' },
      { key: 'thermal_curves', label: 'Thermal Curves' },
    )
  }
  return tabs
})

function getTransistorId() {
  if (!props.transistor) return null
  return props.transistor.metadata.name
}

onMounted(async () => {
  if (props.transistor) {
    isEditing.value = true
    Object.assign(formData.metadata, props.transistor.metadata)
    Object.assign(formData.electrical, props.transistor.electrical)
    Object.assign(formData.thermal, props.transistor.thermal, {
      housing_area: squareMetresToSquareCentimetres(props.transistor.thermal.housing_area),
      cooling_area: squareMetresToSquareCentimetres(props.transistor.thermal.cooling_area)
    })
    await loadCurves()
  }
})

async function loadCurves() {
  const id = getTransistorId()
  if (!id) return
  try {
    const data = await transistorApi.getCurves(id)
    curveLoadError.value = ''
    curves.switch_channel = data.switch?.channel_data || []
    curves.switching_e_on = data.switch?.e_on_data || []
    curves.switching_e_off = data.switch?.e_off_data || []
    curves.diode_channel = data.diode?.channel_data || []
    curves.switching_e_rr = data.diode?.e_rr_data || []
    curves.gate_charge = data.switch?.gate_charge_curves || []
    curves.soa = data.switch?.soa || []
    curves.capacitance_c_oss = data.capacitances?.c_oss || []
    curves.capacitance_c_iss = data.capacitances?.c_iss || []
    curves.capacitance_c_rss = data.capacitances?.c_rss || []
    for (const component of ['switch', 'diode']) {
      thermalCurves[component] = data.thermal_foster?.[component] || { graph_t_rthjc: null, _edit_token: '' }
    }
  } catch (error) {
    console.error('Failed to load curves:', error)
    curveLoadError.value = 'Unable to load saved curves. Reload the device and try again.'
  }
}

function startThermalCurve() {
  thermalDraftComponent.value = thermalComponent.value
  thermalDraft.value = [[0, 1], [0, 0]]
}

function editThermalCurve() {
  const graph = thermalCurves[thermalComponent.value].graph_t_rthjc
  if (!Array.isArray(graph) || graph.length !== 2) return
  thermalDraftComponent.value = thermalComponent.value
  thermalDraft.value = graph.map(row => [...row])
}

function cancelThermalEdit() {
  thermalDraft.value = null
  thermalDraftComponent.value = null
}

function startThermalDelete() {
  thermalDeleteComponent.value = thermalComponent.value
}

function cancelThermalDelete() {
  thermalDeleteComponent.value = null
}

function addThermalPoint() {
  if (!thermalDraft.value) return
  const times = thermalDraft.value[0]
  const lastTime = times[times.length - 1]
  thermalDraft.value[0].push(lastTime + Math.max(Math.abs(lastTime) * 0.1, 1))
  thermalDraft.value[1].push(thermalDraft.value[1][thermalDraft.value[1].length - 1])
}

function removeThermalPoint(index) {
  if (!thermalDraft.value || thermalDraft.value[0].length <= 2) return
  thermalDraft.value[0].splice(index, 1)
  thermalDraft.value[1].splice(index, 1)
}

function viewThermalCurve() {
  const graph = thermalCurves[thermalComponent.value].graph_t_rthjc
  const points = graphToPoints(graph)
  selectedCurve.value = {
    title: `${thermalComponent.value} transient Zth(j-c)`,
    plot: points.length ? {
      points,
      xLabel: 'Time (s)',
      yLabel: 'Zth(j-c) (K/W)',
      xValueScientific: true,
      yValueScientific: true,
      xLogScale: false,
      yLogScale: false
    } : null
  }
}

async function saveThermalCurve() {
  const id = getTransistorId()
  if (!id || !thermalDraft.value) return
  const component = thermalDraftComponent.value
  if (!component) return
  const existing = thermalCurves[component]
  const request = {
    expected_token: existing._edit_token,
    graph_t_rthjc: thermalDraft.value.map(row => [...row])
  }
  try {
    if (existing.graph_t_rthjc) await transistorApi.updateThermalCurve(id, component, request)
    else await transistorApi.addThermalCurve(id, component, request)
    cancelThermalEdit()
    showNotification('Thermal curve saved successfully')
    await loadCurves()
  } catch (error) {
    const detail = error.response?.data?.detail
    showNotification('Failed to save thermal curve: ' + (typeof detail === 'string' ? detail : error.message), 'error')
    if (error.response?.status === 409) {
      cancelThermalEdit()
      await loadCurves()
    }
  }
}

async function deleteThermalCurve() {
  const id = getTransistorId()
  const component = thermalDeleteComponent.value
  if (!id || !component) return
  try {
    await transistorApi.deleteThermalCurve(id, component, thermalCurves[component]._edit_token)
    cancelThermalDelete()
    selectedCurve.value = null
    showNotification('Thermal curve deleted')
    await loadCurves()
  } catch (error) {
    const detail = error.response?.data?.detail
    showNotification('Failed to delete thermal curve: ' + (typeof detail === 'string' ? detail : error.message), 'error')
    if (error.response?.status === 409) {
      cancelThermalDelete()
      await loadCurves()
    }
  }
}

function openCurveEditor(type) {
  editingCurve.value = null
  editorCurveType.value = type
  showEditor.value = true
}

function editCurve(type, index) {
  const workflow = curveWorkflow[type]
  const series = workflow?.series()
  const curve = series?.[index]
  const route = workflow?.curveRoute()
  if (!curve || !series || !route || typeof curve._edit_token !== 'string') {
    showNotification('This curve is stale or cannot be edited. Reload the device and try again.', 'error')
    return
  }
  editingCurve.value = {
    index,
    expectedCount: series.length,
    expectedToken: curve._edit_token,
    route,
    initialData: { ...curve }
  }
  editorCurveType.value = type
  showEditor.value = true
}

function curveEditorKey(curveType) {
  const selection = editingCurve.value
  const route = selection?.route || curveWorkflow[curveType]?.curveRoute()
  const mode = selection
    ? `edit:${selection.index}:${selection.expectedToken}`
    : 'add'
  return `${curveType}:${route?.component || ''}:${route?.type || ''}:${mode}`
}

function closeEditor() {
  showEditor.value = false
  editorCurveType.value = null
  editingCurve.value = null
}

function showNotification(text, type = 'success') {
  curveMessage.value = { text, type }
  setTimeout(() => { curveMessage.value = null }, 3000)
}

async function handleCurveSubmit(curveData) {
  const id = getTransistorId()
  if (!id) return

  try {
    const workflow = curveWorkflow[editorCurveType.value]
    if (!workflow) return
    if (editingCurve.value) {
      const route = editingCurve.value.route
      await transistorApi.updateCurve(id, route.component, route.type, editingCurve.value.index, {
        expected_count: editingCurve.value.expectedCount,
        expected_token: editingCurve.value.expectedToken,
        curve: curveData
      })
      showNotification('Curve updated successfully')
    } else {
      await workflow.add(id, curveData)
      showNotification('Curve added successfully')
    }
    selectedCurve.value = null
    closeEditor()
    await loadCurves()
  } catch (error) {
    console.error(editingCurve.value ? 'Failed to update curve:' : 'Failed to add curve:', error)
    const detail = error.response?.data?.detail
    showNotification('Failed to save curve: ' + (typeof detail === 'string' ? detail : error.message), 'error')
  }
}

async function deleteCurve(curveType, index) {
  const id = getTransistorId()
  if (!id) return

  const route = curveWorkflow[curveType]?.curveRoute()
  if (!route) return

  try {
    await transistorApi.deleteCurve(id, route.component, route.type, index)
    showNotification('Curve deleted')
    selectedCurve.value = null
    await loadCurves()
  } catch (error) {
    console.error('Failed to delete curve:', error)
    showNotification('Failed to delete curve: ' + error.message, 'error')
  }
}

function viewCurve(curveType, index) {
  let xLogScale = false
  let yLogScale = false
  const workflow = curveWorkflow[curveType]
  const curve = workflow?.series()?.[index]
  if (!curve) return

  const plot = workflow.plot(curve)
  const points = graphToPoints(plot.graph, plot.xScale)
  selectedCurve.value = {
    title: `${workflow.title(curve)} curve #${index + 1}`,
    plot: points.length > 0 ? { points, xLabel: plot.xLabel, yLabel: plot.yLabel, xLogScale, yLogScale } : null
  }
}

const selectedCurvePlot = computed(() => {
  if (!selectedCurve.value?.plot) return null
  return {
    ...selectedCurve.value.plot,
    dataset: {
      label: selectedCurve.value.title,
      data: selectedCurve.value.plot.points
    }
  }
})

function graphToPoints(graph, xScale = 1) {
  if (!Array.isArray(graph) || graph.length !== 2 || !Array.isArray(graph[0]) || !Array.isArray(graph[1])) {
    return []
  }
  if (graph[0].length !== graph[1].length) return []
  if (graph[0].some(value => !Number.isFinite(value)) || graph[1].some(value => !Number.isFinite(value))) return []
  return graph[0].map((x, index) => ({ x: x * xScale, y: graph[1][index] }))
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
      if (formData.thermal.r_th_switch_cs < 0) errors.push('Switch case-to-sink resistance cannot be negative')
      if (formData.thermal.r_th_diode_cs < 0) errors.push('Diode case-to-sink resistance cannot be negative')
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

function squareMetresToSquareCentimetres(area) {
  return Number((area * 10000).toPrecision(12))
}

function squareCentimetresToSquareMetres(area) {
  return Number((area / 10000).toPrecision(12))
}

function toApiPayload() {
  return {
    metadata: { ...formData.metadata },
    electrical: { ...formData.electrical },
    thermal: {
      ...formData.thermal,
      housing_area: squareCentimetresToSquareMetres(formData.thermal.housing_area),
      cooling_area: squareCentimetresToSquareMetres(formData.thermal.cooling_area)
    }
  }
}

async function handleSubmit() {
  isSubmitting.value = true
  saveError.value = ''

  try {
    if (isEditing.value) {
      const transistorId = getTransistorId()
      await transistorApi.update(transistorId, toApiPayload())
    } else {
      await transistorApi.create(toApiPayload())
    }

    emit('saved')
  } catch (error) {
    console.error('Save error:', error)
    saveError.value = error.response?.data?.detail || error.message || 'Failed to save transistor.'
  } finally {
    isSubmitting.value = false
  }
}
</script>

<style scoped>
.transistor-form {
  --editor-padding: clamp(.75rem, 2vw, 1.3rem);
  width: min(100%, 1440px);
  margin-inline: auto;
  padding: var(--editor-padding);
  color: var(--text-primary);
  font-size: .875rem;
}

.form-header,
.form-actions,
.curve-tab-header,
.curve-viewer-header {
  display: flex;
  align-items: center;
  gap: .55rem;
}

.form-header {
  justify-content: space-between;
  flex-wrap: wrap;
  padding-bottom: .8rem;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: .85rem;
}

.device-heading { min-width: 0; }
.form-header h2 { font-size: 1.4rem; line-height: 1.25; letter-spacing: -.015em; }
.part-number {
  color: var(--text-secondary);
  font-family: ui-monospace, monospace;
  font-size: .8rem;
  overflow-wrap: anywhere;
}

.form-actions { flex-wrap: wrap; }
.transistor-form .btn {
  min-height: 2rem;
  padding: .25rem .65rem;
  font: inherit;
  font-weight: 550;
}
.transistor-form .btn-primary { background: var(--action-bg); border-color: var(--action-bg); }
.transistor-form .btn-primary:hover:not(:disabled) { background: var(--action-hover); }
.transistor-form .btn-secondary:hover:not(:disabled) { border-color: var(--accent-blue); }

.editor-content { min-width: 0; }
.has-curve-nav {
  display: grid;
  grid-template-columns: 10.75rem minmax(0, 1fr);
  gap: .8rem;
  align-items: start;
}

.section-tabs {
  display: grid;
  gap: 2px;
  padding: .3rem;
  border: 1px solid var(--border-color);
  border-radius: 5px;
  background: var(--bg-secondary);
}

.tab-btn, .toggle-btn, .sub-tab-btn {
  padding: .37rem .55rem;
  border: 1px solid transparent;
  border-radius: 3px;
  background: transparent;
  color: var(--text-secondary);
  font: inherit;
  text-align: left;
  white-space: nowrap;
}

.tab-btn:hover, .toggle-btn:hover, .sub-tab-btn:hover,
.tab-btn.active, .toggle-btn.active, .sub-tab-btn.active {
  background: var(--accent-blue-light);
  color: var(--accent-blue);
}

.tab-btn.active { border-inline-start: 2px solid var(--accent-blue); font-weight: 650; }

.form-content {
  display: grid;
  grid-template-columns: minmax(0, 4fr) minmax(0, 5fr);
  grid-template-rows: auto 1fr;
  align-items: start;
  gap: .65rem;
}

.form-section, .curve-tab-content, .curve-viewer, .validation-section, .save-error {
  min-width: 0;
  border: 1px solid var(--border-color);
  border-radius: 5px;
  background: var(--bg-secondary);
}

.identity-section { grid-row: 1 / span 2; }

.form-section h3 {
  padding: .47rem .75rem;
  border-bottom: 1px solid var(--border-color);
  border-radius: 5px 5px 0 0;
  background: var(--bg-tertiary);
  font-size: .8rem;
  letter-spacing: .025em;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .7rem;
  padding: .75rem;
}

.form-group { display: grid; align-content: start; gap: .23rem; min-width: 0; }
.form-group-wide { grid-column: 1 / -1; }
.form-group label { font-size: .79rem; font-weight: 600; }
.form-group label sub { line-height: 0; }
.form-group small { color: var(--text-secondary); line-height: 1.35; }

.form-group input, .form-group select, .form-group textarea {
  width: 100%;
  min-width: 0;
  min-height: 2.1rem;
  padding: .28rem .5rem;
  font: inherit;
}

.form-group input[type="number"], #name { font-family: ui-monospace, monospace; }
.form-group input[readonly] { background: var(--bg-tertiary); }
.form-group textarea { resize: vertical; }

.transistor-form :is(button, input, select, textarea):focus-visible {
  outline: 2px solid var(--accent-blue);
  outline-offset: 1px;
}

.transistor-form button:focus:not(:focus-visible) { outline: none; }

.curve-tab-content, .curve-viewer, .validation-section, .save-error { padding: .75rem; }
.curve-tab-content > h3, .curve-tab-header { margin-bottom: .65rem; }
.curve-tab-content h3, .validation-section h3 { font-size: 1rem; }
.curve-tab-header, .curve-viewer-header { justify-content: space-between; flex-wrap: wrap; }
.curve-viewer { margin-top: .65rem; }
.curve-viewer-header { margin-bottom: .65rem; }
.curve-viewer-empty { color: var(--text-secondary); }

.sub-tabs, .component-toggle {
  display: flex;
  flex-wrap: wrap;
  gap: .25rem;
  margin-bottom: .55rem;
}

.curve-tab-header .component-toggle { margin-bottom: 0; }

.toggle-btn, .sub-tab-btn {
  border-color: var(--border-color);
  background: var(--bg-tertiary);
  white-space: normal;
}

.thermal-curve-card, .thermal-curve-editor {
  display: grid;
  justify-items: start;
  gap: .55rem;
  padding: .65rem 0;
}

.thermal-curve-editor table {
  display: block;
  max-width: 100%;
  overflow-x: auto;
}

.thermal-curve-editor input { width: 100%; min-width: 4rem; }

.modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 600;
  display: grid;
  place-items: center;
  padding: 1rem;
  background: rgb(15 23 42 / 48%);
}

.modal-content {
  padding: 1rem;
  border: 1px solid var(--border-color);
  border-radius: 5px;
  background: var(--bg-secondary);
}

.modal-actions { display: flex; justify-content: flex-end; flex-wrap: wrap; gap: .5rem; margin-top: .7rem; }
.card-actions { display: flex; flex-wrap: wrap; gap: .4rem; }

.validation-section, .save-error { margin-bottom: .65rem; }
.save-error, .validation-section .errors { color: var(--text-primary); }
.save-error { border-inline-start: 3px solid var(--danger-color); }
.validation-section ul { padding-inline-start: 1.2rem; }
.validation-section h4 { margin-top: .3rem; }
.validation-section .success { font-weight: 600; }

.curve-notification {
  position: fixed;
  inset: auto var(--editor-padding) var(--editor-padding) auto;
  max-width: calc(100% - 2 * var(--editor-padding));
  padding: .55rem .75rem;
  border: 1px solid var(--border-color);
  border-inline-start: 3px solid var(--accent-blue);
  border-radius: 5px;
  background: var(--bg-secondary);
  box-shadow: 0 2px 6px var(--shadow);
  overflow-wrap: anywhere;
  z-index: 550;
}
.curve-notification.error { border-inline-start-color: var(--danger-color); }

@media (max-width: 900px) {
  .has-curve-nav { grid-template-columns: minmax(0, 1fr); }
  .section-tabs { display: flex; overflow-x: auto; }
}

@media (max-width: 680px) {
  .form-content { grid-template-columns: minmax(0, 1fr); grid-template-rows: none; }
  .identity-section { grid-row: auto; }
}

@media (max-width: 450px) {
  .form-grid { grid-template-columns: minmax(0, 1fr); }
  .form-header { align-items: flex-start; }
  .form-actions .btn, .tab-btn { min-height: 2.45rem; }
}
</style>
