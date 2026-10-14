<template>
  <div class="dpt-manager">
    <h2>Double Pulse Test (DPT) Manager</h2>
    <p class="description">
      Generate LTspice netlists for double pulse testing and validate measurement data
      against datasheet specifications.
    </p>

    <!-- Transistor Selector -->
    <div class="transistor-selector">
      <label>Select Transistor:</label>
      <select v-model="selectedTransistorId" @change="loadDPTData">
        <option value="">-- Choose a transistor --</option>
        <option v-for="t in transistorList" :key="t.id" :value="t.id">
          {{ t.name }}
        </option>
      </select>
    </div>

    <div v-if="selectedTransistorId" class="dpt-content">
      <!-- Tabs -->
      <div class="tab-nav">
        <button
          :class="{ active: activeTab === 'data' }"
          @click="activeTab = 'data'"
          class="tab-btn"
        >
          📊 DPT Data
        </button>
        <button
          :class="{ active: activeTab === 'netlist' }"
          @click="activeTab = 'netlist'"
          class="tab-btn"
        >
          📝 Generate Netlist
        </button>
        <button
          :class="{ active: activeTab === 'validate' }"
          @click="activeTab = 'validate'"
          class="tab-btn"
        >
          ✅ Validate Data
        </button>
      </div>

      <!-- DPT Data Tab -->
      <div v-if="activeTab === 'data'" class="tab-panel">
        <h3>Measurement Data Overview</h3>

        <div v-if="dptData" class="data-overview">
          <div class="status-grid">
            <div class="status-card" :class="{ available: dptData.has_e_on }">
              <div class="status-icon">{{ dptData.has_e_on ? '✅' : '❌' }}</div>
              <div class="status-label">Turn-On Energy (E_on)</div>
              <div class="status-text">
                {{ dptData.has_e_on ? 'Available' : 'Not available' }}
              </div>
            </div>

            <div class="status-card" :class="{ available: dptData.has_e_off }">
              <div class="status-icon">{{ dptData.has_e_off ? '✅' : '❌' }}</div>
              <div class="status-label">Turn-Off Energy (E_off)</div>
              <div class="status-text">
                {{ dptData.has_e_off ? 'Available' : 'Not available' }}
              </div>
            </div>

            <div class="status-card" :class="{ available: dptData.has_e_rr }">
              <div class="status-icon">{{ dptData.has_e_rr ? '✅' : '❌' }}</div>
              <div class="status-label">Reverse Recovery (E_rr)</div>
              <div class="status-text">
                {{ dptData.has_e_rr ? 'Available' : 'Not available' }}
              </div>
            </div>
          </div>

          <!-- Metadata -->
          <div v-if="dptData.metadata" class="metadata-section">
            <h4>Measurement Metadata</h4>
            <div class="metadata-grid">
              <div class="metadata-item">
                <span class="metadata-label">Junction Temperature:</span>
                <span class="metadata-value">{{ dptData.metadata.t_j }}°C</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Supply Voltage:</span>
                <span class="metadata-value">{{ dptData.metadata.v_supply }}V</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Gate Voltage:</span>
                <span class="metadata-value">{{ dptData.metadata.v_g }}V</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Gate Resistance:</span>
                <span class="metadata-value">{{ dptData.metadata.r_g }}Ω</span>
              </div>
            </div>
            <div v-if="dptData.metadata.comment" class="metadata-comment">
              <strong>Comment:</strong> {{ dptData.metadata.comment }}
            </div>
          </div>

          <section class="raw-waveforms-section" aria-labelledby="raw-waveforms-title">
            <h4 id="raw-waveforms-title">Stored Raw DPT Waveforms</h4>
            <p class="waveform-description">
              These plots show stored samples only. They do not calculate switching energy.
            </p>

            <div v-if="rawMeasurementRecords.length === 0" class="no-data" role="status">
              No raw DPT waveform measurements are stored for this transistor.
            </div>

            <template v-else>
              <div class="measurement-filter">
                <label for="dpt-measurement-search">Filter stored measurements</label>
                <div class="measurement-filter-controls">
                  <input
                    id="dpt-measurement-search"
                    v-model="measurementSearch"
                    data-testid="dpt-measurement-search"
                    type="search"
                    placeholder="Search conditions, test bench, comment…"
                  />
                  <button
                    v-if="measurementSearch"
                    class="btn-clear-measurement-search"
                    data-testid="dpt-measurement-search-clear"
                    type="button"
                    @click="measurementSearch = ''"
                  >
                    Clear
                  </button>
                </div>
                <p class="measurement-filter-count" aria-live="polite">
                  Showing {{ filteredMeasurementRecords.length }} of
                  {{ rawMeasurementRecords.length }} stored measurements.
                </p>
                <p
                  v-if="filteredMeasurementRecords.length === 0"
                  class="no-data"
                  data-testid="dpt-measurement-search-empty"
                  role="status"
                >
                  No stored DPT measurements match this search. Clear the search to view all records.
                </p>
              </div>

              <div v-if="!comparisonMode" class="waveform-selectors">
                <label
                  v-if="filteredMeasurementRecords.length"
                  for="dpt-measurement-select"
                >
                  Measurement record:
                </label>
                <select
                  v-if="filteredMeasurementRecords.length"
                  id="dpt-measurement-select"
                  data-testid="dpt-measurement-select"
                  v-model.number="selectedMeasurementIndex"
                  @change="selectedAcquisitionIndex = 0"
                >
                  <option
                    v-for="record in filteredMeasurementRecords"
                    :key="record.index"
                    :value="record.index"
                  >
                    {{ measurementLabel(record.measurement, record.index) }}
                  </option>
                </select>
                <button
                  v-if="rawMeasurementRecords.length > 1"
                  class="btn-compare-waveforms"
                  data-testid="dpt-compare-toggle"
                  type="button"
                  @click="toggleComparisonMode"
                >
                  Compare measurements ({{ filteredMeasurementRecords.length }} available)
                </button>
              </div>

              <div v-if="comparisonMode" class="dpt-comparison-controls">
                <div class="comparison-record-selector">
                  <div class="comparison-selector-heading">
                    <h5>Choose measurements to compare</h5>
                    <button
                      class="btn-compare-waveforms"
                      data-testid="dpt-compare-toggle"
                      type="button"
                      @click="toggleComparisonMode"
                    >
                      Return to one measurement
                    </button>
                  </div>
                  <label
                    v-for="record in filteredMeasurementRecords"
                    :key="record.index"
                    class="comparison-record-option"
                  >
                    <input
                      type="checkbox"
                      :data-testid="`dpt-compare-record-${record.index}`"
                      :checked="selectedComparisonIndices.includes(record.index)"
                      @change="toggleComparisonRecord(record.index)"
                    />
                    <span>{{ measurementLabel(record.measurement, record.index) }}</span>
                  </label>
                  <p
                    v-if="hiddenSelectedComparisonRecords.length"
                    class="comparison-filter-note"
                    data-testid="dpt-hidden-comparison-note"
                    role="status"
                  >
                    {{ hiddenSelectedComparisonRecords.length }} selected measurement<span
                      v-if="hiddenSelectedComparisonRecords.length !== 1"
                    >s</span>
                    hidden by this search stay in the comparison. Clear the search to change their selection.
                  </p>
                </div>

                <div
                  v-if="comparisonRecords.length < 2"
                  class="no-data"
                  data-testid="dpt-comparison-selection-status"
                  role="status"
                >
                  Select at least two stored measurements to compare.
                </div>

                <div v-else class="comparison-record-details">
                  <section
                    v-for="record in comparisonRecords"
                    :key="record.index"
                    class="comparison-record-card"
                    :data-testid="`dpt-comparison-record-${record.index}`"
                  >
                    <div class="comparison-record-heading">
                      <span
                        class="comparison-record-swatch"
                        :style="{ backgroundColor: comparisonColor(record.index) }"
                        aria-hidden="true"
                      ></span>
                      <h5>{{ record.label }}</h5>
                    </div>
                    <div v-if="record.acquisitionCount > 1" class="acquisition-selector">
                      <label :for="`dpt-comparison-acquisition-${record.index}`">
                        Acquisition for {{ record.label }}:
                      </label>
                      <select
                        :id="`dpt-comparison-acquisition-${record.index}`"
                        :data-testid="`dpt-comparison-acquisition-${record.index}`"
                        v-model.number="comparisonAcquisitionIndices[record.index]"
                      >
                        <option
                          v-for="acquisitionIndex in record.acquisitionCount"
                          :key="acquisitionIndex - 1"
                          :value="acquisitionIndex - 1"
                        >
                          Acquisition {{ acquisitionIndex }}
                        </option>
                      </select>
                    </div>
                    <p v-else-if="record.acquisitionCount === 1" class="single-acquisition-note">
                      Acquisition 1
                    </p>
                    <p v-else class="no-data" role="status">
                      No acquisitions are available for this measurement.
                    </p>
                    <dl v-if="record.conditions.length" class="comparison-conditions">
                      <div v-for="condition in record.conditions" :key="condition.label">
                        <dt>{{ condition.label }}</dt>
                        <dd>{{ condition.value }}</dd>
                      </div>
                    </dl>
                  </section>
                </div>
              </div>

              <div
                v-if="!comparisonMode && selectedMeasurementVisible"
                class="raw-measurement-details"
              >
                <div v-if="measurementConditions.length" class="measurement-conditions">
                  <h5>Test conditions</h5>
                  <dl>
                    <div v-for="condition in measurementConditions" :key="condition.label">
                      <dt>{{ condition.label }}</dt>
                      <dd>{{ condition.value }}</dd>
                    </div>
                  </dl>
                </div>

                <div v-if="acquisitionCount > 0" class="acquisition-selector">
                  <label for="dpt-acquisition-select">Acquisition:</label>
                  <select
                    id="dpt-acquisition-select"
                    data-testid="dpt-acquisition-select"
                    v-model.number="selectedAcquisitionIndex"
                  >
                    <option
                      v-for="index in acquisitionCount"
                      :key="index - 1"
                      :value="index - 1"
                    >
                      Acquisition {{ index }}
                    </option>
                  </select>
                </div>
                <div v-else class="no-data" role="status">
                  No acquisitions are available for this measurement.
                </div>

              </div>

              <div
                v-if="comparisonMode
                  ? comparisonRecords.length >= 2
                  : selectedMeasurementVisible && acquisitionCount > 0"
                class="waveform-grid"
              >
                  <section
                    v-for="plot in activeWaveformPlots"
                    :key="plot.key"
                    class="waveform-panel"
                    :data-testid="`dpt-waveform-${plot.key}`"
                  >
                    <CurvePlotter
                      v-if="plot.hasNumericPoints"
                      :title="plot.title"
                      :datasets="plot.datasets"
                      x-label="Time [µs]"
                      :x-value-decimals="6"
                      :y-label="plot.yLabel"
                      :show-legend="comparisonMode"
                    />
                    <div v-else class="no-data" role="status">
                      No finite samples are available for {{ plot.title }}.
                    </div>
                    <div
                      v-if="plot.missingLabels.length"
                      class="missing-comparison-traces"
                      role="status"
                    >
                      No {{ plot.title }} samples for: {{ plot.missingLabels.join('; ') }}.
                    </div>
                    <p v-if="plot.hasNonFiniteSamples" class="waveform-gap-note">
                      Non-finite raw samples are preserved and shown as gaps.
                    </p>
                  </section>
              </div>
            </template>
          </section>
        </div>

        <div v-else class="no-data">
          <p>No DPT data available for this transistor.</p>
        </div>
      </div>

      <!-- Generate Netlist Tab -->
      <div v-if="activeTab === 'netlist'" class="tab-panel">
        <h3>LTspice DPT Netlist Generator</h3>

        <div class="config-form">
          <div class="form-grid">
            <div class="form-section">
              <h4>Circuit Parameters</h4>
              <div class="form-row">
                <label>DC Bus Voltage:</label>
                <input v-model.number="dptConfig.v_dc" type="number" step="10" />
                <span class="unit">V</span>
              </div>
              <div class="form-row">
                <label>Target Current:</label>
                <input v-model.number="dptConfig.i_target" type="number" step="1" />
                <span class="unit">A</span>
              </div>
              <div class="form-row">
                <label>Load Inductance:</label>
                <input v-model.number="dptConfig.l_load" type="number" step="10e-6" />
                <span class="unit">H</span>
              </div>
            </div>

            <div class="form-section">
              <h4>Gate Drive</h4>
              <div class="form-row">
                <label>R_g Turn-On:</label>
                <input v-model.number="dptConfig.r_g_on" type="number" step="0.5" />
                <span class="unit">Ω</span>
              </div>
              <div class="form-row">
                <label>R_g Turn-Off:</label>
                <input v-model.number="dptConfig.r_g_off" type="number" step="0.5" />
                <span class="unit">Ω</span>
              </div>
              <div class="form-row">
                <label>V_gate On:</label>
                <input v-model.number="dptConfig.v_g_on" type="number" step="1" />
                <span class="unit">V</span>
              </div>
              <div class="form-row">
                <label>V_gate Off:</label>
                <input v-model.number="dptConfig.v_g_off" type="number" step="1" />
                <span class="unit">V</span>
              </div>
            </div>

            <div class="form-section">
              <h4>Timing</h4>
              <div class="form-row">
                <label>Dead Time:</label>
                <input v-model.number="dptConfig.t_dead" type="number" step="1e-6" />
                <span class="unit">s</span>
              </div>
              <div class="form-row">
                <label>Pulse 2 Duration:</label>
                <input v-model.number="dptConfig.t_pulse2" type="number" step="1e-6" />
                <span class="unit">s</span>
              </div>
            </div>
          </div>

          <button @click="generateNetlist" :disabled="generating" class="btn-generate">
            {{ generating ? 'Generating...' : 'Generate LTspice Netlist' }}
          </button>
        </div>

        <!-- Netlist Display -->
        <div v-if="generatedNetlist" class="netlist-display">
          <div class="netlist-header">
            <h4>Generated Netlist</h4>
            <div class="netlist-actions">
              <button @click="copyNetlist" class="btn-copy">
                📋 Copy to Clipboard
              </button>
              <button @click="downloadNetlist" class="btn-download">
                💾 Download .asc file
              </button>
            </div>
          </div>
          <pre class="netlist-content">{{ generatedNetlist.netlist }}</pre>
        </div>
      </div>

      <!-- Validate Tab -->
      <div v-if="activeTab === 'validate'" class="tab-panel">
        <h3>Validate DPT Data</h3>
        <p class="validation-description">
          Check DPT measurement data against datasheet specifications to identify
          potential issues or out-of-range test conditions.
        </p>

        <button @click="validateData" :disabled="validating" class="btn-validate">
          {{ validating ? 'Validating...' : 'Run Validation Checks' }}
        </button>

        <!-- Validation Results -->
        <div v-if="validationResults" class="validation-results">
          <div class="validation-summary" :class="validationResults.valid ? 'valid' : 'invalid'">
            <div class="summary-icon">
              {{ validationResults.valid ? '✅' : '⚠️' }}
            </div>
            <div class="summary-text">
              <strong>{{ validationResults.valid ? 'Valid' : 'Issues Found' }}</strong>
              <p>{{ validationResults.valid ? 'All checks passed' : 'See details below' }}</p>
            </div>
          </div>

          <!-- Statistics -->
          <div v-if="validationResults.statistics" class="statistics-section">
            <h4>Test Conditions Statistics</h4>
            <div class="stats-grid">
              <div class="stat-card">
                <div class="stat-label">Supply Voltage Range</div>
                <div class="stat-value">
                  {{ validationResults.statistics.v_supply_range[0] }} -
                  {{ validationResults.statistics.v_supply_range[1] }} V
                </div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Max Test Current</div>
                <div class="stat-value">
                  {{ validationResults.statistics.i_max_tested.toFixed(1) }} A
                </div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Temperature Range</div>
                <div class="stat-value">
                  {{ validationResults.statistics.t_j_range[0] }} -
                  {{ validationResults.statistics.t_j_range[1] }} °C
                </div>
              </div>
            </div>
          </div>

          <!-- Errors -->
          <div v-if="validationResults.errors.length > 0" class="error-list">
            <h4>Errors ({{ validationResults.errors.length }})</h4>
            <div v-for="(error, index) in validationResults.errors" :key="index" class="error-item">
              ❌ {{ error }}
            </div>
          </div>

          <!-- Warnings -->
          <div v-if="validationResults.warnings.length > 0" class="warning-list">
            <h4>Warnings ({{ validationResults.warnings.length }})</h4>
            <div v-for="(warning, index) in validationResults.warnings" :key="index" class="warning-item">
              ⚠️ {{ warning }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Error Display -->
    <div v-if="error" class="error-message">
      <strong>Error:</strong> {{ error }}
    </div>
  </div>
</template>

<script>
import { computed, ref, watch, onMounted } from 'vue'
import { transistorApi } from '../services/api'
import CurvePlotter from './CurvePlotter.vue'

export default {
  name: 'DPTManager',
  components: { CurvePlotter },
  setup() {
    const transistorList = ref([])
    const selectedTransistorId = ref('')
    const activeTab = ref('data')
    const dptData = ref(null)
    const selectedMeasurementIndex = ref(0)
    const selectedAcquisitionIndex = ref(0)
    const measurementSearch = ref('')
    const comparisonMode = ref(false)
    const selectedComparisonIndices = ref([])
    const comparisonAcquisitionIndices = ref({})
    const comparisonColors = ['#2563eb', '#dc2626', '#7c3aed', '#ea580c', '#059669', '#9333ea']
    const generating = ref(false)
    const validating = ref(false)
    const error = ref(null)

    // DPT Configuration
    const dptConfig = ref({
      v_dc: 400.0,
      i_target: 20.0,
      l_load: 100e-6,
      r_g_on: 10.0,
      r_g_off: 10.0,
      v_g_on: 15.0,
      v_g_off: -5.0,
      t_dead: 5e-6,
      t_pulse2: 2e-6
    })

    const generatedNetlist = ref(null)
    const validationResults = ref(null)
    const rawMeasurementRecords = computed(() => (
      Array.isArray(dptData.value?.raw_measurement_data)
        ? dptData.value.raw_measurement_data
        : []
    ))
    const filteredMeasurementRecords = computed(() => {
      const terms = measurementSearch.value
        .trim()
        .toLocaleLowerCase()
        .split(/\s+/)
        .filter(Boolean)
      return rawMeasurementRecords.value
        .map((measurement, index) => ({ measurement, index }))
        .filter(({ measurement }) => {
          if (terms.length === 0) return true
          const searchableText = measurementSearchText(measurement)
          return terms.every(term => searchableText.includes(term))
        })
    })
    const selectedMeasurement = computed(() => (
      rawMeasurementRecords.value[selectedMeasurementIndex.value] || null
    ))
    const selectedMeasurementVisible = computed(() => (
      filteredMeasurementRecords.value.some(
        record => record.index === selectedMeasurementIndex.value
      )
    ))
    function getAcquisitionCount(measurement) {
      if (!measurement) return 0
      return Math.max(
        0,
        ...['dpt_on_vds', 'dpt_on_id', 'dpt_off_vds', 'dpt_off_id']
          .map(field => Array.isArray(measurement[field]) ? measurement[field].length : 0)
      )
    }
    const acquisitionCount = computed(() => {
      return getAcquisitionCount(selectedMeasurement.value)
    })
    function conditionsForMeasurement(measurement) {
      const metadata = measurement?.metadata || {}
      const definitions = [
        ['dataset_type', 'Dataset type', ''],
        ['t_j', 'Junction temperature', '°C'],
        ['v_supply', 'Supply voltage', 'V'],
        ['v_g', 'Gate voltage', 'V'],
        ['v_g_off', 'Turn-off gate voltage', 'V'],
        ['r_g', 'Gate resistance', 'Ω'],
        ['r_g_off', 'Turn-off gate resistance', 'Ω'],
        ['load_inductance', 'Load inductance', 'H'],
        ['commutation_inductance', 'Commutation inductance', 'H'],
        ['measurement_date', 'Measurement date', ''],
        ['measurement_testbench', 'Test bench', ''],
        ['commutation_device', 'Commutation device', ''],
        ['comment', 'Comment', ''],
      ]
      return definitions.flatMap(([key, label, unit]) => {
        const value = metadata[key]
        if (value === null || value === undefined || value === '') return []
        const formatted = Array.isArray(value) ? value.join(', ') : String(value)
        return [{ label, value: unit ? `${formatted} ${unit}` : formatted }]
      })
    }
    const measurementConditions = computed(() => (
      conditionsForMeasurement(selectedMeasurement.value)
    ))
    function measurementSearchText(measurement) {
      const metadata = measurement?.metadata || {}
      const metadataValues = Object.values(metadata).flatMap(value => (
        Array.isArray(value) ? value : [value]
      ))
      const conditionValues = conditionsForMeasurement(measurement)
        .flatMap(condition => [condition.label, condition.value])
      return [...metadataValues, ...conditionValues]
        .filter(value => value !== null && value !== undefined && value !== '')
        .map(value => (
          typeof value === 'object' ? JSON.stringify(value) : String(value)
        ))
        .join(' ')
        .toLocaleLowerCase()
    }
    function projectRawSamples(samples) {
      const data = []
      let hasNonFiniteSamples = false
      if (Array.isArray(samples)) {
        for (const sample of samples) {
          if (!Array.isArray(sample) || sample.length < 2) continue
          const [time, value] = sample
          if (typeof time !== 'number' || !Number.isFinite(time)) continue
          const finiteValue = typeof value === 'number' && Number.isFinite(value)
          if (!finiteValue) hasNonFiniteSamples = true
          data.push({ x: time * 1e6, y: finiteValue ? value : null })
        }
      }
      return {
        data,
        hasNumericPoints: data.some(point => point.y !== null),
        hasNonFiniteSamples,
      }
    }
    const comparisonRecords = computed(() => (
      selectedComparisonIndices.value
        .filter(index => rawMeasurementRecords.value[index])
        .map(index => {
          const measurement = rawMeasurementRecords.value[index]
          const count = getAcquisitionCount(measurement)
          const requestedAcquisition = Number(comparisonAcquisitionIndices.value[index] || 0)
          const acquisitionIndex = count > 0
            ? Math.min(Math.max(0, requestedAcquisition), count - 1)
            : 0
          return {
            index,
            measurement,
            label: measurementLabel(measurement, index),
            conditions: conditionsForMeasurement(measurement),
            acquisitionCount: count,
            acquisitionIndex,
          }
        })
    ))
    const hiddenSelectedComparisonRecords = computed(() => (
      comparisonRecords.value.filter(record => (
        !filteredMeasurementRecords.value.some(
          visibleRecord => visibleRecord.index === record.index
        )
      ))
    ))
    const waveformPlots = computed(() => {
      const measurement = selectedMeasurement.value
      if (!measurement) return []
      const plots = [
        { key: 'on-vds', field: 'dpt_on_vds', title: 'Turn-on Vds', yLabel: 'Vds [V]', color: '#2563eb' },
        { key: 'on-id', field: 'dpt_on_id', title: 'Turn-on Id', yLabel: 'Id [A]', color: '#dc2626' },
        { key: 'off-vds', field: 'dpt_off_vds', title: 'Turn-off Vds', yLabel: 'Vds [V]', color: '#7c3aed' },
        { key: 'off-id', field: 'dpt_off_id', title: 'Turn-off Id', yLabel: 'Id [A]', color: '#ea580c' },
      ]
      return plots.map(plot => {
        const acquisitions = measurement[plot.field]
        const samples = Array.isArray(acquisitions)
          ? acquisitions[selectedAcquisitionIndex.value]
          : null
        const projection = projectRawSamples(samples)
        return {
          ...plot,
          datasets: [{
            label: plot.title,
            data: projection.data,
            borderColor: plot.color,
          }],
          hasNumericPoints: projection.hasNumericPoints,
          hasNonFiniteSamples: projection.hasNonFiniteSamples,
          missingLabels: [],
        }
      })
    })

    const comparisonWaveformPlots = computed(() => {
      const plots = [
        { key: 'on-vds', field: 'dpt_on_vds', title: 'Turn-on Vds', yLabel: 'Vds [V]' },
        { key: 'on-id', field: 'dpt_on_id', title: 'Turn-on Id', yLabel: 'Id [A]' },
        { key: 'off-vds', field: 'dpt_off_vds', title: 'Turn-off Vds', yLabel: 'Vds [V]' },
        { key: 'off-id', field: 'dpt_off_id', title: 'Turn-off Id', yLabel: 'Id [A]' },
      ]
      return plots.map(plot => {
        const datasets = []
        const missingLabels = []
        let hasNonFiniteSamples = false
        for (const record of comparisonRecords.value) {
          const acquisitions = record.measurement[plot.field]
          const samples = Array.isArray(acquisitions)
            ? acquisitions[record.acquisitionIndex]
            : null
          const traceLabel = `${record.label} · Acquisition ${record.acquisitionIndex + 1}`
          if (!Array.isArray(samples) || samples.length === 0) {
            missingLabels.push(traceLabel)
            continue
          }
          const projection = projectRawSamples(samples)
          if (projection.data.length === 0) {
            missingLabels.push(`${traceLabel} (no finite timestamps)`)
            continue
          }
          hasNonFiniteSamples ||= projection.hasNonFiniteSamples
          datasets.push({
            label: traceLabel,
            data: projection.data,
            borderColor: comparisonColor(record.index),
          })
        }
        return {
          ...plot,
          datasets,
          hasNumericPoints: datasets.some(dataset => (
            dataset.data.some(point => point.y !== null)
          )),
          hasNonFiniteSamples,
          missingLabels,
        }
      })
    })
    const activeWaveformPlots = computed(() => (
      comparisonMode.value ? comparisonWaveformPlots.value : waveformPlots.value
    ))

    function selectFirstVisibleMeasurementIfNeeded() {
      const visibleIndexes = filteredMeasurementRecords.value.map(
        record => record.index
      )
      if (
        visibleIndexes.length &&
        !visibleIndexes.includes(selectedMeasurementIndex.value)
      ) {
        selectedMeasurementIndex.value = visibleIndexes[0]
        selectedAcquisitionIndex.value = 0
      }
    }

    watch(measurementSearch, () => {
      if (comparisonMode.value) return
      selectFirstVisibleMeasurementIfNeeded()
    })

    function measurementLabel(measurement, index) {
      const metadata = measurement?.metadata || {}
      const label = [metadata.dataset_type, metadata.comment]
        .filter(value => value !== null && value !== undefined && value !== '')
        .join(' · ')
      return label
        ? `Measurement ${index + 1} · ${label}`
        : `Measurement ${index + 1}`
    }

    function toggleComparisonMode() {
      comparisonMode.value = !comparisonMode.value
      if (comparisonMode.value) {
        const visibleIndexes = filteredMeasurementRecords.value.map(
          record => record.index
        )
        const firstIndex = visibleIndexes.includes(selectedMeasurementIndex.value)
          ? selectedMeasurementIndex.value
          : visibleIndexes[0]
        const secondIndex = visibleIndexes.find(index => index !== firstIndex)
        selectedComparisonIndices.value = firstIndex === undefined
          ? []
          : secondIndex !== undefined
            ? [firstIndex, secondIndex]
            : [firstIndex]
        comparisonAcquisitionIndices.value = Object.fromEntries(
          selectedComparisonIndices.value.map(index => [index, 0])
        )
      } else {
        selectFirstVisibleMeasurementIfNeeded()
      }
    }

    function toggleComparisonRecord(index) {
      if (selectedComparisonIndices.value.includes(index)) {
        selectedComparisonIndices.value = selectedComparisonIndices.value.filter(
          selectedIndex => selectedIndex !== index
        )
        const nextAcquisitions = { ...comparisonAcquisitionIndices.value }
        delete nextAcquisitions[index]
        comparisonAcquisitionIndices.value = nextAcquisitions
      } else {
        selectedComparisonIndices.value = [
          ...selectedComparisonIndices.value,
          index,
        ].sort((left, right) => left - right)
        comparisonAcquisitionIndices.value = {
          ...comparisonAcquisitionIndices.value,
          [index]: 0,
        }
      }
    }

    function comparisonColor(index) {
      return comparisonColors[index % comparisonColors.length]
    }

    onMounted(async () => {
      try {
        const transistors = await transistorApi.getAll()
        transistorList.value = transistors.map(t => ({
          id: t.metadata?.name || t.name,
          name: t.metadata?.name || t.name
        }))
      } catch (err) {
        error.value = 'Failed to load transistor list'
      }
    })

    async function loadDPTData() {
      if (!selectedTransistorId.value) return

      try {
        error.value = null
        dptData.value = await transistorApi.getDPTData(selectedTransistorId.value)
        selectedMeasurementIndex.value = 0
        selectedAcquisitionIndex.value = 0
        measurementSearch.value = ''
        comparisonMode.value = false
        selectedComparisonIndices.value = []
        comparisonAcquisitionIndices.value = {}
      } catch (err) {
        error.value = 'Failed to load DPT data'
        dptData.value = null
      }
    }

    async function generateNetlist() {
      if (!selectedTransistorId.value) return

      generating.value = true
      error.value = null

      try {
        generatedNetlist.value = await transistorApi.generateDPTNetlist(
          selectedTransistorId.value,
          dptConfig.value
        )
      } catch (err) {
        error.value = 'Failed to generate netlist: ' + err.message
      } finally {
        generating.value = false
      }
    }

    async function validateData() {
      if (!selectedTransistorId.value) return

      validating.value = true
      error.value = null

      try {
        validationResults.value = await transistorApi.validateDPTData(selectedTransistorId.value)
      } catch (err) {
        error.value = 'Failed to validate data: ' + err.message
      } finally {
        validating.value = false
      }
    }

    function copyNetlist() {
      if (!generatedNetlist.value) return

      navigator.clipboard.writeText(generatedNetlist.value.netlist).then(() => {
        alert('Netlist copied to clipboard!')
      })
    }

    function downloadNetlist() {
      if (!generatedNetlist.value) return

      const blob = new Blob([generatedNetlist.value.netlist], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${generatedNetlist.value.transistor_name}_DPT.asc`
      link.click()
      URL.revokeObjectURL(url)
    }

    return {
      transistorList,
      selectedTransistorId,
      activeTab,
      dptData,
      rawMeasurementRecords,
      filteredMeasurementRecords,
      measurementSearch,
      selectedMeasurementVisible,
      hiddenSelectedComparisonRecords,
      selectedMeasurement,
      selectedMeasurementIndex,
      selectedAcquisitionIndex,
      comparisonMode,
      selectedComparisonIndices,
      comparisonRecords,
      comparisonAcquisitionIndices,
      toggleComparisonMode,
      toggleComparisonRecord,
      comparisonColor,
      acquisitionCount,
      measurementConditions,
      waveformPlots,
      activeWaveformPlots,
      measurementLabel,
      dptConfig,
      generating,
      validating,
      error,
      generatedNetlist,
      validationResults,
      loadDPTData,
      generateNetlist,
      validateData,
      copyNetlist,
      downloadNetlist
    }
  }
}
</script>

<style scoped>
.dpt-manager {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

h2 {
  font-size: 24px;
  margin-bottom: 10px;
}

.description {
  color: #666;
  margin-bottom: 30px;
}

/* Transistor Selector */
.transistor-selector {
  margin-bottom: 30px;
}

.transistor-selector label {
  display: block;
  font-weight: bold;
  margin-bottom: 10px;
}

.transistor-selector select {
  width: 100%;
  max-width: 400px;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
}

/* Tabs */
.tab-nav {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  border-bottom: 2px solid #e0e0e0;
}

.tab-btn {
  padding: 12px 20px;
  background: none;
  border: none;
  border-bottom: 3px solid transparent;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s;
}

.tab-btn.active {
  border-bottom-color: #4CAF50;
  font-weight: bold;
}

.tab-btn:hover {
  background: #f0f0f0;
}

/* Tab Panel */
.tab-panel {
  background: white;
  padding: 30px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.tab-panel h3 {
  font-size: 20px;
  margin-bottom: 20px;
}

/* Data Overview */
.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.status-card {
  padding: 20px;
  border-radius: 8px;
  text-align: center;
  background: #f5f5f5;
  border: 2px solid #e0e0e0;
}

.status-card.available {
  background: #e8f5e9;
  border-color: #4CAF50;
}

.status-icon {
  font-size: 36px;
  margin-bottom: 10px;
}

.status-label {
  font-weight: bold;
  margin-bottom: 5px;
}

.status-text {
  color: #666;
  font-size: 14px;
}

/* Metadata */
.metadata-section {
  padding: 20px;
  background: #f9f9f9;
  border-radius: 6px;
}

.metadata-section h4 {
  font-size: 16px;
  margin-bottom: 15px;
}

.metadata-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 15px;
}

.metadata-item {
  display: flex;
  justify-content: space-between;
}

.metadata-label {
  font-weight: bold;
}

.metadata-comment {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #e0e0e0;
}

.raw-waveforms-section {
  margin-top: 30px;
  padding: 20px;
  border: 1px solid #dbe3eb;
  border-radius: 8px;
  background: #fbfdff;
}

.raw-waveforms-section h4 {
  margin: 0 0 8px;
}

.waveform-description,
.waveform-gap-note {
  color: #5b6573;
  font-size: 14px;
}

.measurement-filter {
  max-width: 620px;
  margin: 18px 0;
}

.measurement-filter > label {
  display: block;
  margin-bottom: 7px;
  font-weight: 600;
}

.measurement-filter-controls {
  display: flex;
  gap: 8px;
}

.measurement-filter-controls input {
  min-width: 0;
  flex: 1;
  padding: 9px 10px;
  border: 1px solid #b8c2cc;
  border-radius: 4px;
  background: white;
}

.btn-clear-measurement-search {
  padding: 8px 12px;
  border: 1px solid #b8c2cc;
  border-radius: 4px;
  color: #344054;
  background: white;
  cursor: pointer;
}

.btn-clear-measurement-search:hover {
  background: #f3f4f6;
}

.measurement-filter-count,
.comparison-filter-note {
  margin: 7px 0 0;
  color: #5b6573;
  font-size: 13px;
}

.waveform-selectors,
.acquisition-selector {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin: 16px 0;
}

.waveform-selectors label,
.acquisition-selector label {
  font-weight: 600;
}

.waveform-selectors select,
.acquisition-selector select {
  min-width: 220px;
  padding: 8px 10px;
  border: 1px solid #b8c2cc;
  border-radius: 4px;
  background: white;
}

.btn-compare-waveforms {
  padding: 8px 12px;
  border: 1px solid #2563eb;
  border-radius: 4px;
  color: #1d4ed8;
  background: white;
  cursor: pointer;
}

.btn-compare-waveforms:hover {
  background: #eff6ff;
}

.dpt-comparison-controls {
  margin-top: 18px;
}

.comparison-record-selector {
  padding: 16px;
  border: 1px solid #dbe3eb;
  border-radius: 6px;
  background: white;
}

.comparison-selector-heading,
.comparison-record-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.comparison-selector-heading h5,
.comparison-record-heading h5 {
  margin: 0;
}

.comparison-record-option {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-top: 12px;
  cursor: pointer;
}

.comparison-record-option input {
  width: 16px;
  height: 16px;
}

.comparison-record-details {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 14px;
  margin-top: 16px;
}

.comparison-record-card {
  min-width: 0;
  padding: 15px;
  border: 1px solid #dbe3eb;
  border-radius: 6px;
  background: white;
}

.comparison-record-heading {
  justify-content: flex-start;
  margin-bottom: 12px;
}

.comparison-record-swatch {
  width: 14px;
  height: 14px;
  flex: 0 0 14px;
  border-radius: 50%;
}

.comparison-record-card .acquisition-selector {
  margin: 10px 0;
}

.comparison-conditions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 6px 12px;
  margin: 12px 0 0;
}

.comparison-conditions > div {
  display: flex;
  justify-content: space-between;
  gap: 8px;
}

.comparison-conditions dt {
  font-weight: 600;
}

.comparison-conditions dd {
  margin: 0;
  overflow-wrap: anywhere;
  text-align: right;
}

.single-acquisition-note,
.missing-comparison-traces {
  margin: 10px 0;
  color: #5b6573;
  font-size: 14px;
}

.measurement-conditions {
  margin: 18px 0;
  padding: 16px;
  border-radius: 6px;
  background: #f1f5f9;
}

.measurement-conditions h5 {
  margin: 0 0 10px;
}

.measurement-conditions dl {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 8px 18px;
  margin: 0;
}

.measurement-conditions dl > div {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.measurement-conditions dt {
  font-weight: 600;
}

.measurement-conditions dd {
  margin: 0;
  text-align: right;
  overflow-wrap: anywhere;
}

.waveform-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 20px;
  margin-top: 18px;
}

.waveform-panel {
  min-width: 0;
  padding: 14px;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: white;
}

.waveform-panel :deep(.curve-plotter .chart-container) {
  height: 280px;
}

.waveform-gap-note {
  margin: 8px 0 0;
}

/* Config Form */
.config-form {
  margin-bottom: 30px;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 30px;
  margin-bottom: 30px;
}

.form-section h4 {
  font-size: 16px;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 2px solid #e0e0e0;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 100px 40px;
  gap: 10px;
  align-items: center;
  margin-bottom: 15px;
}

.form-row label {
  font-size: 14px;
}

.form-row input {
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
}

.form-row .unit {
  font-size: 14px;
  color: #666;
}

/* Buttons */
.btn-generate,
.btn-validate {
  width: 100%;
  padding: 15px;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-generate:hover:not(:disabled),
.btn-validate:hover:not(:disabled) {
  background: #45a049;
}

.btn-generate:disabled,
.btn-validate:disabled {
  background: #ccc;
  cursor: not-allowed;
}

/* Netlist Display */
.netlist-display {
  margin-top: 30px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  overflow: hidden;
}

.netlist-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  background: #f5f5f5;
  border-bottom: 1px solid #e0e0e0;
}

.netlist-header h4 {
  margin: 0;
  font-size: 16px;
}

.netlist-actions {
  display: flex;
  gap: 10px;
}

.btn-copy,
.btn-download {
  padding: 8px 16px;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
}

.btn-copy:hover,
.btn-download:hover {
  background: #1976D2;
}

.netlist-content {
  padding: 20px;
  background: #fafafa;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  overflow-x: auto;
  max-height: 500px;
  overflow-y: auto;
}

/* Validation Results */
.validation-description {
  color: #666;
  margin-bottom: 20px;
}

.validation-results {
  margin-top: 30px;
}

.validation-summary {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 30px;
}

.validation-summary.valid {
  background: #e8f5e9;
  border-left: 4px solid #4CAF50;
}

.validation-summary.invalid {
  background: #fff3e0;
  border-left: 4px solid #ff9800;
}

.summary-icon {
  font-size: 48px;
}

.summary-text strong {
  font-size: 20px;
}

.summary-text p {
  margin: 5px 0 0 0;
  color: #666;
}

/* Statistics */
.statistics-section {
  margin-bottom: 30px;
}

.statistics-section h4 {
  font-size: 16px;
  margin-bottom: 15px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.stat-card {
  padding: 15px;
  background: #f9f9f9;
  border-radius: 6px;
  text-align: center;
}

.stat-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 20px;
  font-weight: bold;
}

/* Error and Warning Lists */
.error-list,
.warning-list {
  padding: 15px;
  border-radius: 6px;
  margin-bottom: 20px;
}

.error-list {
  background: #ffebee;
  border-left: 4px solid #f44336;
}

.warning-list {
  background: #fff3e0;
  border-left: 4px solid #ff9800;
}

.error-list h4,
.warning-list h4 {
  font-size: 16px;
  margin-bottom: 10px;
}

.error-item,
.warning-item {
  padding: 8px 0;
  border-bottom: 1px solid rgba(0,0,0,0.1);
}

.error-item:last-child,
.warning-item:last-child {
  border-bottom: none;
}

/* Error Message */
.error-message {
  margin-top: 20px;
  padding: 15px;
  background: #f8d7da;
  border-left: 4px solid #f44336;
  border-radius: 4px;
  color: #721c24;
}

.no-data {
  text-align: center;
  padding: 40px;
  color: #999;
}
</style>
