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
      <select
        v-model="selectedTransistorId"
        :disabled="rawImportSaving"
        @change="loadDPTData"
      >
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

          <section class="raw-import-section" data-testid="dpt-raw-import">
            <h4>Import an LTspice RAW measurement</h4>
            <p class="waveform-description">
              Supports one native LTspice transient result without parameter steps.
              Choose the Vds and Id traces, then enter each event window in seconds.
              The uploaded file is discarded after each request.
            </p>
            <div class="raw-import-file-row">
              <label for="dpt-raw-file">LTspice .raw file</label>
              <input
                id="dpt-raw-file"
                ref="rawImportFileInput"
                data-testid="dpt-raw-file"
                type="file"
                accept=".raw"
                :disabled="rawImportSaving"
                @change="selectRawImportFile"
              />
              <button
                type="button"
                data-testid="dpt-raw-preview-file"
                :disabled="!rawImportFile || rawImportBusy || rawImportSaving"
                @click="previewRawImportFile"
              >
                {{ rawImportBusy ? 'Reading…' : 'Read file and list traces' }}
              </button>
            </div>

            <div v-if="rawImportPreview" class="raw-import-preview" data-testid="dpt-raw-file-preview">
              <p>
                {{ rawImportPreview.filename }} · {{ rawImportPreview.plot_name }} ·
                {{ rawImportPreview.sample_count.toLocaleString() }} samples
              </p>
              <p class="raw-import-axis-note">
                Time axis identified by the parser: <strong>{{ rawImportPreview.time_trace_name }}</strong>
              </p>
              <div class="raw-import-trace-grid">
                <div>
                  <label for="dpt-raw-vds-trace">Vds trace (volts)</label>
                  <select
                    id="dpt-raw-vds-trace"
                    data-testid="dpt-raw-vds-trace"
                    v-model="rawImportForm.vds_trace"
                    :disabled="rawImportSaving"
                  >
                    <option value="">Choose a trace</option>
                    <option
                      v-for="trace in rawImportValueTraces"
                      :key="trace"
                      :value="trace"
                    >
                      {{ trace }}
                    </option>
                  </select>
                </div>
                <div>
                  <label for="dpt-raw-id-trace">Id trace (amperes)</label>
                  <select
                    id="dpt-raw-id-trace"
                    data-testid="dpt-raw-id-trace"
                    v-model="rawImportForm.id_trace"
                    :disabled="rawImportSaving"
                  >
                    <option value="">Choose a trace</option>
                    <option
                      v-for="trace in rawImportValueTraces"
                      :key="trace"
                      :value="trace"
                    >
                      {{ trace }}
                    </option>
                  </select>
                </div>
              </div>
              <button
                type="button"
                data-testid="dpt-raw-preview-traces"
                :disabled="!rawImportTraceMappingReady || rawImportBusy || rawImportSaving"
                @click="previewRawImportTraces"
              >
                Preview selected traces
              </button>
            </div>

            <div
              v-if="rawImportTracePreview"
              class="raw-import-waveform-preview"
              data-testid="dpt-raw-waveform-preview"
            >
              <p>
                Preview shows up to {{ rawImportTracePreview.preview_sample_count.toLocaleString() }}
                display samples from {{ rawImportTracePreview.sample_count.toLocaleString() }}
                source samples. Stored data will use every sample inside each selected window.
                Time range: {{ rawImportTracePreview.time_range_s[0] }} to
                {{ rawImportTracePreview.time_range_s[1] }} s.
              </p>
              <p
                v-if="rawImportHasNonFiniteSamples"
                class="raw-import-gap-note"
                role="status"
              >
                Non-finite trace samples are shown as gaps. Event windows can be imported
                when every selected Vds and Id sample is finite.
              </p>
              <div class="raw-import-plots">
                <CurvePlotter
                  title="Selected Vds trace preview"
                  :datasets="rawImportVdsPreviewDatasets"
                  x-label="Time [µs]"
                  :x-value-decimals="6"
                  y-label="Vds [V]"
                  :show-legend="false"
                />
                <CurvePlotter
                  title="Selected Id trace preview"
                  :datasets="rawImportIdPreviewDatasets"
                  x-label="Time [µs]"
                  :x-value-decimals="6"
                  y-label="Id [A]"
                  :show-legend="false"
                />
              </div>
              <div class="raw-import-window-grid">
                <fieldset>
                  <legend>Turn-on event window</legend>
                  <label for="dpt-raw-on-start">Start [s]</label>
                  <input
                    id="dpt-raw-on-start"
                    v-model="rawImportForm.turn_on_start_s"
                    data-testid="dpt-raw-on-start"
                    type="number"
                    step="any"
                    :disabled="rawImportSaving"
                  />
                  <label for="dpt-raw-on-end">End [s]</label>
                  <input
                    id="dpt-raw-on-end"
                    v-model="rawImportForm.turn_on_end_s"
                    data-testid="dpt-raw-on-end"
                    type="number"
                    step="any"
                    :disabled="rawImportSaving"
                  />
                </fieldset>
                <fieldset>
                  <legend>Turn-off event window</legend>
                  <label for="dpt-raw-off-start">Start [s]</label>
                  <input
                    id="dpt-raw-off-start"
                    v-model="rawImportForm.turn_off_start_s"
                    data-testid="dpt-raw-off-start"
                    type="number"
                    step="any"
                    :disabled="rawImportSaving"
                  />
                  <label for="dpt-raw-off-end">End [s]</label>
                  <input
                    id="dpt-raw-off-end"
                    v-model="rawImportForm.turn_off_end_s"
                    data-testid="dpt-raw-off-end"
                    type="number"
                    step="any"
                    :disabled="rawImportSaving"
                  />
                </fieldset>
              </div>

              <details class="raw-import-metadata">
                <summary>Optional measurement conditions and provenance</summary>
                <div class="raw-import-metadata-grid" :class="{ 'is-saving': rawImportSaving }">
                  <label>Junction temperature [°C]<input v-model="rawImportMetadata.t_j" data-testid="dpt-raw-metadata-tj" :disabled="rawImportSaving" type="number" step="any" /></label>
                  <label>Supply voltage [V]<input v-model="rawImportMetadata.v_supply" :disabled="rawImportSaving" type="number" step="any" /></label>
                  <label>Turn-on gate voltage [V]<input v-model="rawImportMetadata.v_g" :disabled="rawImportSaving" type="number" step="any" /></label>
                  <label>Turn-off gate voltage [V]<input v-model="rawImportMetadata.v_g_off" :disabled="rawImportSaving" type="number" step="any" /></label>
                  <label>Turn-on gate resistance [Ω]<input v-model="rawImportMetadata.r_g" :disabled="rawImportSaving" type="number" step="any" /></label>
                  <label>Turn-off gate resistance [Ω]<input v-model="rawImportMetadata.r_g_off" :disabled="rawImportSaving" type="number" step="any" /></label>
                  <label>Load inductance [µH]<input v-model="rawImportMetadata.load_inductance" :disabled="rawImportSaving" type="number" step="any" /></label>
                  <label>Commutation inductance [µH]<input v-model="rawImportMetadata.commutation_inductance" :disabled="rawImportSaving" type="number" step="any" /></label>
                  <label>Measurement date<input v-model="rawImportMetadata.measurement_date" :disabled="rawImportSaving" type="date" /></label>
                  <label>Test bench<input v-model="rawImportMetadata.measurement_testbench" :disabled="rawImportSaving" type="text" maxlength="256" /></label>
                  <label>Commutation device<input v-model="rawImportMetadata.commutation_device" :disabled="rawImportSaving" type="text" maxlength="256" /></label>
                  <label class="raw-import-comment">Comment<textarea v-model="rawImportMetadata.comment" :disabled="rawImportSaving" maxlength="2000" rows="2" /></label>
                </div>
              </details>
              <button
                type="button"
                class="btn-import-raw-dpt"
                data-testid="dpt-raw-import-submit"
                :disabled="!rawImportSelectionValid || rawImportBusy || rawImportSaving"
                @click="importRawDptMeasurement"
              >
                {{ rawImportSaving ? 'Importing…' : 'Append DPT measurement' }}
              </button>
            </div>
            <p v-if="rawImportStatus" class="raw-import-status" role="status">
              {{ rawImportStatus }}
            </p>
            <p v-if="rawImportError" class="raw-import-error" role="alert">
              {{ rawImportError }}
            </p>
          </section>

          <section class="raw-waveforms-section" aria-labelledby="raw-waveforms-title">
            <h4 id="raw-waveforms-title">Stored Raw DPT Waveforms</h4>
            <p class="waveform-description">
              These plots show stored samples only. They do not calculate switching energy.
            </p>
            <p
              v-if="dptData.waveform_preview"
              class="waveform-preview-notice"
              data-testid="dpt-waveform-preview-notice"
              role="status"
            >
              <template v-if="dptData.waveform_preview.downsampled">
                {{ dptData.waveform_preview.downsampled_acquisition_count }} stored acquisitions are uniformly downsampled to at most
                {{ dptData.waveform_preview.max_points_per_acquisition }} points each, with first and last samples retained.
              </template>
              <template v-else>
                Stored acquisitions are shown in full up to
                {{ dptData.waveform_preview.max_points_per_acquisition }} points each.
              </template>
              The response is capped at {{ dptData.waveform_preview.max_total_points }} sample points; all source samples remain stored.
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
    const dptDataRequestVersion = ref(0)
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
    const rawImportFileInput = ref(null)
    const rawImportFile = ref(null)
    const rawImportBusy = ref(false)
    const rawImportSaving = ref(false)
    const rawImportRequestVersion = ref(0)
    const rawImportHeaderState = ref(null)
    const rawImportReviewedState = ref(null)
    const rawImportPreview = ref(null)
    const rawImportTracePreview = ref(null)
    const rawImportStatus = ref('')
    const rawImportError = ref('')
    const rawImportForm = ref(emptyRawImportForm())
    const rawImportMetadata = ref(emptyRawImportMetadata())

    function emptyRawImportForm() {
      return {
        vds_trace: '',
        id_trace: '',
        turn_on_start_s: '',
        turn_on_end_s: '',
        turn_off_start_s: '',
        turn_off_end_s: ''
      }
    }

    function emptyRawImportMetadata() {
      return {
        t_j: '',
        v_supply: '',
        v_g: '',
        v_g_off: '',
        r_g: '',
        r_g_off: '',
        load_inductance: '',
        commutation_inductance: '',
        measurement_date: '',
        measurement_testbench: '',
        commutation_device: '',
        comment: ''
      }
    }

    function invalidateRawImportReview({ resetWindows = true } = {}) {
      rawImportRequestVersion.value += 1
      rawImportBusy.value = false
      rawImportTracePreview.value = null
      rawImportReviewedState.value = null
      if (resetWindows) {
        rawImportForm.value = {
          ...rawImportForm.value,
          turn_on_start_s: '',
          turn_on_end_s: '',
          turn_off_start_s: '',
          turn_off_end_s: ''
        }
      }
    }

    function resetRawImportWorkflow() {
      rawImportRequestVersion.value += 1
      rawImportFile.value = null
      rawImportHeaderState.value = null
      rawImportPreview.value = null
      rawImportTracePreview.value = null
      rawImportReviewedState.value = null
      rawImportForm.value = emptyRawImportForm()
      rawImportMetadata.value = emptyRawImportMetadata()
      rawImportBusy.value = false
      rawImportStatus.value = ''
      rawImportError.value = ''
      if (rawImportFileInput.value) rawImportFileInput.value.value = ''
    }

    function rawImportRequestSnapshot() {
      return {
        version: rawImportRequestVersion.value,
        file: rawImportFile.value,
        transistorId: selectedTransistorId.value,
        timeTrace: rawImportPreview.value?.time_trace_name || null,
        vdsTrace: rawImportForm.value.vds_trace,
        idTrace: rawImportForm.value.id_trace
      }
    }

    function rawImportSnapshotIsCurrent(snapshot) {
      return snapshot.version === rawImportRequestVersion.value
        && snapshot.file === rawImportFile.value
        && snapshot.transistorId === selectedTransistorId.value
        && snapshot.timeTrace === (rawImportPreview.value?.time_trace_name || null)
        && snapshot.vdsTrace === rawImportForm.value.vds_trace
        && snapshot.idTrace === rawImportForm.value.id_trace
    }

    function rawImportHeaderIsCurrent() {
      return Boolean(rawImportPreview.value && rawImportHeaderState.value)
        && rawImportHeaderState.value.file === rawImportFile.value
        && rawImportHeaderState.value.transistorId === selectedTransistorId.value
    }

    function rawImportReviewIsCurrent() {
      const review = rawImportReviewedState.value
      return Boolean(review && rawImportTracePreview.value && rawImportHeaderIsCurrent())
        && review.file === rawImportFile.value
        && review.transistorId === selectedTransistorId.value
        && review.timeTrace === rawImportPreview.value.time_trace_name
        && review.vdsTrace === rawImportForm.value.vds_trace
        && review.idTrace === rawImportForm.value.id_trace
    }

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
    const rawImportValueTraces = computed(() => (
      rawImportHeaderIsCurrent() && rawImportPreview.value?.trace_names?.filter(
        trace => trace !== rawImportPreview.value.time_trace_name
      ) || []
    ))
    const rawImportTraceMappingReady = computed(() => (
      Boolean(rawImportFile.value)
      && rawImportHeaderIsCurrent()
      && Boolean(rawImportPreview.value?.time_trace_name)
      && Boolean(rawImportForm.value.vds_trace)
      && Boolean(rawImportForm.value.id_trace)
      && rawImportForm.value.vds_trace !== rawImportForm.value.id_trace
    ))
    const rawImportSelectionValid = computed(() => {
      if (!rawImportReviewIsCurrent() || !rawImportTraceMappingReady.value) return false
      const windows = [
        [rawImportForm.value.turn_on_start_s, rawImportForm.value.turn_on_end_s],
        [rawImportForm.value.turn_off_start_s, rawImportForm.value.turn_off_end_s]
      ]
      return windows.every(([start, end]) => (
        start !== '' && end !== ''
        && Number.isFinite(Number(start))
        && Number.isFinite(Number(end))
        && Number(end) >= Number(start)
      ))
    })
    const rawImportHasNonFiniteSamples = computed(() => (
      Object.values(rawImportTracePreview.value?.non_finite_preview_sample_counts || {})
        .some(count => count > 0)
    ))
    function rawImportRequest(includeWindows = true) {
      const metadata = {}
      const numberFields = new Set([
        't_j', 'v_supply', 'v_g', 'v_g_off', 'r_g', 'r_g_off',
        'load_inductance', 'commutation_inductance'
      ])
      for (const [key, value] of Object.entries(rawImportMetadata.value)) {
        if (value === '' || value === null || value === undefined) continue
        if (numberFields.has(key)) {
          const numericValue = Number(value)
          if (Number.isFinite(numericValue)) metadata[key] = numericValue
        } else {
          metadata[key] = value
        }
      }
      const request = {
        time_trace: rawImportPreview.value.time_trace_name,
        vds_trace: rawImportForm.value.vds_trace,
        id_trace: rawImportForm.value.id_trace
      }
      if (includeWindows) {
        request.turn_on = {
          start_s: Number(rawImportForm.value.turn_on_start_s),
          end_s: Number(rawImportForm.value.turn_on_end_s)
        }
        request.turn_off = {
          start_s: Number(rawImportForm.value.turn_off_start_s),
          end_s: Number(rawImportForm.value.turn_off_end_s)
        }
        request.metadata = metadata
      }
      return request
    }
    function rawImportErrorMessage(err) {
      return err?.response?.data?.detail || err?.message || 'DPT RAW import failed.'
    }
    function selectRawImportFile(event) {
      invalidateRawImportReview()
      rawImportFile.value = event.target.files?.[0] || null
      rawImportHeaderState.value = null
      rawImportPreview.value = null
      rawImportStatus.value = ''
      rawImportError.value = ''
      rawImportForm.value = emptyRawImportForm()
    }
    async function previewRawImportFile() {
      if (!rawImportFile.value || !selectedTransistorId.value) return
      invalidateRawImportReview()
      rawImportPreview.value = null
      rawImportHeaderState.value = null
      const snapshot = rawImportRequestSnapshot()
      rawImportBusy.value = true
      rawImportError.value = ''
      rawImportStatus.value = ''
      try {
        const preview = await transistorApi.previewDPTRawImport(
          snapshot.transistorId,
          snapshot.file
        )
        if (!rawImportSnapshotIsCurrent(snapshot)) return
        rawImportPreview.value = preview
        rawImportHeaderState.value = {
          file: snapshot.file,
          transistorId: snapshot.transistorId
        }
        rawImportStatus.value = 'File header read. Choose the Vds and Id traces.'
      } catch (err) {
        if (!rawImportSnapshotIsCurrent(snapshot)) return
        rawImportPreview.value = null
        rawImportError.value = rawImportErrorMessage(err)
      } finally {
        if (snapshot.version === rawImportRequestVersion.value) {
          rawImportBusy.value = false
        }
      }
    }
    async function previewRawImportTraces() {
      if (!rawImportTraceMappingReady.value || !selectedTransistorId.value) return
      invalidateRawImportReview()
      const snapshot = rawImportRequestSnapshot()
      const request = rawImportRequest(false)
      rawImportBusy.value = true
      rawImportError.value = ''
      rawImportStatus.value = ''
      try {
        const preview = await transistorApi.previewDPTRawTraces(
          snapshot.transistorId,
          snapshot.file,
          request
        )
        if (!rawImportSnapshotIsCurrent(snapshot)) return
        rawImportTracePreview.value = preview
        rawImportReviewedState.value = {
          file: snapshot.file,
          transistorId: snapshot.transistorId,
          timeTrace: snapshot.timeTrace,
          vdsTrace: snapshot.vdsTrace,
          idTrace: snapshot.idTrace
        }
        rawImportStatus.value = 'Trace preview loaded. Review the plots and enter both event windows.'
      } catch (err) {
        if (!rawImportSnapshotIsCurrent(snapshot)) return
        rawImportTracePreview.value = null
        rawImportError.value = rawImportErrorMessage(err)
      } finally {
        if (snapshot.version === rawImportRequestVersion.value) {
          rawImportBusy.value = false
        }
      }
    }
    function rawImportPreviewDataset(trace, title, color) {
      const samples = rawImportTracePreview.value?.traces?.[trace] || []
      return [{
        label: title,
        data: samples.map(([time, value]) => ({ x: time * 1e6, y: value })),
        borderColor: color,
        pointRadius: 0
      }]
    }
    const rawImportVdsPreviewDatasets = computed(() => (
      rawImportPreviewDataset(rawImportForm.value.vds_trace, 'Vds', '#2563eb')
    ))
    const rawImportIdPreviewDatasets = computed(() => (
      rawImportPreviewDataset(rawImportForm.value.id_trace, 'Id', '#dc2626')
    ))
    async function importRawDptMeasurement() {
      if (!rawImportSelectionValid.value || !selectedTransistorId.value) return
      const snapshot = rawImportRequestSnapshot()
      if (!rawImportSnapshotIsCurrent(snapshot) || !rawImportReviewIsCurrent()) return
      const request = rawImportRequest(true)
      rawImportSaving.value = true
      rawImportError.value = ''
      rawImportStatus.value = ''
      try {
        const result = await transistorApi.importDPTRawMeasurement(
          snapshot.transistorId,
          snapshot.file,
          request
        )
        if (!rawImportSnapshotIsCurrent(snapshot) || !rawImportReviewIsCurrent()) return
        rawImportStatus.value = (
          `Added one DPT record with ${result.on_sample_count} turn-on and `
          + `${result.off_sample_count} turn-off samples.`
        )
        rawImportFile.value = null
        rawImportHeaderState.value = null
        rawImportPreview.value = null
        rawImportTracePreview.value = null
        rawImportReviewedState.value = null
        if (rawImportFileInput.value) rawImportFileInput.value.value = ''
        await loadDPTData()
      } catch (err) {
        if (rawImportSnapshotIsCurrent(snapshot)) {
          rawImportError.value = rawImportErrorMessage(err)
        }
      } finally {
        rawImportSaving.value = false
      }
    }
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
        ['load_inductance', 'Load inductance', 'µH'],
        ['commutation_inductance', 'Commutation inductance', 'µH'],
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
            pointRadius: 0,
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
            pointRadius: 0,
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

    watch(
      selectedTransistorId,
      () => {
        dptDataRequestVersion.value += 1
        dptData.value = null
        resetRawImportWorkflow()
      },
      { flush: 'sync' }
    )

    watch(
      [() => rawImportForm.value.vds_trace, () => rawImportForm.value.id_trace],
      () => {
        invalidateRawImportReview()
        rawImportBusy.value = false
        rawImportStatus.value = ''
        rawImportError.value = ''
      },
      { flush: 'sync' }
    )

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
      const requestedTransistorId = selectedTransistorId.value
      if (!requestedTransistorId) return
      const requestVersion = ++dptDataRequestVersion.value

      try {
        error.value = null
        const response = await transistorApi.getDPTData(
          requestedTransistorId,
          { waveformPreviewPoints: 800 }
        )
        if (
          requestVersion !== dptDataRequestVersion.value
          || requestedTransistorId !== selectedTransistorId.value
        ) return
        dptData.value = response
        selectedMeasurementIndex.value = 0
        selectedAcquisitionIndex.value = 0
        measurementSearch.value = ''
        comparisonMode.value = false
        selectedComparisonIndices.value = []
        comparisonAcquisitionIndices.value = {}
      } catch (err) {
        if (
          requestVersion !== dptDataRequestVersion.value
          || requestedTransistorId !== selectedTransistorId.value
        ) return
        error.value = err?.response?.data?.detail || 'Failed to load DPT data'
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
      rawImportFileInput,
      rawImportFile,
      rawImportBusy,
      rawImportSaving,
      rawImportPreview,
      rawImportTracePreview,
      rawImportStatus,
      rawImportError,
      rawImportForm,
      rawImportMetadata,
      rawImportValueTraces,
      rawImportTraceMappingReady,
      rawImportSelectionValid,
      rawImportHasNonFiniteSamples,
      rawImportVdsPreviewDatasets,
      rawImportIdPreviewDatasets,
      selectRawImportFile,
      previewRawImportFile,
      previewRawImportTraces,
      importRawDptMeasurement,
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

.raw-import-section {
  margin-top: 24px;
  padding: 20px;
  border: 1px solid #cbd8e4;
  border-radius: 8px;
  background: #f7fafc;
}

.raw-import-section h4 {
  margin: 0 0 8px;
}

.raw-import-file-row,
.raw-import-trace-grid,
.raw-import-window-grid,
.raw-import-metadata-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 12px;
  align-items: end;
  margin: 16px 0;
}

.raw-import-file-row {
  grid-template-columns: minmax(220px, 1fr) auto;
}

.raw-import-section label {
  display: grid;
  gap: 6px;
  font-weight: 600;
}

.raw-import-section input,
.raw-import-section select,
.raw-import-section textarea {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  padding: 8px;
  border: 1px solid #aebbc8;
  border-radius: 4px;
  background: #fff;
  font: inherit;
  font-weight: 400;
}

.raw-import-axis-note {
  color: #334155;
}

.raw-import-plots {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
  margin: 16px 0;
}

.raw-import-window-grid fieldset {
  display: grid;
  grid-template-columns: minmax(72px, auto) minmax(120px, 1fr);
  gap: 8px;
  align-items: center;
  padding: 12px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
}

.raw-import-window-grid legend {
  padding: 0 5px;
  font-weight: 600;
}

.raw-import-metadata {
  margin: 18px 0;
}

.raw-import-metadata summary {
  cursor: pointer;
  font-weight: 600;
}

.raw-import-metadata-grid {
  margin-top: 14px;
}

.raw-import-comment {
  grid-column: 1 / -1;
}

.btn-import-raw-dpt {
  margin-top: 8px;
}

.raw-import-status {
  color: #166534;
  margin-bottom: 0;
}

.raw-import-error {
  color: #b91c1c;
  margin-bottom: 0;
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
