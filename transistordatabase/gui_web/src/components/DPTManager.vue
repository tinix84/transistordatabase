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
        :disabled="rawImportSaving || dptEnergySaving"
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
              Select one transient plot, the source steps to import, and explicit
              turn-on and turn-off windows in seconds. Step parameters and conditions
              are never read from the RAW file or a companion log.
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
              <div v-if="rawImportPreview.plots.length > 1" class="raw-import-plot-select">
                <label for="dpt-raw-plot">Transient plot</label>
                <select
                  id="dpt-raw-plot"
                  data-testid="dpt-raw-plot"
                  v-model="rawImportForm.plot_index"
                  :disabled="rawImportSaving"
                  @change="selectRawImportPlot"
                >
                  <option value="">Choose a transient plot</option>
                  <option
                    v-for="plot in rawImportPreview.plots"
                    :key="plot.plot_index"
                    :value="String(plot.plot_index)"
                    :disabled="!plot.is_transient"
                  >
                    Plot {{ plot.plot_index + 1 }} · {{ plot.plot_name }} ·
                    {{ plot.sample_count.toLocaleString() }} samples
                  </option>
                </select>
              </div>
              <p>
                {{ rawImportPreview.filename }} ·
                {{ rawImportSelectedPlot?.plot_name || 'Choose a transient plot' }}
                <template v-if="rawImportSelectedPlot">
                  · {{ rawImportSelectedPlot.sample_count.toLocaleString() }} samples
                </template>
              </p>
              <p v-if="rawImportSelectedPlot" class="raw-import-axis-note">
                Time axis identified by the parser:
                <strong>{{ rawImportSelectedPlot.trace_names[0] }}</strong>
              </p>
              <fieldset
                v-if="rawImportSelectedPlot?.is_stepped"
                class="raw-import-step-select"
                data-testid="dpt-raw-step-select"
              >
                <legend>Select source steps</legend>
                <label
                  v-for="step in rawImportSelectedPlot.steps"
                  :key="step.step_index"
                  class="raw-import-step-option"
                >
                  <input
                    v-model="rawImportForm.step_indices"
                    type="checkbox"
                    :value="String(step.step_index)"
                    :disabled="rawImportSaving"
                    @change="selectRawImportSteps"
                  />
                  Step {{ step.step_index }} · {{ step.sample_count }} samples ·
                  {{ step.time_range_s[0] }} to {{ step.time_range_s[1] }} s
                </label>
              </fieldset>
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
                {{ rawImportWindowInputsValid
                  ? 'Review import windows and conditions'
                  : 'Preview selected traces' }}
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
                <template v-if="rawImportTracePreview.time_range_s">
                  Time range: {{ rawImportTracePreview.time_range_s[0] }} to
                  {{ rawImportTracePreview.time_range_s[1] }} s.
                </template>
                <template v-else>
                  {{ rawImportTracePreview.steps.length }} selected source steps.
                </template>
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
                  :show-legend="Boolean(rawImportSelectedPlot?.is_stepped)"
                />
                <CurvePlotter
                  title="Selected Id trace preview"
                  :datasets="rawImportIdPreviewDatasets"
                  x-label="Time [µs]"
                  :x-value-decimals="6"
                  y-label="Id [A]"
                  :show-legend="Boolean(rawImportSelectedPlot?.is_stepped)"
                />
              </div>
              <div
                v-if="rawImportRequiresExplicitContext"
                class="raw-import-step-windows"
                data-testid="dpt-raw-step-windows"
              >
                <fieldset
                  v-for="step in rawImportSelectedSteps"
                  :key="step.step_index"
                  class="raw-import-step-window"
                >
                  <legend>Step {{ step.step_index }} acquisition</legend>
                  <label>
                    Acquisition current [A]
                    <input
                      v-model="rawImportStepForms[step.step_index].i_x"
                      :data-testid="'dpt-raw-step-' + step.step_index + '-current'"
                      type="number"
                      step="any"
                      :disabled="rawImportSaving"
                    />
                  </label>
                  <label>
                    Turn-on start [s]
                    <input
                      v-model="rawImportStepForms[step.step_index].turn_on_start_s"
                      :data-testid="'dpt-raw-step-' + step.step_index + '-on-start'"
                      type="number"
                      step="any"
                      :disabled="rawImportSaving"
                    />
                  </label>
                  <label>
                    Turn-on end [s]
                    <input
                      v-model="rawImportStepForms[step.step_index].turn_on_end_s"
                      :data-testid="'dpt-raw-step-' + step.step_index + '-on-end'"
                      type="number"
                      step="any"
                      :disabled="rawImportSaving"
                    />
                  </label>
                  <label>
                    Turn-off start [s]
                    <input
                      v-model="rawImportStepForms[step.step_index].turn_off_start_s"
                      :data-testid="'dpt-raw-step-' + step.step_index + '-off-start'"
                      type="number"
                      step="any"
                      :disabled="rawImportSaving"
                    />
                  </label>
                  <label>
                    Turn-off end [s]
                    <input
                      v-model="rawImportStepForms[step.step_index].turn_off_end_s"
                      :data-testid="'dpt-raw-step-' + step.step_index + '-off-end'"
                      type="number"
                      step="any"
                      :disabled="rawImportSaving"
                    />
                  </label>
                </fieldset>
              </div>
              <div v-else class="raw-import-window-grid">
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

              <details class="raw-import-metadata" :open="rawImportRequiresExplicitContext">
                <summary>
                  {{ rawImportRequiresExplicitContext
                    ? 'Required conditions for this source selection'
                    : 'Optional measurement conditions and provenance' }}
                </summary>
                <p v-if="rawImportRequiresExplicitContext" class="raw-import-context-note">
                  Enter values known from the test setup. RAW and .log step values are
                  not read. Group selected steps only when every condition below is the
                  same; use separate records if another condition changed.
                  <template v-if="!rawImportSelectedPlot?.is_stepped">
                    The selected plot is unstepped, so its sole source acquisition is
                    step 0; enter its current explicitly.
                  </template>
                </p>
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
              <label
                v-if="rawImportRequiresExplicitContext"
                class="raw-import-context-confirmation"
              >
                <input
                  v-model="rawImportForm.same_context_confirmed"
                  data-testid="dpt-raw-same-context"
                  type="checkbox"
                  :disabled="rawImportSaving"
                  @change="selectRawImportContext"
                />
                I confirm selected steps share these entered conditions; only their
                acquisition current differs.
              </label>
              <button
                type="button"
                class="btn-import-raw-dpt"
                data-testid="dpt-raw-import-submit"
                :disabled="!rawImportSelectionValid || rawImportBusy || rawImportSaving"
                @click="importRawDptMeasurement"
              >
                {{ rawImportSaving
                  ? 'Importing…'
                  : rawImportRequiresExplicitContext
                    ? 'Confirm and append reviewed DPT measurement'
                    : 'Append DPT measurement' }}
              </button>
            </div>
            <p v-if="rawImportStatus" class="raw-import-status" role="status">
              {{ rawImportStatus }}
            </p>
            <p v-if="rawImportError" class="raw-import-error" data-testid="dpt-raw-error" role="alert">
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

          <section class="dpt-energy-section" data-testid="dpt-energy-derivation">
            <h4>Derive switching energy from stored event windows</h4>
            <p class="waveform-description">
              This is a raw-window measurement. Signed instantaneous power
              <code>p(t) = Vds(t) · Id(t)</code> is integrated by the trapezoid rule
              from the first through last exact stored sample. No boundary detection,
              resampling, interpolation, threshold extension, or automatic sign change
              is applied. Choose an explicit current-axis value and both polarity signs
              for every event; current is not inferred from the transient.
            </p>

            <div class="dpt-energy-form">
              <label for="dpt-energy-loss-type">Energy curve</label>
              <select
                id="dpt-energy-loss-type"
                data-testid="dpt-energy-loss-type"
                v-model="dptEnergyForm.loss_type"
                :disabled="dptEnergyBusy || dptEnergySaving"
              >
                <option value="e_on">Turn-on E_on</option>
                <option value="e_off">Turn-off E_off</option>
              </select>
            </div>

            <div
              v-for="(point, pointIndex) in dptEnergyForm.points"
              :key="point.id"
              class="dpt-energy-point"
              :data-testid="`dpt-energy-point-${pointIndex}`"
            >
              <h5>Event {{ pointIndex + 1 }}</h5>
              <label :for="`dpt-energy-record-${pointIndex}`">Stored measurement</label>
              <select
                :id="`dpt-energy-record-${pointIndex}`"
                :data-testid="`dpt-energy-record-${pointIndex}`"
                v-model.number="point.measurement_index"
                :disabled="dptEnergyBusy || dptEnergySaving"
                @change="point.acquisition_index = null"
              >
                <option :value="null">Choose a stored measurement</option>
                <option
                  v-for="record in dptEnergyMeasurementOptions"
                  :key="record.index"
                  :value="record.index"
                >
                  {{ record.label }}
                </option>
              </select>

              <label :for="`dpt-energy-acquisition-${pointIndex}`">Stored event acquisition</label>
              <select
                :id="`dpt-energy-acquisition-${pointIndex}`"
                :data-testid="`dpt-energy-acquisition-${pointIndex}`"
                v-model.number="point.acquisition_index"
                :disabled="dptEnergyBusy || dptEnergySaving || !dptEnergyAcquisitionCount(point.measurement_index)"
              >
                <option :value="null">Choose an acquisition</option>
                <option
                  v-for="index in dptEnergyAcquisitionCount(point.measurement_index)"
                  :key="index - 1"
                  :value="index - 1"
                >
                  Acquisition {{ index }}
                </option>
              </select>

              <label :for="`dpt-energy-current-${pointIndex}`">Explicit graph current i_x [A]</label>
              <input
                :id="`dpt-energy-current-${pointIndex}`"
                :data-testid="`dpt-energy-current-${pointIndex}`"
                v-model.number="point.i_x"
                type="number"
                min="0"
                step="any"
                :disabled="dptEnergyBusy || dptEnergySaving"
              />

              <label :for="`dpt-energy-vds-sign-${pointIndex}`">Vds polarity multiplier</label>
              <select
                :id="`dpt-energy-vds-sign-${pointIndex}`"
                :data-testid="`dpt-energy-vds-sign-${pointIndex}`"
                v-model.number="point.vds_sign"
                :disabled="dptEnergyBusy || dptEnergySaving"
              >
                <option :value="null">Choose Vds sign</option>
                <option :value="1">+1 · use stored Vds sign</option>
                <option :value="-1">−1 · reverse stored Vds sign</option>
              </select>

              <label :for="`dpt-energy-id-sign-${pointIndex}`">Id polarity multiplier</label>
              <select
                :id="`dpt-energy-id-sign-${pointIndex}`"
                :data-testid="`dpt-energy-id-sign-${pointIndex}`"
                v-model.number="point.id_sign"
                :disabled="dptEnergyBusy || dptEnergySaving"
              >
                <option :value="null">Choose Id sign</option>
                <option :value="1">+1 · use stored Id sign</option>
                <option :value="-1">−1 · reverse stored Id sign</option>
              </select>

              <button
                v-if="dptEnergyForm.points.length > 2"
                type="button"
                class="btn-remove-energy-point"
                :data-testid="`dpt-energy-remove-${pointIndex}`"
                :disabled="dptEnergyBusy || dptEnergySaving"
                @click="removeDptEnergyPoint(pointIndex)"
              >
                Remove event
              </button>
            </div>

            <p v-if="!dptEnergyMeasurementOptions.length" class="no-data" role="status">
              No stored paired {{ dptEnergyForm.loss_type === 'e_on' ? 'turn-on' : 'turn-off' }}
              Vds and Id acquisitions are available.
            </p>
            <p v-if="dptEnergyFormError" class="raw-import-error" role="alert">
              {{ dptEnergyFormError }}
            </p>

            <div class="dpt-energy-actions">
              <button
                type="button"
                data-testid="dpt-energy-add-point"
                :disabled="dptEnergyForm.points.length >= 50 || dptEnergyBusy || dptEnergySaving"
                @click="addDptEnergyPoint"
              >
                Add event
              </button>
              <button
                type="button"
                data-testid="dpt-energy-preview"
                :disabled="!dptEnergyCanPreview || dptEnergyBusy || dptEnergySaving"
                @click="previewDptEnergy"
              >
                {{ dptEnergyBusy ? 'Calculating…' : 'Preview raw-window energy' }}
              </button>
            </div>

            <p v-if="dptEnergyError" class="raw-import-error" role="alert">
              {{ dptEnergyError }}
            </p>
            <p v-if="dptEnergyStatus" class="raw-import-status" role="status">
              {{ dptEnergyStatus }}
            </p>

            <div
              v-if="dptEnergyPreview && dptEnergyReviewIsCurrent"
              class="dpt-energy-preview"
              data-testid="dpt-energy-preview-result"
            >
              <h5>Review raw-window result</h5>
              <p>
                {{ dptEnergyPreview.save_action === 'extend'
                  ? `Extend existing curve #${dptEnergyPreview.existing_curve_index + 1}`
                  : 'Create a new curve' }}
                · {{ dptEnergyPreview.loss_type === 'e_on' ? 'E_on' : 'E_off' }}
                · {{ dptEnergyPreview.integration.quantity }}
              </p>
              <p>
                {{ dptEnergyPreview.integration.formula }};
                {{ dptEnergyPreview.integration.power_formula }}.
                Integration interval: {{ dptEnergyPreview.integration.interval }}.
              </p>
              <table>
                <thead>
                  <tr>
                    <th>Source</th><th>i_x [A]</th><th>Energy [J]</th>
                    <th>Exact interval [s]</th><th>Samples</th><th>Signs (Vds, Id)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="point in dptEnergyPreview.points" :key="`${point.measurement_index}-${point.acquisition_index}`">
                    <td>Measurement {{ point.measurement_index + 1 }}, acquisition {{ point.acquisition_index + 1 }}</td>
                    <td>{{ point.i_x }}</td>
                    <td>{{ point.energy_j }}</td>
                    <td>[{{ point.start_s }}, {{ point.end_s }}]</td>
                    <td>{{ point.sample_count }}</td>
                    <td>{{ point.vds_sign > 0 ? '+' : '−' }}1, {{ point.id_sign > 0 ? '+' : '−' }}1</td>
                  </tr>
                </tbody>
              </table>
              <dl class="dpt-energy-conditions">
                <div v-for="(value, key) in dptEnergyPreview.conditions" :key="key">
                  <dt>{{ key }}</dt><dd>{{ value === null ? 'not recorded' : value }}</dd>
                </div>
              </dl>
              <label class="dpt-energy-confirm">
                <input
                  v-model="dptEnergyConfirmed"
                  data-testid="dpt-energy-confirm"
                  type="checkbox"
                  :disabled="dptEnergySaving"
                />
                I reviewed these raw-window measurements and confirm saving this curve.
              </label>
              <button
                type="button"
                class="btn-save-energy"
                data-testid="dpt-energy-save"
                :disabled="!dptEnergyConfirmed || !dptEnergyReviewIsCurrent || dptEnergySaving"
                @click="saveDptEnergy"
              >
                {{ dptEnergySaving ? 'Saving…' : 'Save reviewed energy curve' }}
              </button>
            </div>
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
    const rawImportStepForms = ref({})
    const dptEnergyForm = ref({ loss_type: 'e_on', points: [emptyDptEnergyPoint(), emptyDptEnergyPoint()] })
    const dptEnergyPreview = ref(null)
    const dptEnergyPreviewSignature = ref('')
    const dptEnergyBusy = ref(false)
    const dptEnergySaving = ref(false)
    const dptEnergyConfirmed = ref(false)
    const dptEnergyError = ref('')
    const dptEnergyStatus = ref('')
    const dptEnergyRequestVersion = ref(0)

    function emptyDptEnergyPoint() {
      return {
        id: `${Date.now()}-${Math.random()}`,
        measurement_index: null,
        acquisition_index: null,
        i_x: '',
        vds_sign: null,
        id_sign: null
      }
    }

    function invalidateDptEnergyReview({ clearStatus = true } = {}) {
      dptEnergyRequestVersion.value += 1
      dptEnergyPreview.value = null
      dptEnergyPreviewSignature.value = ''
      dptEnergyConfirmed.value = false
      dptEnergyError.value = ''
      if (clearStatus) dptEnergyStatus.value = ''
      dptEnergyBusy.value = false
    }

    const dptEnergyInputSignature = computed(() => JSON.stringify({
      transistor_id: selectedTransistorId.value,
      loss_type: dptEnergyForm.value.loss_type,
      points: dptEnergyForm.value.points.map(({ measurement_index, acquisition_index, i_x, vds_sign, id_sign }) => ({
        measurement_index,
        acquisition_index,
        i_x,
        vds_sign,
        id_sign
      }))
    }))

    const dptEnergyReviewIsCurrent = computed(() => (
      Boolean(dptEnergyPreview.value)
      && dptEnergyPreviewSignature.value === dptEnergyInputSignature.value
    ))

    watch(dptEnergyInputSignature, () => {
      invalidateDptEnergyReview()
    }, { flush: 'sync' })

    const dptEnergyMeasurementOptions = computed(() => {
      const suffix = dptEnergyForm.value.loss_type === 'e_on' ? 'on' : 'off'
      return rawMeasurementRecords.value.flatMap((measurement, index) => {
        const count = dptEnergyAcquisitionCount(index)
        if (!count) return []
        return [{
          index,
          label: `${measurementLabel(measurement, index)} · ${count} paired ${suffix} acquisition${count === 1 ? '' : 's'}`
        }]
      })
    })

    function dptEnergyAcquisitionCount(measurementIndex) {
      if (!Number.isInteger(measurementIndex)) return 0
      const measurement = rawMeasurementRecords.value[measurementIndex]
      if (!measurement) return 0
      const suffix = dptEnergyForm.value.loss_type === 'e_on' ? 'on' : 'off'
      const vds = measurement[`dpt_${suffix}_vds`]
      const current = measurement[`dpt_${suffix}_id`]
      return Array.isArray(vds) && Array.isArray(current)
        ? Math.min(vds.length, current.length)
        : 0
    }

    const dptEnergyRequest = computed(() => {
      const points = dptEnergyForm.value.points
      if (points.length < 2 || points.length > 50) return null
      const identities = new Set()
      const currents = new Set()
      const requestPoints = []
      for (const point of points) {
        if (
          !Number.isInteger(point.measurement_index)
          || !Number.isInteger(point.acquisition_index)
          || point.acquisition_index < 0
          || point.acquisition_index >= dptEnergyAcquisitionCount(point.measurement_index)
          || !Number.isFinite(point.i_x)
          || point.i_x <= 0
          || ![1, -1].includes(point.vds_sign)
          || ![1, -1].includes(point.id_sign)
        ) return null
        const identity = `${point.measurement_index}:${point.acquisition_index}`
        if (identities.has(identity) || currents.has(point.i_x)) return null
        identities.add(identity)
        currents.add(point.i_x)
        requestPoints.push({
          measurement_index: point.measurement_index,
          acquisition_index: point.acquisition_index,
          i_x: point.i_x,
          vds_sign: point.vds_sign,
          id_sign: point.id_sign
        })
      }
      return { loss_type: dptEnergyForm.value.loss_type, points: requestPoints }
    })

    const dptEnergyFormError = computed(() => {
      if (dptEnergyForm.value.points.length < 2) return 'Select at least two event windows.'
      if (dptEnergyForm.value.points.some(point => (
        Number.isFinite(point.i_x) && point.i_x <= 0
      ))) return 'Every graph current i_x must be greater than zero.'
      const points = dptEnergyForm.value.points
      const identities = points.map(point => `${point.measurement_index}:${point.acquisition_index}`)
      if (identities.some((identity, index) => identities.indexOf(identity) !== index)
        && points.every(point => Number.isInteger(point.measurement_index) && Number.isInteger(point.acquisition_index))) {
        return 'Choose each stored event acquisition only once.'
      }
      const currents = points.map(point => point.i_x)
      if (currents.every(Number.isFinite) && currents.some((value, index) => currents.indexOf(value) !== index)) {
        return 'Every event needs a distinct i_x value.'
      }
      if (dptEnergyForm.value.points.some(point => (
        !Number.isInteger(point.measurement_index)
        || !Number.isInteger(point.acquisition_index)
        || !Number.isFinite(point.i_x)
        || point.i_x <= 0
        || ![1, -1].includes(point.vds_sign)
        || ![1, -1].includes(point.id_sign)
      ))) {
        return 'Choose a stored event, enter a positive i_x value, and choose both polarity signs for every event.'
      }
      return ''
    })

    const dptEnergyCanPreview = computed(() => (
      Boolean(selectedTransistorId.value) && Boolean(dptEnergyRequest.value)
    ))

    function addDptEnergyPoint() {
      if (dptEnergyForm.value.points.length >= 50) return
      dptEnergyForm.value.points.push(emptyDptEnergyPoint())
    }

    function removeDptEnergyPoint(index) {
      if (dptEnergyForm.value.points.length <= 2) return
      dptEnergyForm.value.points.splice(index, 1)
    }

    async function previewDptEnergy() {
      const request = dptEnergyRequest.value
      if (!request || !selectedTransistorId.value) return
      invalidateDptEnergyReview()
      const requestVersion = dptEnergyRequestVersion.value
      const transistorId = selectedTransistorId.value
      const signature = dptEnergyInputSignature.value
      dptEnergyBusy.value = true
      dptEnergyError.value = ''
      dptEnergyStatus.value = ''
      try {
        const preview = await transistorApi.previewDPTEnergy(transistorId, request)
        if (
          requestVersion !== dptEnergyRequestVersion.value
          || signature !== dptEnergyInputSignature.value
        ) return
        dptEnergyPreview.value = preview
        dptEnergyPreviewSignature.value = signature
        dptEnergyStatus.value = 'Review every event interval, current-axis value, sign, energy, and condition before confirming save.'
      } catch (err) {
        if (requestVersion === dptEnergyRequestVersion.value) {
          dptEnergyError.value = err?.response?.data?.detail || 'Failed to preview raw-window energy.'
        }
      } finally {
        if (requestVersion === dptEnergyRequestVersion.value) dptEnergyBusy.value = false
      }
    }

    async function saveDptEnergy() {
      if (!dptEnergyReviewIsCurrent.value || !dptEnergyConfirmed.value) return
      const review = dptEnergyPreview.value
      const request = dptEnergyRequest.value
      if (!request || !selectedTransistorId.value) return
      const transistorId = selectedTransistorId.value
      dptEnergySaving.value = true
      dptEnergyError.value = ''
      dptEnergyStatus.value = ''
      try {
        const saved = await transistorApi.saveDPTEnergy(transistorId, {
          ...request,
          source_token: review.source_token,
          request_token: review.request_token,
          save_action: review.save_action,
          existing_curve_index: review.existing_curve_index,
          target_curve_count: review.target_curve_count,
          target_edit_token: review.target_edit_token
        })
        const savedStatus = `${saved.action === 'extended' ? 'Extended' : 'Added'} ${saved.loss_type === 'e_on' ? 'E_on' : 'E_off'} curve #${saved.curve_index + 1} with ${saved.point_count} measured points.`
        invalidateDptEnergyReview()
        await loadDPTData()
        dptEnergyStatus.value = savedStatus
      } catch (err) {
        const detail = err?.response?.data?.detail
        if (err?.response?.status === 409) invalidateDptEnergyReview()
        dptEnergyError.value = detail || 'Failed to save the reviewed energy curve.'
      } finally {
        dptEnergySaving.value = false
      }
    }

    function emptyRawImportForm() {
      return {
        plot_index: '',
        step_indices: [],
        same_context_confirmed: false,
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

    function invalidateRawImportReview({
      resetWindows = true,
      clearTracePreview = true
    } = {}) {
      rawImportRequestVersion.value += 1
      rawImportBusy.value = false
      if (clearTracePreview) rawImportTracePreview.value = null
      rawImportReviewedState.value = null
      if (resetWindows) {
        rawImportForm.value = {
          ...rawImportForm.value,
          turn_on_start_s: '',
          turn_on_end_s: '',
          turn_off_start_s: '',
          turn_off_end_s: '',
          same_context_confirmed: false
        }
        rawImportStepForms.value = Object.fromEntries(
          Object.keys(rawImportStepForms.value).map(index => [
            index,
            emptyRawImportStepForm()
          ])
        )
      }
    }

    function emptyRawImportStepForm() {
      return {
        i_x: '',
        turn_on_start_s: '',
        turn_on_end_s: '',
        turn_off_start_s: '',
        turn_off_end_s: ''
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
      rawImportStepForms.value = {}
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
        plotIndex: rawImportForm.value.plot_index,
        stepIndices: [...(rawImportForm.value.step_indices || [])].join(','),
        timeTrace: rawImportSelectedPlot.value?.trace_names?.[0] || null,
        vdsTrace: rawImportForm.value.vds_trace,
        idTrace: rawImportForm.value.id_trace,
        inputSignature: rawImportInputSignature()
      }
    }

    function rawImportInputSignature() {
      return JSON.stringify({
        eventWindows: [
          rawImportForm.value.turn_on_start_s,
          rawImportForm.value.turn_on_end_s,
          rawImportForm.value.turn_off_start_s,
          rawImportForm.value.turn_off_end_s
        ],
        stepForms: rawImportStepForms.value,
        metadata: rawImportMetadata.value,
        sameContextConfirmed: rawImportForm.value.same_context_confirmed
      })
    }

    function rawImportSnapshotIsCurrent(snapshot) {
      return snapshot.version === rawImportRequestVersion.value
        && snapshot.file === rawImportFile.value
        && snapshot.transistorId === selectedTransistorId.value
        && snapshot.plotIndex === rawImportForm.value.plot_index
        && snapshot.stepIndices === [...(rawImportForm.value.step_indices || [])].join(',')
        && snapshot.timeTrace === (rawImportSelectedPlot.value?.trace_names?.[0] || null)
        && snapshot.vdsTrace === rawImportForm.value.vds_trace
        && snapshot.idTrace === rawImportForm.value.id_trace
        && snapshot.inputSignature === rawImportInputSignature()
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
        && review.plotIndex === rawImportForm.value.plot_index
        && review.stepIndices === [...(rawImportForm.value.step_indices || [])].join(',')
        && review.timeTrace === rawImportSelectedPlot.value?.trace_names?.[0]
        && review.vdsTrace === rawImportForm.value.vds_trace
        && review.idTrace === rawImportForm.value.id_trace
        && review.inputSignature === rawImportInputSignature()
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
    const rawImportSelectedPlot = computed(() => {
      if (!rawImportHeaderIsCurrent() || rawImportForm.value.plot_index === '') return null
      return rawImportPreview.value?.plots?.find(
        plot => plot.plot_index === Number(rawImportForm.value.plot_index)
      ) || null
    })
    const rawImportRequiresExplicitContext = computed(() => (
      Boolean(rawImportSelectedPlot.value)
      && (rawImportSelectedPlot.value.is_stepped
        || (rawImportPreview.value?.plots?.length || 0) > 1)
    ))
    const rawImportSelectedSteps = computed(() => {
      const selected = new Set(
        (rawImportForm.value.step_indices || []).map(Number)
      )
      return (rawImportSelectedPlot.value?.steps || [])
        .filter(step => selected.has(step.step_index))
    })
    const rawImportValueTraces = computed(() => (
      rawImportSelectedPlot.value?.trace_names?.filter(
        trace => trace !== rawImportSelectedPlot.value.trace_names[0]
      ) || []
    ))
    const rawImportTraceMappingReady = computed(() => (
      Boolean(rawImportFile.value)
      && Boolean(rawImportSelectedPlot.value?.is_transient)
      && Boolean(rawImportSelectedPlot.value?.trace_names?.[0])
      && Boolean(rawImportForm.value.vds_trace)
      && Boolean(rawImportForm.value.id_trace)
      && rawImportForm.value.vds_trace !== rawImportForm.value.id_trace
      && (!rawImportRequiresExplicitContext.value
        || rawImportSelectedSteps.value.length > 0)
    ))
    const rawImportWindowInputsValid = computed(() => {
      if (!rawImportTraceMappingReady.value) return false
      if (rawImportRequiresExplicitContext.value) {
        const numericWindowIsValid = (start, end) => (
          start !== '' && end !== ''
          && Number.isFinite(Number(start))
          && Number.isFinite(Number(end))
          && Number(end) >= Number(start)
        )
        const numericConditionFields = [
          't_j', 'v_supply', 'v_g', 'v_g_off', 'r_g', 'r_g_off',
          'load_inductance', 'commutation_inductance'
        ]
        const completeConditions = numericConditionFields.every(key => (
          rawImportMetadata.value[key] !== ''
          && rawImportMetadata.value[key] !== null
          && rawImportMetadata.value[key] !== undefined
          && Number.isFinite(Number(rawImportMetadata.value[key]))
        )) && ['measurement_testbench', 'commutation_device'].every(key => (
          String(rawImportMetadata.value[key] || '').trim() !== ''
        ))
        return rawImportForm.value.same_context_confirmed
          && completeConditions
          && rawImportSelectedSteps.value.length <= 50
          && rawImportSelectedSteps.value.every(step => {
            const form = rawImportStepForms.value[step.step_index]
            if (!form || !Number.isFinite(Number(form.i_x)) || Number(form.i_x) <= 0) {
              return false
            }
            return numericWindowIsValid(form.turn_on_start_s, form.turn_on_end_s)
              && numericWindowIsValid(form.turn_off_start_s, form.turn_off_end_s)
          })
      }
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
    const rawImportSelectionValid = computed(() => (
      rawImportReviewIsCurrent()
      && rawImportWindowInputsValid.value
      && (!rawImportRequiresExplicitContext.value
        || Boolean(rawImportReviewedState.value?.reviewToken))
    ))
    const rawImportHasNonFiniteSamples = computed(() => (
      Object.values(rawImportTracePreview.value?.steps || {})
        .flatMap(step => Object.values(step.non_finite_preview_sample_counts || {}))
        .concat(Object.values(rawImportTracePreview.value?.non_finite_preview_sample_counts || {}))
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
        plot_index: Number(rawImportForm.value.plot_index),
        step_indices: (rawImportForm.value.step_indices || []).map(Number),
        time_trace: rawImportSelectedPlot.value.trace_names[0],
        vds_trace: rawImportForm.value.vds_trace,
        id_trace: rawImportForm.value.id_trace
      }
      if (includeWindows) {
        if (rawImportRequiresExplicitContext.value) {
          request.step_selections = rawImportSelectedSteps.value.map(step => {
            const form = rawImportStepForms.value[step.step_index]
            return {
              step_index: step.step_index,
              i_x: Number(form.i_x),
              turn_on: {
                start_s: Number(form.turn_on_start_s),
                end_s: Number(form.turn_on_end_s)
              },
              turn_off: {
                start_s: Number(form.turn_off_start_s),
                end_s: Number(form.turn_off_end_s)
              }
            }
          })
          request.same_context_confirmed = rawImportForm.value.same_context_confirmed
        } else {
          request.turn_on = {
            start_s: Number(rawImportForm.value.turn_on_start_s),
            end_s: Number(rawImportForm.value.turn_on_end_s)
          }
          request.turn_off = {
            start_s: Number(rawImportForm.value.turn_off_start_s),
            end_s: Number(rawImportForm.value.turn_off_end_s)
          }
        }
        request.metadata = metadata
      }
      if (rawImportReviewedState.value?.reviewToken) {
        request.review_token = rawImportReviewedState.value.reviewToken
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
      rawImportStepForms.value = {}
    }

    function selectRawImportPlot() {
      const plotIndex = rawImportForm.value.plot_index
      const plot = rawImportPreview.value?.plots?.find(
        item => item.plot_index === Number(plotIndex)
      )
      const chooseSingleSourceStep = Boolean(
        plot && !plot.is_stepped && rawImportPreview.value.plots.length > 1
      )
      invalidateRawImportReview()
      rawImportForm.value = {
        ...emptyRawImportForm(),
        plot_index: plotIndex,
        step_indices: chooseSingleSourceStep ? ['0'] : []
      }
      rawImportStepForms.value = chooseSingleSourceStep
        ? { 0: emptyRawImportStepForm() }
        : {}
      rawImportStatus.value = ''
      rawImportError.value = ''
    }

    function selectRawImportSteps() {
      const plot = rawImportSelectedPlot.value
      const selectedIndexes = new Set(
        (rawImportForm.value.step_indices || []).map(Number)
      )
      const selectedSteps = (plot?.steps || []).filter(step => (
        selectedIndexes.has(step.step_index)
      ))
      rawImportForm.value = {
        ...rawImportForm.value,
        step_indices: selectedSteps.map(step => String(step.step_index)),
        same_context_confirmed: false
      }
      const forms = {}
      for (const step of selectedSteps) {
        forms[step.step_index] = rawImportStepForms.value[step.step_index]
          || emptyRawImportStepForm()
      }
      rawImportStepForms.value = forms
      invalidateRawImportReview({ resetWindows: false })
      rawImportStatus.value = ''
      rawImportError.value = ''
    }

    function selectRawImportContext() {
      invalidateRawImportReview({ resetWindows: false, clearTracePreview: false })
      rawImportStatus.value = ''
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
        const plots = Array.isArray(preview.plots)
          ? preview.plots
          : [{
              plot_index: 0,
              plot_name: preview.plot_name,
              sample_count: preview.sample_count,
              trace_names: preview.trace_names || [],
              is_transient: true,
              is_stepped: false,
              steps: []
            }]
        rawImportPreview.value = { ...preview, plots }
        rawImportForm.value = {
          ...emptyRawImportForm(),
          plot_index: plots.length === 1 ? String(plots[0].plot_index) : ''
        }
        rawImportStepForms.value = {}
        rawImportHeaderState.value = {
          file: snapshot.file,
          transistorId: snapshot.transistorId
        }
        rawImportStatus.value = plots.length > 1
          ? 'File header read. Choose a transient plot, then its Vds and Id traces.'
          : 'File header read. Choose the Vds and Id traces.'
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
      invalidateRawImportReview({ resetWindows: false })
      const snapshot = rawImportRequestSnapshot()
      const request = rawImportRequest(rawImportWindowInputsValid.value)
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
          idTrace: snapshot.idTrace,
          plotIndex: snapshot.plotIndex,
          stepIndices: snapshot.stepIndices,
          inputSignature: snapshot.inputSignature,
          reviewToken: preview.review_token || null
        }
        rawImportStatus.value = preview.review_token
          ? 'The exact file, traces, source steps, event windows, currents, and conditions are reviewed. Confirm the append below.'
          : 'Trace preview loaded. Review the plots and enter each event window.'
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
      const stepPreviews = rawImportTracePreview.value?.steps || []
      const stepColors = [
        '#2563eb', '#dc2626', '#059669', '#7c3aed', '#ea580c',
        '#0891b2', '#be123c', '#4d7c0f', '#9333ea', '#0f766e'
      ]
      if (stepPreviews.length > 1 || rawImportSelectedPlot.value?.is_stepped) {
        return stepPreviews.map(step => ({
          label: `Step ${step.step_index} · ${title}`,
          data: (step.traces?.[trace] || []).map(([time, value]) => ({
            x: time * 1e6,
            y: value
          })),
          borderColor: stepColors[step.step_index % stepColors.length] || color,
          pointRadius: 0
        }))
      }
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
          result.acquisition_count > 1
            ? `Added one DPT record with ${result.acquisition_count} selected step acquisitions.`
            : `Added one DPT record with ${result.on_sample_count} turn-on and `
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
          const detail = rawImportErrorMessage(err)
          if (err?.response?.status === 409) {
            invalidateRawImportReview({ resetWindows: false, clearTracePreview: false })
            rawImportStatus.value = ''
          }
          rawImportError.value = detail
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
        invalidateDptEnergyReview()
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

    watch(
      () => [
        rawImportForm.value.turn_on_start_s,
        rawImportForm.value.turn_on_end_s,
        rawImportForm.value.turn_off_start_s,
        rawImportForm.value.turn_off_end_s,
        rawImportForm.value.same_context_confirmed
      ],
      () => {
        if (!rawImportTracePreview.value && !rawImportReviewedState.value) return
        invalidateRawImportReview({ resetWindows: false, clearTracePreview: false })
        rawImportStatus.value = 'Import inputs changed. Preview selected traces again to review them.'
      },
      { flush: 'sync' }
    )

    watch(
      rawImportStepForms,
      () => {
        if (!rawImportTracePreview.value && !rawImportReviewedState.value) return
        invalidateRawImportReview({ resetWindows: false, clearTracePreview: false })
        rawImportStatus.value = 'Step current or windows changed. Preview selected traces again to review them.'
      },
      { deep: true, flush: 'sync' }
    )

    watch(
      rawImportMetadata,
      () => {
        if (!rawImportTracePreview.value && !rawImportReviewedState.value) return
        invalidateRawImportReview({ resetWindows: false, clearTracePreview: false })
        rawImportStatus.value = 'Measurement conditions changed. Preview selected traces again to review them.'
      },
      { deep: true, flush: 'sync' }
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
        invalidateDptEnergyReview()
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
      rawImportStepForms,
      rawImportSelectedPlot,
      rawImportSelectedSteps,
      rawImportRequiresExplicitContext,
      dptEnergyForm,
      dptEnergyPreview,
      dptEnergyBusy,
      dptEnergySaving,
      dptEnergyConfirmed,
      dptEnergyError,
      dptEnergyStatus,
      dptEnergyMeasurementOptions,
      dptEnergyAcquisitionCount,
      dptEnergyFormError,
      dptEnergyCanPreview,
      dptEnergyReviewIsCurrent,
      addDptEnergyPoint,
      removeDptEnergyPoint,
      previewDptEnergy,
      saveDptEnergy,
      rawImportValueTraces,
      rawImportTraceMappingReady,
      rawImportWindowInputsValid,
      rawImportSelectionValid,
      rawImportHasNonFiniteSamples,
      rawImportVdsPreviewDatasets,
      rawImportIdPreviewDatasets,
      selectRawImportFile,
      selectRawImportPlot,
      selectRawImportSteps,
      selectRawImportContext,
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

.dpt-energy-section {
  margin-top: 28px;
  padding: 20px;
  border: 1px solid #d7dee8;
  border-radius: 8px;
  background: #fbfcfe;
}

.dpt-energy-section h4 {
  margin: 0 0 12px;
}

.dpt-energy-section code {
  color: #24344d;
  font-weight: 600;
}

.dpt-energy-form,
.dpt-energy-point {
  display: grid;
  grid-template-columns: minmax(170px, 240px) minmax(220px, 1fr);
  align-items: center;
  gap: 10px 14px;
}

.dpt-energy-form {
  margin: 18px 0;
  max-width: 560px;
}

.dpt-energy-point {
  position: relative;
  margin: 14px 0;
  padding: 16px;
  border: 1px solid #e1e6ee;
  border-radius: 6px;
  background: white;
}

.dpt-energy-point h5 {
  grid-column: 1 / -1;
  margin: 0 0 4px;
}

.dpt-energy-point input,
.dpt-energy-point select,
.dpt-energy-form select {
  min-width: 0;
  padding: 7px 8px;
  border: 1px solid #bcc6d4;
  border-radius: 4px;
}

.btn-remove-energy-point {
  grid-column: 2;
  justify-self: start;
}

.dpt-energy-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 14px 0;
}

.dpt-energy-preview {
  overflow-x: auto;
  margin-top: 18px;
  padding: 16px;
  border-left: 4px solid #2563eb;
  background: white;
}

.dpt-energy-preview h5 {
  margin: 0 0 10px;
}

.dpt-energy-preview table {
  width: 100%;
  margin: 12px 0;
  border-collapse: collapse;
  text-align: left;
}

.dpt-energy-preview th,
.dpt-energy-preview td {
  padding: 8px;
  border-bottom: 1px solid #e1e6ee;
  white-space: nowrap;
}

.dpt-energy-conditions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 8px 16px;
}

.dpt-energy-conditions div {
  display: grid;
  grid-template-columns: minmax(90px, 0.7fr) minmax(100px, 1fr);
  gap: 8px;
}

.dpt-energy-conditions dt {
  font-weight: 600;
}

.dpt-energy-conditions dd {
  margin: 0;
  overflow-wrap: anywhere;
}

.dpt-energy-confirm {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin: 14px 0;
}

.btn-save-energy {
  padding: 9px 14px;
  border: 0;
  border-radius: 4px;
  background: #155e34;
  color: white;
}

.btn-save-energy:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

@media (max-width: 680px) {
  .dpt-energy-form,
  .dpt-energy-point {
    grid-template-columns: 1fr;
  }

  .dpt-energy-point h5,
  .btn-remove-energy-point {
    grid-column: 1;
  }
}
</style>
