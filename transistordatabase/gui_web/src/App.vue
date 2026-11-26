<script setup>
import { ref, shallowRef, onMounted, onBeforeUnmount, nextTick } from 'vue'
import TransistorForm from './components/TransistorForm.vue'
import TransistorComparison from './components/TransistorComparison.vue'
import TransistorComparisonEnhanced from './components/TransistorComparisonEnhanced.vue'
import SearchDatabase from './components/SearchDatabase.vue'
import ExportingTools from './components/ExportingTools.vue'
import BatchExportPanel from './components/BatchExportPanel.vue'
import TopologyCalculator from './components/TopologyCalculator.vue'
import PLECSImporter from './components/PLECSImporter.vue'
import AnalyticalCalculator from './components/AnalyticalCalculator.vue'
import DPTManager from './components/DPTManager.vue'
import DatabaseBackupManager from './components/DatabaseBackupManager.vue'
import CatalogFom from './components/CatalogFom.vue'
import CatalogSourceWorkbench from './components/CatalogSourceWorkbench.vue'
import WaveformWorkbench from './components/WaveformWorkbench.vue'
import ThermalWorkbench from './components/ThermalWorkbench.vue'
import GridWorkbench from './components/GridWorkbench.vue'
import EnergyMapWorkbench from './components/EnergyMapWorkbench.vue'
import SemiconductorAssemblyWorkbench from './components/SemiconductorAssemblyWorkbench.vue'
import ConverterWorkbench from './components/ConverterWorkbench.vue'
import PlecsWorkbench from './components/PlecsWorkbench.vue'
import DigitizerWorkbench from './components/DigitizerWorkbench.vue'
import LegacyMatWorkbench from './components/LegacyMatWorkbench.vue'
import MeasurementWorkbench from './components/MeasurementWorkbench.vue'
import StressIndicesWorkbench from './components/StressIndicesWorkbench.vue'
import { transistorApi } from './services/api.js'

const initialView = new URLSearchParams(window.location.search).get('view')
const currentView = ref(['catalog-fom', 'catalog-sources', 'waveform', 'thermal', 'analysis-grid', 'energy-map', 'assemblies', 'converter', 'plecs-data', 'dpt', 'compare-advanced', 'digitizer', 'legacy-mat', 'measurements', 'stress-indices', 'analytical'].includes(initialView) ? initialView : 'search')
const stagedThermal = shallowRef(null)
function stageThermal(handoff) {
  stagedThermal.value = JSON.parse(JSON.stringify(handoff))
  currentView.value = 'thermal'
}
function navigateWorkbench(view) {
  if (['plecs-data', 'dpt'].includes(view)) currentView.value = view
}
const suppliedCatalog = shallowRef(null)
function exploreCatalog(catalog) {
  suppliedCatalog.value = catalog
  currentView.value = 'catalog-fom'
}
const transistors = ref([])
const selectedTransistor = ref(null)
const selectedForExport = ref([])
const selectedForTopology = ref(null)
const isLoading = ref(false)
const isDarkTheme = ref(false)
const showMoreMenu = ref(false)
const operationMessage = ref(null)
const moreMenuTrigger = ref(null), moreMenuPopup = ref(null), moreMenuStyle = ref({})
let moreMenuAnchor = null
function closeMoreMenu(returnFocus = false) {
  showMoreMenu.value = false
  if (returnFocus === true) moreMenuTrigger.value?.focus({ preventScroll: true })
}
async function toggleMoreMenu() {
  if (showMoreMenu.value) { closeMoreMenu(); return }
  const box = moreMenuTrigger.value.getBoundingClientRect(), margin = 8
  moreMenuAnchor = { left: box.left, top: box.top }
  const below = window.innerHeight - box.bottom - margin, above = box.top - margin
  const openBelow = below >= Math.min(240, above)
  const height = Math.min(560, openBelow ? below : above), width = Math.min(270, window.innerWidth - margin * 2)
  moreMenuStyle.value = { left: `${Math.max(margin, Math.min(box.left, window.innerWidth - width - margin))}px`, width: `${width}px`, maxHeight: `${Math.max(1, height)}px`,
    ...(openBelow ? { top: `${box.bottom}px` } : { bottom: `${window.innerHeight - box.top}px` }) }
  showMoreMenu.value = true
  await nextTick()
  moreMenuPopup.value?.querySelector('button')?.focus({ preventScroll: true })
}
function outsideMenu(event) {
  if (!moreMenuTrigger.value?.contains(event.target) && !moreMenuPopup.value?.contains(event.target)) closeMoreMenu()
}
function menuKey(event) { if (event.key === 'Escape' && showMoreMenu.value) { event.preventDefault(); closeMoreMenu(true) } }
function menuScroll(event) {
  if (!showMoreMenu.value || moreMenuPopup.value?.contains(event.target)) return
  if (event.target === moreMenuTrigger.value?.closest('nav')?.querySelector('.workspace-tabs')) {
    const box = moreMenuTrigger.value.getBoundingClientRect()
    if (moreMenuAnchor && Math.abs(box.left - moreMenuAnchor.left) < .5 && Math.abs(box.top - moreMenuAnchor.top) < .5) return
  }
  closeMoreMenu()
}
onMounted(() => {
  document.addEventListener('pointerdown', outsideMenu)
  document.addEventListener('keydown', menuKey)
  document.addEventListener('scroll', menuScroll, true)
  window.addEventListener('resize', closeMoreMenu)
})
onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', outsideMenu)
  document.removeEventListener('keydown', menuKey)
  document.removeEventListener('scroll', menuScroll, true)
  window.removeEventListener('resize', closeMoreMenu)
})

onMounted(async () => {
  console.log('[App.vue] Component mounted, loading transistors...')
  // Load theme preference
  const savedTheme = localStorage.getItem('darkTheme')
  if (savedTheme !== null) {
    isDarkTheme.value = JSON.parse(savedTheme)
  }
  applyTheme()
  await loadTransistors()
  console.log('[App.vue] Transistors loaded, count:', transistors.value.length)
  console.log('[App.vue] Initialization complete')
})

async function loadTransistors() {
  console.log('[loadTransistors] Starting...')
  isLoading.value = true
  try {
    const data = await transistorApi.getAll()
    console.log('[loadTransistors] API returned:', data)
    console.log('[loadTransistors] Data type:', typeof data, 'Is Array:', Array.isArray(data))
    console.log('[loadTransistors] Data length:', data?.length)
    transistors.value = data
    console.log('[loadTransistors] State updated, transistors.value.length:', transistors.value.length)
    return true
  } catch (error) {
    console.error('[loadTransistors] Error:', error)
    console.error('[loadTransistors] Error details:', {
      message: error.message,
      response: error.response,
      stack: error.stack
    })
    return false
  } finally {
    isLoading.value = false
    console.log('[loadTransistors] Finished, isLoading:', isLoading.value)
  }
}

function toggleTheme() {
  isDarkTheme.value = !isDarkTheme.value
  localStorage.setItem('darkTheme', JSON.stringify(isDarkTheme.value))
  applyTheme()
}

function applyTheme() {
  if (isDarkTheme.value) {
    document.documentElement.classList.add('dark-theme')
  } else {
    document.documentElement.classList.remove('dark-theme')
  }
}

function showSearch() {
  currentView.value = 'search'
}

function showCreate() {
  operationMessage.value = null
  selectedTransistor.value = null
  currentView.value = 'create'
}

function showExport() {
  currentView.value = 'export'
}

function showComparison() {
  currentView.value = 'compare'
}

function showTopology() {
  currentView.value = 'topology'
}

function showAdvancedComparison() {
  currentView.value = 'compare-advanced'
}

function showBatchExport() {
  currentView.value = 'batch-export'
}

function showPLECSImport() {
  currentView.value = 'plecs-import'
}

function showAnalytical() {
  currentView.value = 'analytical'
}

function showDPT() {
  currentView.value = 'dpt'
}

function showDatabaseBackup() {
  currentView.value = 'database-backup'
}

function showEditForm(transistor) {
  operationMessage.value = null
  const name = transistor?.metadata?.name
  if (typeof name !== 'string' || !name || name.includes('/') || name.includes('\\')) {
    operationMessage.value = {
      type: 'error',
      text: 'Device names containing path separators cannot be edited in the web database.'
    }
    return
  }
  selectedTransistor.value = transistor
  currentView.value = 'create'
}

async function handleTransistorSaved() {
  const refreshed = await loadTransistors()
  currentView.value = 'search'
  operationMessage.value = refreshed
    ? { type: 'success', text: 'Transistor saved successfully.' }
    : { type: 'error', text: 'Transistor was saved, but the database list could not be refreshed.' }
}

async function handleTransistorDeleted(transistor) {
  const transistorId = typeof transistor === 'string'
    ? transistor
    : transistor?.metadata?.name
  if (!transistorId) {
    operationMessage.value = { type: 'error', text: 'Could not identify the transistor to delete.' }
    return
  }
  if (transistorId.includes('/') || transistorId.includes('\\')) {
    operationMessage.value = {
      type: 'error',
      text: 'Device names containing path separators cannot be deleted in the web database.'
    }
    return
  }

  if (!window.confirm(`Delete ${transistorId}? This cannot be undone.`)) return

  try {
    await transistorApi.delete(transistorId)
    const refreshed = await loadTransistors()
    operationMessage.value = refreshed
      ? { type: 'success', text: `${transistorId} was deleted.` }
      : { type: 'error', text: `${transistorId} was deleted, but the database list could not be refreshed.` }
  } catch (error) {
    console.error('Error deleting transistor:', error)
    operationMessage.value = {
      type: 'error',
      text: error.response?.data?.detail || error.message || `Could not delete ${transistorId}.`
    }
  }
}

// Handle inter-component communication
function handleLoadToExporting(transistor) {
  selectedForExport.value = [transistor]
  currentView.value = 'export'
}

function handleLoadToComparison(transistor) {
  // Let comparison component handle the transistor
  currentView.value = 'compare'
  // You could emit an event or use a store for more complex state management
}

function handleLoadToTopology(transistor) {
  selectedForTopology.value = transistor
  currentView.value = 'topology'
}

function handleTransistorSelected(transistor) {
  selectedTransistor.value = transistor
}

function handleOpenDatabaseSearch() {
  currentView.value = 'search'
}
</script>

<template>
  <div class="app-shell" :class="{ 'dark-theme': isDarkTheme }">
    <div class="app-toolbar">
      <header class="app-header">
        <h1>Transistor Database</h1>
        <div class="stats">
          <span>Total Transistors: {{ transistors.length }}</span>
          <span v-if="selectedTransistor" class="selected-device" :title="selectedTransistor.metadata.name">
            Selected: {{ selectedTransistor.metadata.name }}
          </span>
        </div>
      </header>
      <div class="contact-header">
        <div class="contact-links">
          <a href="https://github.com/tinix84/transistordatabase" target="_blank" title="GitHub">
            GitHub
          </a>
        </div>
        <button @click="toggleTheme" class="theme-toggle" :title="isDarkTheme ? 'Switch to Light Theme' : 'Switch to Dark Theme'">
          {{ isDarkTheme ? 'Light mode' : 'Dark mode' }}
        </button>
      </div>

      <nav class="main-nav" aria-label="Workspaces">
        <div class="workspace-tabs">
          <button
            @click="showSearch"
            :class="{ active: currentView === 'search' }"
            class="nav-btn"
          >
            Search Database
          </button>
          <button
            @click="showCreate"
            :class="{ active: currentView === 'create' }"
            class="nav-btn"
          >
            Create/Edit
          </button>
          <button
            @click="showExport"
            :class="{ active: currentView === 'export' }"
            class="nav-btn"
          >
            Export
          </button>
          <button
            @click="showComparison"
            :class="{ active: currentView === 'compare' }"
            class="nav-btn"
          >
            Compare
          </button>
          <button
            @click="showAdvancedComparison"
            :class="{ active: currentView === 'compare-advanced' }"
            class="nav-btn"
          >
            Advanced Compare
          </button>
          <button
            @click="showTopology"
            :class="{ active: currentView === 'topology' }"
            class="nav-btn"
          >
            Topology
          </button>
          <button
            @click="showDatabaseBackup"
            :class="{ active: currentView === 'database-backup' }"
            class="nav-btn"
          >
            Database Backup
          </button>
          <button @click="currentView = 'catalog-fom'" :class="{ active: currentView === 'catalog-fom' }" class="nav-btn">
            FOM &amp; Pareto
          </button>
          <button @click="currentView = 'catalog-sources'" :class="{ active: currentView === 'catalog-sources' }" class="nav-btn">Catalog source import</button>
          <button @click="currentView = 'legacy-mat'" :class="{ active: currentView === 'legacy-mat' }" class="nav-btn">Legacy MAT</button>
          <button @click="currentView = 'measurements'" :class="{ active: currentView === 'measurements' }" class="nav-btn">Measurement import</button>
        </div>
        <div class="nav-dropdown">
          <button
            ref="moreMenuTrigger"
            @click="toggleMoreMenu"
            :aria-expanded="showMoreMenu" aria-controls="more-tools-popup" aria-haspopup="true"
            :class="{ active: ['plecs-import', 'plecs-data', 'analytical', 'dpt', 'batch-export', 'waveform', 'thermal', 'analysis-grid', 'energy-map', 'assemblies', 'converter', 'digitizer', 'stress-indices'].includes(currentView) }"
            class="nav-btn"
          >
            More Tools <span aria-hidden="true">▾</span>
          </button>
          <Teleport to="body"><div v-if="showMoreMenu" ref="moreMenuPopup" id="more-tools-popup" class="dropdown-menu" :style="moreMenuStyle" aria-label="More tools">
            <div class="dropdown-group" role="group" aria-label="Analysis">
              <p class="dropdown-heading" aria-hidden="true">Analysis</p>
              <button @click="currentView = 'waveform'; showMoreMenu = false" class="dropdown-item">Waveform workbench</button>
              <button @click="currentView = 'stress-indices'; showMoreMenu = false" class="dropdown-item">Semiconductor stress indices</button>
              <button @click="currentView = 'analysis-grid'; showMoreMenu = false" class="dropdown-item">Analysis grid</button>
              <button @click="currentView = 'energy-map'; showMoreMenu = false" class="dropdown-item">Energy map</button>
              <button @click="currentView = 'thermal'; showMoreMenu = false" class="dropdown-item">Thermal workbench</button>
              <button @click="currentView = 'assemblies'; showMoreMenu = false" class="dropdown-item">Semiconductor assemblies</button>
              <button @click="currentView = 'converter'; showMoreMenu = false" class="dropdown-item">Converter loss workbench</button>
              <button @click="showAnalytical(); showMoreMenu = false" class="dropdown-item">
                Analytical Models
              </button>
            </div>
            <div class="dropdown-group" role="group" aria-label="Data and models">
              <p class="dropdown-heading" aria-hidden="true">Data and models</p>
              <button @click="currentView = 'digitizer'; showMoreMenu = false" class="dropdown-item">Image digitizer</button>
              <button @click="currentView = 'plecs-data'; showMoreMenu = false" class="dropdown-item">PLECS data workbench</button>
              <button @click="showDPT(); showMoreMenu = false" class="dropdown-item">
                DPT Manager
              </button>
            </div>
            <div class="dropdown-group" role="group" aria-label="Import and export">
              <p class="dropdown-heading" aria-hidden="true">Import and export</p>
              <button @click="showPLECSImport(); showMoreMenu = false" class="dropdown-item">
                PLECS Import
              </button>
              <button @click="showBatchExport(); showMoreMenu = false" class="dropdown-item">
                Batch Export
              </button>
            </div>
          </div></Teleport>
        </div>
      </nav>
    </div>

    <div
      v-if="operationMessage"
      class="operation-message"
      :class="operationMessage.type"
      :role="operationMessage.type === 'error' ? 'alert' : 'status'"
      :aria-live="operationMessage.type === 'error' ? 'assertive' : 'polite'"
    >
      {{ operationMessage.text }}
    </div>

    <main class="app-main">
      <CatalogFom v-if="currentView === 'catalog-fom'" :supplied-catalog="suppliedCatalog" />
      <CatalogSourceWorkbench v-show="currentView === 'catalog-sources'" @explore-catalog="exploreCatalog" />
      <WaveformWorkbench v-if="currentView === 'waveform'" @thermal-handoff="stageThermal" />
      <GridWorkbench v-if="currentView === 'analysis-grid'" @thermal-handoff="stageThermal" />
      <EnergyMapWorkbench v-if="currentView === 'energy-map'" />
      <DigitizerWorkbench v-if="currentView === 'digitizer'" />
      <LegacyMatWorkbench v-if="currentView === 'legacy-mat'" />
      <MeasurementWorkbench v-if="currentView === 'measurements'" />
      <StressIndicesWorkbench v-if="currentView === 'stress-indices'" />
      <ThermalWorkbench v-if="currentView === 'thermal'" :staged-handoff="stagedThermal" />
      <SemiconductorAssemblyWorkbench v-if="currentView === 'assemblies'" />
      <ConverterWorkbench v-if="currentView === 'converter'" />
      <PlecsWorkbench v-if="currentView === 'plecs-data'" />
      <div v-if="isLoading" class="loading">
        <div class="loading-spinner"></div>
        <p>Loading transistor database...</p>
      </div>
      
      <!-- Search Database View (Advanced filtering) -->
      <SearchDatabase 
        v-if="currentView === 'search' && !isLoading"
        :transistors="transistors"
        @load-to-exporting="handleLoadToExporting"
        @load-to-comparison="handleLoadToComparison"
        @load-to-topology="handleLoadToTopology"
        @edit-transistor="showEditForm"
        @delete-transistor="handleTransistorDeleted"
      />
      
      <!-- Create/Edit Transistor Form -->
      <TransistorForm 
        v-if="currentView === 'create'"
        :transistor="selectedTransistor"
        @saved="handleTransistorSaved"
        @cancel="showSearch"
      />
      
      <!-- Exporting Tools -->
      <ExportingTools 
        v-if="currentView === 'export'"
        :transistors="transistors"
        :selected-from-search="selectedForExport"
        @open-database-search="handleOpenDatabaseSearch"
        @navigate-workbench="navigateWorkbench"
      />
      
      <!-- Basic Comparison -->
      <TransistorComparison
        v-if="currentView === 'compare'"
        :transistors="transistors"
      />

      <!-- Advanced Comparison (3 transistors, 9 plots, R_g sliders) -->
      <TransistorComparisonEnhanced
        v-if="currentView === 'compare-advanced'"
      />

      <!-- Topology Calculator -->
      <TopologyCalculator
        v-if="currentView === 'topology'"
        :transistors="transistors"
        @transistor-selected="handleTransistorSelected"
        @view-transistor-details="showEditForm"
      />

      <DatabaseBackupManager
        v-if="currentView === 'database-backup'"
        @restored="loadTransistors"
      />

      <!-- PLECS Import -->
      <PLECSImporter
        v-if="currentView === 'plecs-import'"
        @import-complete="loadTransistors"
        @view-transistor="showEditForm"
      />

      <!-- Analytical Models Calculator -->
      <AnalyticalCalculator
        v-if="currentView === 'analytical'"
      />

      <!-- DPT Manager -->
      <DPTManager
        v-if="currentView === 'dpt'"
        :transistors="transistors"
      />

      <!-- Batch Export -->
      <BatchExportPanel
        v-if="currentView === 'batch-export'"
        @close="showSearch"
        @export-complete="() => {}"
        @navigate-workbench="navigateWorkbench"
      />
    </main>

    <!-- Global notification system -->
    <div class="notification-area" id="notifications">
      <!-- Notifications will be injected here -->
    </div>
  </div>
</template>

<style>
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--bg-primary);
  color: var(--text-primary);
}
</style>

<style scoped>
.app-toolbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  position: sticky;
  top: 0;
  z-index: 200;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
}

.app-header,
.contact-header,
.contact-links,
.stats {
  display: flex;
  align-items: center;
  gap: .75rem;
  min-width: 0;
}

.app-header,
.contact-header {
  padding: .4rem .85rem;
}

.app-header h1 {
  font-size: 1rem;
  font-weight: 700;
  white-space: nowrap;
}

.stats {
  padding-left: .75rem;
  border-left: 1px solid var(--border-color);
  color: var(--text-secondary);
  font-size: .75rem;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.selected-device {
  max-width: 15rem;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: ui-monospace, monospace;
}

.contact-links a,
.theme-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: .35rem;
  color: var(--text-secondary);
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  padding: .22rem .35rem;
  font-size: .75rem;
  text-decoration: none;
}

.theme-toggle {
  border-color: var(--border-color);
  cursor: pointer;
  white-space: nowrap;
}

.contact-links a:hover,
.theme-toggle:hover {
  background: var(--accent-blue-light);
  color: var(--accent-blue);
}

.main-nav {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: .18rem;
  min-width: 0;
  padding: .25rem .7rem;
  border-top: 1px solid var(--border-color);
  background: var(--bg-tertiary);
}

.workspace-tabs {
  display: flex;
  align-items: center;
  gap: .18rem;
  min-width: 0;
  overflow-x: auto;
}

.nav-btn {
  padding: .32rem .57rem;
  border: 1px solid transparent;
  border-radius: 4px;
  background: transparent;
  color: var(--text-secondary);
  font-size: .8rem;
  font-weight: 550;
  white-space: nowrap;
  cursor: pointer;
}

.nav-btn:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.nav-btn.active {
  color: var(--accent-blue);
  border-color: var(--border-color);
  background: var(--accent-blue-light);
}

.nav-dropdown {
  position: relative;
  flex: none;
  margin-left: .25rem;
  padding-left: .4rem;
  border-left: 1px solid var(--border-color);
}

.dropdown-menu {
  position: fixed;
  overflow-y: auto;
  overscroll-behavior: contain;
  color: var(--text-primary);
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 5px;
  box-shadow: 0 4px 12px var(--shadow);
  z-index: 500;
}

.dropdown-group {
  padding: .3rem;
}

.dropdown-group + .dropdown-group {
  border-top: 1px solid var(--border-color);
}

.dropdown-heading {
  padding: .35rem .5rem .2rem;
  color: var(--text-secondary);
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
}

.dropdown-item {
  display: block;
  width: 100%;
  padding: .36rem .5rem;
  border: 1px solid transparent;
  border-radius: 4px;
  background: transparent;
  color: var(--text-primary);
  text-align: left;
  cursor: pointer;
  font-size: .8rem;
}

.dropdown-item:hover {
  background: var(--accent-blue-light);
  color: var(--accent-blue);
}

.app-main {
  flex: 1;
  padding: 0;
  max-width: 100%;
  margin: 0;
  background: var(--bg-primary);
}

.operation-message {
  margin: 1rem 2rem 0;
  padding: 0.75rem 1rem;
  border-radius: 6px;
}

.operation-message.success {
  color: #155724;
  background: #d4edda;
}

.operation-message.error {
  color: #721c24;
  background: #f8d7da;
}

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  color: var(--text-secondary);
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border-color);
  border-top: 3px solid var(--accent-blue);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.notification-area {
  position: fixed;
  top: 1rem;
  right: 1rem;
  z-index: 1000;
  max-width: 400px;
}

@media (max-width: 700px) {
  .app-header { flex-wrap: wrap; gap: .15rem .6rem; }
  .contact-header, .contact-links { gap: .25rem; }
  .stats { border: 0; padding-left: 0; }
  .selected-device { display: none; }
}

@media (pointer: coarse) {
  .nav-btn, .theme-toggle, .contact-links a, .dropdown-item { min-height: 36px; }
}

.nav-btn:focus-visible,
.dropdown-item:focus-visible,
.theme-toggle:focus-visible,
.contact-links a:focus-visible {
  outline: 2px solid var(--accent-blue);
  outline-offset: -2px;
}

.app-toolbar button:focus:not(:focus-visible) { outline: none; }

@media print {
  .app-toolbar { display: none; }
}
</style>



