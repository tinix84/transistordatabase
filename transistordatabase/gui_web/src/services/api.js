import axios from 'axios'

// Base URL for the API - automatically detects if running on Vercel or locally
const getBaseURL = () => {
  if (typeof window !== 'undefined') {
    // Browser environment
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      // The bundled desktop-replacement app is served by FastAPI on port 8000.
      // Vite's development server remains on 5173/5174 and uses backend port 8002.
      if (['5173', '5174'].includes(window.location.port)) {
        return `http://${window.location.hostname}:8002`
      }
      return ''
    } else {
      // Production (Vercel) - use relative paths
      return ''
    }
  }
  // Server environment fallback
  return 'http://localhost:8002'
}

const BASE_URL = getBaseURL()

const exactRecordUrl = (name) => `/api/transistor-record?name=${encodeURIComponent(name)}`
const pathSegment = (value) => encodeURIComponent(value)

// Debug logging
console.log('[API] Initialized with BASE_URL:', BASE_URL)

// Create axios instance with default config
const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Blob downloads also receive JSON validation errors from the backend.
async function downloadBlob(request) {
  try { return (await request).data }
  catch (error) {
    let data = error.response?.data
    if (data instanceof Blob) {
      try { data = JSON.parse(await data.text()) } catch { /* Keep the transport error. */ }
    }
    const detail = data?.detail
    if (detail) {
      error.message = typeof detail === 'string' ? detail : detail.map(item => item.msg || String(item)).join('\n')
    }
    throw error
  }
}

// Request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log(`[API Request] ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`)
    return config
  },
  (error) => {
    console.error('[API Request Error]:', error)
    return Promise.reject(error)
  }
)

export const catalogApi = {
  async sources() {
    return (await api.get('/api/catalog/sources')).data
  },
  async screen(request) {
    return (await api.post('/api/catalog/screen', request)).data
  },
}

export const converterApi = {
  async presets() {
    return (await api.get('/api/converters/presets')).data
  },
  async analyze(request) {
    return (await api.post('/api/converters/analyze', request)).data
  },
}

export const plecsDataApi = {
  async preview(request) {
    return (await api.post('/api/plecs/preview', request)).data
  },
  async export(request) {
    return (await api.post('/api/plecs/export', request, { responseType: 'blob' })).data
  },
  async finalize(request) {
    return (await api.post('/api/plecs/finalize', request)).data
  },
  async contexts(id) {
    return (await api.get(`/api/plecs/core/${encodeURIComponent(id)}/contexts`)).data
  },
  async previewCore(id, request) {
    return (await api.post(`/api/plecs/core/${encodeURIComponent(id)}/preview`, request)).data
  },
  async exportCore(id, request) {
    return (await api.post(`/api/plecs/core/${encodeURIComponent(id)}/export`, request, { responseType: 'blob' })).data
  },
}

export const measurementApi = {
  async capabilities() { return (await api.get('/api/measurement/capabilities')).data },
  async preview(request) { return (await api.post('/api/measurement/preview', request)).data },
  async extract(request) { return (await api.post('/api/measurement/extract', request)).data },
}

async function reviewPost(path, request, config = {}) {
  try {
    const response = await api.post(path, request, { ...config, responseType: 'text', transformResponse: [value => value] })
    return { data: JSON.parse(response.data), rawJson: response.data }
  } catch (error) {
    if (typeof error.response?.data === 'string') {
      try { error.response.data = JSON.parse(error.response.data) } catch { /* Keep the transport error. */ }
    }
    throw error
  }
}

export const stressApi = {
  async calculate(request) { return reviewPost('/api/stress/calculate', request) },
}

export const deviceModelApi = {
  async train(request) { return reviewPost('/api/models/train', request) },
  async evaluate(request) { return reviewPost('/api/models/evaluate', request) },
  async evaluateModelJson(modelJson, queries) {
    return reviewPost('/api/models/evaluate', `{"model":${modelJson},"queries":${JSON.stringify(queries)}}`, { transformRequest: [value => value] })
  },
}

export const rgEnergyApi = {
  async inventory(id) { return (await api.get(`/api/transistors/${encodeURIComponent(id)}/rg-energy`)).data },
  async calculate(id, request) { return reviewPost(`/api/transistors/${encodeURIComponent(id)}/rg-energy`, request) },
}

export const matApi = {
  async preview(request) { return (await api.post('/api/mat/preview', request)).data },
  async array(request) { return (await api.post('/api/mat/array', request)).data },
  async export(request) { return (await api.post('/api/mat/export', request, { responseType: 'blob' })).data },
  async finalize(request) { return (await api.post('/api/mat/finalize', request)).data },
  async calculate(request) { return (await api.post('/api/mat/calculate', request)).data },
}

export const thermalApi = {
  async analyze(request) {
    return (await api.post('/api/thermal/analyze', request)).data
  }
}

export const assemblyApi = {
  async reverseSharing(request) { return reviewPost('/api/assemblies/reverse-sharing', request) },
  async analyze(request) {
    return (await api.post('/api/assemblies/analyze', request)).data
  }
}

export const waveformApi = {
  async allocate(request) { return reviewPost('/api/waveform/gate-allocation', request) },
  async analyze(request) {
    return (await api.post('/api/waveform/analyze', request)).data
  },
}

export const analysisApi = {
  async grid(request) {
    const config = typeof request === 'string' ? { headers: { 'Content-Type': 'application/json' }, transformRequest: [value => value] } : undefined
    return (await api.post('/api/analysis/grid', request, config)).data
  },
  async energyMap(request) { return (await api.post('/api/analysis/energy-map', request)).data },
  async importXyzEnergy(request) { return reviewPost('/api/analysis/energy-map/xyz', request) },
  async thermalHandoff(request) { return (await api.post('/api/analysis/thermal-handoff', request)).data },
}

export const digitizerApi = {
  async image(request) {
    return (await api.post('/api/digitizer/image', request)).data
  },
  async transform(request) {
    return (await api.post('/api/digitizer/transform', request)).data
  },
}

// Transistor API service
export const transistorApi = {
  // Get all transistors
  async getAll() {
    console.log('[transistorApi.getAll] Fetching all transistors...')
    try {
      const response = await api.get('/api/transistors')
      if (!Array.isArray(response.data)) throw new Error('Invalid transistor list response')
      console.log('[transistorApi.getAll] Success! Received', response.data?.length || 0, 'transistors')
      console.log('[transistorApi.getAll] Sample data:', response.data?.[0])
      return response.data
    } catch (error) {
      console.error('[transistorApi.getAll] Failed:', error)
      throw error
    }
  },

  // Get a specific transistor by ID
  async getById(id) {
    const response = await api.get(exactRecordUrl(id))
    return response.data
  },

  // Create a new transistor
  async create(transistorData) {
    const response = await api.post('/api/transistors', transistorData)
    return response.data
  },

  // Update an existing transistor
  async update(id, transistorData) {
    const response = await api.put(exactRecordUrl(id), transistorData)
    return response.data
  },

  // Delete a transistor
  async delete(id) {
    const response = await api.delete(exactRecordUrl(id))
    return response.data
  },

  // Validate a transistor
  async validate(id) {
    const response = await api.post(`/api/transistors/${pathSegment(id)}/validate`)
    return response.data
  },

  // Compare multiple transistors (basic)
  async compare(transistorIds) {
    const response = await api.post('/api/transistors/compare', transistorIds)
    return response.data
  },

  // Advanced comparison with 9 plot types and configuration
  async compareAdvanced(transistorIds, config) {
    const response = await api.post('/api/comparison/advanced', {
      transistor_ids: transistorIds,
      config: config || {}
    })
    return response.data
  },

  // Export transistor in specified format
  async export(id, format, selection = {}) {
    return downloadBlob(api.post(`/api/transistors/${pathSegment(id)}/export/${encodeURIComponent(format)}`, selection, {
      responseType: 'blob'
    }))
  },

  // Preview export before downloading
  async exportPreview(id, format) {
    const response = await api.get(`/api/transistors/${pathSegment(id)}/export/${format}/preview`)
    return response.data
  },

  // Batch export multiple transistors
  async batchExport(transistorIds, format, options = {}) {
    return downloadBlob(api.post('/api/transistors/batch_export', {
      transistor_ids: transistorIds,
      format: format,
      options: options
    }, {
      responseType: 'blob'
    }))
  },

  // Upload transistor from file
  async upload(file) {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post('/api/transistors/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Full raw-record database backup and additive restore
  async downloadDatabaseBackup() {
    const response = await api.get('/api/database/backup', { responseType: 'blob' })
    return response.data
  },

  async validateDatabaseBackup(file) {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post('/api/database/backup/validate', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
  },

  async restoreDatabaseBackup(file) {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post('/api/database/backup/restore', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
  },

  // Get all curves for a transistor
  async getCurves(id) {
    const response = await api.get(`/api/transistors/${pathSegment(id)}/curves`)
    return response.data
  },

  // Add channel curve
  async addChannelCurve(id, curveData, component = 'switch') {
    const response = await api.post(
      `/api/transistors/${pathSegment(id)}/curves/channel?component=${component}`,
      curveData
    )
    return response.data
  },

  // Add switching loss curve (e_on, e_off, e_rr)
  async addSwitchingLossCurve(id, lossType, curveData) {
    const response = await api.post(
      `/api/transistors/${pathSegment(id)}/curves/switching/${lossType}`,
      curveData
    )
    return response.data
  },

  // Add gate charge curve
  async addGateChargeCurve(id, curveData) {
    const response = await api.post(
      `/api/transistors/${pathSegment(id)}/curves/gate_charge`,
      curveData
    )
    return response.data
  },

  // Add SOA curve
  async addSOACurve(id, curveData) {
    const response = await api.post(
      `/api/transistors/${pathSegment(id)}/curves/soa`,
      curveData
    )
    return response.data
  },

  // Add capacitance curve
  async addCapacitanceCurve(id, capType, curveData) {
    const response = await api.post(
      `/api/transistors/${pathSegment(id)}/curves/capacitance/${capType}`,
      curveData
    )
    return response.data
  },

  // Delete curve
  async deleteCurve(id, component, curveType, index) {
    const response = await api.delete(
      `/api/transistors/${pathSegment(id)}/curves/${component}/${curveType}/${index}`
    )
    return response.data
  },

  // Update one curve with list-count and raw-entry concurrency preconditions
  async updateCurve(id, component, curveType, index, update) {
    const response = await api.put(
      `/api/transistors/${pathSegment(id)}/curves/${component}/${curveType}/${index}`,
      update
    )
    return response.data
  },

  async addThermalCurve(id, component, update) {
    const response = await api.post(
      `/api/transistors/${pathSegment(id)}/curves/thermal/${component}`,
      update
    )
    return response.data
  },

  async updateThermalCurve(id, component, update) {
    const response = await api.put(
      `/api/transistors/${pathSegment(id)}/curves/thermal/${component}`,
      update
    )
    return response.data
  },

  async deleteThermalCurve(id, component, expectedToken) {
    const response = await api.delete(
      `/api/transistors/${pathSegment(id)}/curves/thermal/${component}`,
      { params: { expected_token: expectedToken } }
    )
    return response.data
  },

  // Validate curves
  async validateCurves(id) {
    const response = await api.post(`/api/transistors/${pathSegment(id)}/curves/validate`)
    return response.data
  },

  // ==================== Phase 6: Archive Integration ====================

  // PLECS Import
  async importPLECS(file) {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post('/api/import/plecs', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
  },

  // Analytical Models
  async calculateBiela(params) {
    const response = await api.post('/api/analytical/biela', params)
    return response.data
  },

  async calculateGateCharge(params) {
    const response = await api.post('/api/analytical/gate_charge', params)
    return response.data
  },

  // Topology calculations and Python-generated sweeps
  async calculateTopology(params) {
    const response = await api.post('/api/topology/calculate', params)
    return response.data
  },

  async compareTopologies(params) {
    const response = await api.post('/api/topology/compare', params)
    return response.data
  },

    async calculateTopologyPareto(params) {
      const response = await api.post('/api/topology/pareto', params)
      return response.data
    },

  // DPT (Double Pulse Test)
  async generateDPTNetlist(transistorId, config) {
    const response = await api.post('/api/dpt/generate_netlist', config, {
      params: { transistor_id: transistorId }
    })
    return response.data
  },

  async getDPTData(transistorId, options = {}) {
    const path = `/api/transistors/${pathSegment(transistorId)}/dpt_data`
    const previewPoints = options?.waveformPreviewPoints
    const response = Number.isInteger(previewPoints)
      ? await api.get(path, { params: { waveform_preview_points: previewPoints } })
      : await api.get(path)
    return response.data
  },

  async previewDPTRawImport(transistorId, file) {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post(
      `/api/transistors/${pathSegment(transistorId)}/dpt_raw_import/preview`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    return response.data
  },

  async previewDPTRawTraces(transistorId, file, selection) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('selection_json', JSON.stringify(selection))
    const response = await api.post(
      `/api/transistors/${pathSegment(transistorId)}/dpt_raw_import/trace_preview`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    return response.data
  },

  async importDPTRawMeasurement(transistorId, file, selection) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('selection_json', JSON.stringify(selection))
    const response = await api.post(
      `/api/transistors/${pathSegment(transistorId)}/dpt_raw_import`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    return response.data
  },

  async validateDPTData(transistorId) {
    const response = await api.post(`/api/transistors/${pathSegment(transistorId)}/dpt_validate`)
    return response.data
  },

  async previewDPTEnergy(transistorId, request) {
    const response = await api.post(
      `/api/transistors/${pathSegment(transistorId)}/dpt_energy/preview`,
      request
    )
    return response.data
  },

  async saveDPTEnergy(transistorId, request) {
    const response = await api.post(
      `/api/transistors/${pathSegment(transistorId)}/dpt_energy/save`,
      request
    )
    return response.data
  }
}

// Response interceptor for debugging and error handling
api.interceptors.response.use(
  (response) => {
    console.log(`[API Response] ${response.config.method?.toUpperCase()} ${response.config.url}:`,
      response.status, response.statusText)
    if (response.data) {
      const dataInfo = Array.isArray(response.data)
        ? `Array[${response.data.length}]`
        : typeof response.data
      console.log(`[API Response] Data type:`, dataInfo)
    }
    return response
  },
  (error) => {
    console.error('[API Error]:', {
      message: error.message,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      url: error.config?.url
    })
    throw error
  }
)

export const catalogSourceApi = {
  async preview(payload) { return (await api.post('/api/catalog-sources/preview', payload)).data },
  async snapshot(payload) { return (await api.post('/api/catalog-sources/snapshot', payload)).data },
  async links(payload) { return (await api.post('/api/catalog-sources/links', payload)).data },
  async coefficients(payload) { return (await api.post('/api/catalog-sources/coefficients', payload)).data },
  async diff(payload) { return (await api.post('/api/catalog-sources/diff', payload)).data },
}

export default api
