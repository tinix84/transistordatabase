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

// Transistor API service
export const transistorApi = {
  // Get all transistors
  async getAll() {
    console.log('[transistorApi.getAll] Fetching all transistors...')
    try {
      const response = await api.get('/api/transistors')
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
  async export(id, format) {
    const response = await api.post(`/api/transistors/${pathSegment(id)}/export/${format}`, {}, {
      responseType: 'blob'
    })
    return response.data
  },

  // Preview export before downloading
  async exportPreview(id, format) {
    const response = await api.get(`/api/transistors/${pathSegment(id)}/export/${format}/preview`)
    return response.data
  },

  // Batch export multiple transistors
  async batchExport(transistorIds, format, options = {}) {
    const response = await api.post('/api/transistors/batch_export', {
      transistor_ids: transistorIds,
      format: format,
      options: options
    }, {
      responseType: 'blob'
    })
    return response.data
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

export default api
