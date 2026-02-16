import axios from 'axios'

// Base URL for the API - automatically detects if running on Vercel or locally
const getBaseURL = () => {
  if (typeof window !== 'undefined') {
    // Browser environment
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://localhost:8002'
    } else {
      // Production (Vercel) - use relative paths
      return ''
    }
  }
  // Server environment fallback
  return 'http://localhost:8002'
}

const BASE_URL = getBaseURL()

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
    const response = await api.get(`/api/transistors/${id}`)
    return response.data
  },

  // Create a new transistor
  async create(transistorData) {
    const response = await api.post('/api/transistors', transistorData)
    return response.data
  },

  // Update an existing transistor
  async update(id, transistorData) {
    const response = await api.put(`/api/transistors/${id}`, transistorData)
    return response.data
  },

  // Delete a transistor
  async delete(id) {
    const response = await api.delete(`/api/transistors/${id}`)
    return response.data
  },

  // Validate a transistor
  async validate(id) {
    const response = await api.post(`/api/transistors/${id}/validate`)
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
    const response = await api.post(`/api/transistors/${id}/export/${format}`, {}, {
      responseType: 'blob'
    })
    return response.data
  },

  // Preview export before downloading
  async exportPreview(id, format) {
    const response = await api.get(`/api/transistors/${id}/export/${format}/preview`)
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

  // Get all curves for a transistor
  async getCurves(id) {
    const response = await api.get(`/api/transistors/${id}/curves`)
    return response.data
  },

  // Add channel curve
  async addChannelCurve(id, curveData, component = 'switch') {
    const response = await api.post(
      `/api/transistors/${id}/curves/channel?component=${component}`,
      curveData
    )
    return response.data
  },

  // Add switching loss curve (e_on, e_off, e_rr)
  async addSwitchingLossCurve(id, lossType, curveData) {
    const response = await api.post(
      `/api/transistors/${id}/curves/switching/${lossType}`,
      curveData
    )
    return response.data
  },

  // Add gate charge curve
  async addGateChargeCurve(id, curveData) {
    const response = await api.post(
      `/api/transistors/${id}/curves/gate_charge`,
      curveData
    )
    return response.data
  },

  // Add SOA curve
  async addSOACurve(id, curveData) {
    const response = await api.post(
      `/api/transistors/${id}/curves/soa`,
      curveData
    )
    return response.data
  },

  // Add capacitance curve
  async addCapacitanceCurve(id, capType, curveData) {
    const response = await api.post(
      `/api/transistors/${id}/curves/capacitance/${capType}`,
      curveData
    )
    return response.data
  },

  // Delete curve
  async deleteCurve(id, component, curveType, index) {
    const response = await api.delete(
      `/api/transistors/${id}/curves/${component}/${curveType}/${index}`
    )
    return response.data
  },

  // Validate curves
  async validateCurves(id) {
    const response = await api.post(`/api/transistors/${id}/curves/validate`)
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

  // DPT (Double Pulse Test)
  async generateDPTNetlist(transistorId, config) {
    const response = await api.post('/api/dpt/generate_netlist', config, {
      params: { transistor_id: transistorId }
    })
    return response.data
  },

  async getDPTData(transistorId) {
    const response = await api.get(`/api/transistors/${transistorId}/dpt_data`)
    return response.data
  },

  async validateDPTData(transistorId) {
    const response = await api.post(`/api/transistors/${transistorId}/dpt_validate`)
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
