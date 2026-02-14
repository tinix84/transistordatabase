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

  // Compare multiple transistors
  async compare(transistorIds) {
    const response = await api.post('/api/transistors/compare', transistorIds)
    return response.data
  },

  // Export transistor in specified format
  async export(id, format) {
    const response = await api.post(`/api/transistors/${id}/export/${format}`, {}, {
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
