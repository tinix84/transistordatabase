/**
 * Mock API responses for testing
 */
import MockAdapter from 'axios-mock-adapter'
import { mockTransistors, mockTransistor1 } from '../fixtures/transistors'

/**
 * Create a mock adapter for axios instance
 * @param {import('axios').AxiosInstance} axiosInstance
 * @param {Object} options
 * @returns {MockAdapter}
 */
export function createMockApi(axiosInstance, options = {}) {
  const mock = new MockAdapter(axiosInstance, {
    delayResponse: options.delay || 0,
    onNoMatch: options.onNoMatch || 'throwException'
  })

  // GET /api/transistors - Get all transistors
  mock.onGet('/api/transistors').reply(200, mockTransistors)

  // GET /api/transistors/:name - Get specific transistor
  mock.onGet(/\/api\/transistors\/[^\/]+$/).reply((config) => {
    const name = config.url.split('/').pop()
    const transistor = mockTransistors.find(t => t.metadata.name === name)

    if (transistor) {
      return [200, transistor]
    }
    return [404, { detail: `Transistor ${name} not found` }]
  })

  // POST /api/transistors - Create transistor
  mock.onPost('/api/transistors').reply((config) => {
    const data = JSON.parse(config.data)
    return [201, data]
  })

  // PUT /api/transistors/:name - Update transistor
  mock.onPut(/\/api\/transistors\/[^\/]+$/).reply((config) => {
    const data = JSON.parse(config.data)
    return [200, data]
  })

  // DELETE /api/transistors/:name - Delete transistor
  mock.onDelete(/\/api\/transistors\/[^\/]+$/).reply(200, { message: 'Deleted successfully' })

  // POST /api/transistors/:name/validate - Validate transistor
  mock.onPost(/\/api\/transistors\/[^\/]+\/validate$/).reply(200, {
    valid: true,
    warnings: [],
    errors: []
  })

  // POST /api/transistors/compare - Compare transistors
  mock.onPost('/api/transistors/compare').reply(200, {
    transistors: mockTransistors.slice(0, 2),
    comparison: {
      v_abs_max: [650, 1200],
      i_cont: [60, 300]
    }
  })

  // POST /api/transistors/:name/export/:format - Export transistor
  mock.onPost(/\/api\/transistors\/[^\/]+\/export\/[^\/]+$/).reply(200, new Blob(['mock data']), {
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Disposition': 'attachment; filename="transistor.json"'
    }
  })

  // POST /api/transistors/upload - Upload transistor file
  mock.onPost('/api/transistors/upload').reply(200, mockTransistor1)

  return mock
}

/**
 * Create a mock API that returns errors
 * @param {import('axios').AxiosInstance} axiosInstance
 * @returns {MockAdapter}
 */
export function createErrorMockApi(axiosInstance) {
  const mock = new MockAdapter(axiosInstance)

  // All requests return 500 error
  mock.onAny().reply(500, { detail: 'Internal Server Error' })

  return mock
}

/**
 * Create a mock API that returns network errors
 * @param {import('axios').AxiosInstance} axiosInstance
 * @returns {MockAdapter}
 */
export function createNetworkErrorMockApi(axiosInstance) {
  const mock = new MockAdapter(axiosInstance)

  // All requests return network error
  mock.onAny().networkError()

  return mock
}

/**
 * Create a mock API with custom responses
 * @param {import('axios').AxiosInstance} axiosInstance
 * @param {Object} responses - Custom response map
 * @returns {MockAdapter}
 */
export function createCustomMockApi(axiosInstance, responses = {}) {
  const mock = new MockAdapter(axiosInstance)

  Object.entries(responses).forEach(([endpoint, response]) => {
    const [method, url] = endpoint.split(' ')
    const mockMethod = mock[`on${method}`]

    if (mockMethod) {
      mockMethod.call(mock, url).reply(response.status || 200, response.data)
    }
  })

  return mock
}
