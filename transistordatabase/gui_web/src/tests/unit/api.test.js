/**
 * Unit tests for API service
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import axios from 'axios'
import MockAdapter from 'axios-mock-adapter'
import { mockTransistors, mockTransistor1 } from '../fixtures/transistors'

// We need to mock axios BEFORE importing the API service
// This ensures the api service uses our mocked axios instance
let mock
let transistorApi

describe('API Service', () => {
  beforeEach(async () => {
    // Create mock adapter for the default axios instance
    mock = new MockAdapter(axios, { onNoMatch: 'passthrough' })

    // Clear module cache and re-import to get fresh instance
    vi.resetModules()
    const apiModule = await import('../../services/api.js')
    transistorApi = apiModule.transistorApi

    // Suppress console logs during tests
    vi.spyOn(console, 'log').mockImplementation(() => {})
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    mock.reset()
    mock.restore()
    vi.restoreAllMocks()
  })

  describe('getAll()', () => {
    it('should fetch all transistors successfully', async () => {
      mock.onGet('/api/transistors').reply(200, mockTransistors)

      const result = await transistorApi.getAll()

      expect(result).toEqual(mockTransistors)
      expect(result).toHaveLength(3)
      expect(result[0].metadata.name).toBe('CREE_C3M0060065J')
    })

    it('should handle empty array response', async () => {
      mock.onGet('/api/transistors').reply(200, [])

      const result = await transistorApi.getAll()

      expect(result).toEqual([])
      expect(result).toHaveLength(0)
    })

    it('should throw error on failed request', async () => {
      mock.onGet('/api/transistors').reply(500, { detail: 'Server Error' })

      await expect(transistorApi.getAll()).rejects.toThrow('Request failed with status code 500')
      expect(console.error).toHaveBeenCalled()
    })

    it('should throw error on network failure', async () => {
      mock.onGet('/api/transistors').networkError()

      await expect(transistorApi.getAll()).rejects.toThrow()
    })

    it('should log successful requests', async () => {
      mock.onGet('/api/transistors').reply(200, mockTransistors)

      await transistorApi.getAll()

      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('[transistorApi.getAll] Fetching')
      )
      expect(console.log).toHaveBeenCalledWith(
        expect.stringContaining('[transistorApi.getAll] Success!')
      )
    })

    it('should log failed requests', async () => {
      mock.onGet('/api/transistors').reply(500)

      try {
        await transistorApi.getAll()
        // Should not reach here
        expect(true).toBe(false)
      } catch (error) {
        // Error should be thrown
        expect(error.message).toContain('Request failed with status code 500')
      }

      // Should log the error
      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining('[transistorApi.getAll] Failed:'),
        expect.anything()
      )
    })
  })

  describe('getById()', () => {
    it('should fetch specific transistor by ID', async () => {
      const name = 'CREE_C3M0060065J'
      mock.onGet(`/api/transistors/${name}`).reply(200, mockTransistor1)

      const result = await transistorApi.getById(name)

      expect(result).toEqual(mockTransistor1)
      expect(result.metadata.name).toBe(name)
    })

    it('should handle 404 for non-existent transistor', async () => {
      const name = 'NonExistent'
      mock.onGet(`/api/transistors/${name}`).reply(404, { detail: 'Not found' })

      await expect(transistorApi.getById(name)).rejects.toThrow()
    })
  })

  describe('create()', () => {
    it('should create new transistor', async () => {
      mock.onPost('/api/transistors').reply(201, mockTransistor1)

      const result = await transistorApi.create(mockTransistor1)

      expect(result).toEqual(mockTransistor1)
    })

    it('should handle validation errors', async () => {
      mock.onPost('/api/transistors').reply(400, {
        detail: 'Validation error'
      })

      await expect(transistorApi.create({})).rejects.toThrow()
    })
  })

  describe('update()', () => {
    it('should update existing transistor', async () => {
      const name = 'CREE_C3M0060065J'
      const updated = { ...mockTransistor1, metadata: { ...mockTransistor1.metadata, comment: 'Updated' } }

      mock.onPut(`/api/transistors/${name}`).reply(200, updated)

      const result = await transistorApi.update(name, updated)

      expect(result).toEqual(updated)
      expect(result.metadata.comment).toBe('Updated')
    })

    it('should handle 404 for non-existent transistor', async () => {
      const name = 'NonExistent'
      mock.onPut(`/api/transistors/${name}`).reply(404)

      await expect(transistorApi.update(name, {})).rejects.toThrow()
    })
  })

  describe('delete()', () => {
    it('should delete transistor', async () => {
      const name = 'CREE_C3M0060065J'
      mock.onDelete(`/api/transistors/${name}`).reply(200, { message: 'Deleted' })

      const result = await transistorApi.delete(name)

      expect(result).toEqual({ message: 'Deleted' })
    })

    it('should handle 404 for non-existent transistor', async () => {
      const name = 'NonExistent'
      mock.onDelete(`/api/transistors/${name}`).reply(404)

      await expect(transistorApi.delete(name)).rejects.toThrow()
    })
  })

  describe('validate()', () => {
    it('should validate transistor successfully', async () => {
      const name = 'CREE_C3M0060065J'
      const validationResult = { valid: true, warnings: [], errors: [] }

      mock.onPost(`/api/transistors/${name}/validate`).reply(200, validationResult)

      const result = await transistorApi.validate(name)

      expect(result).toEqual(validationResult)
      expect(result.valid).toBe(true)
    })

    it('should return validation errors', async () => {
      const name = 'CREE_C3M0060065J'
      const validationResult = {
        valid: false,
        warnings: ['Missing data'],
        errors: ['Invalid voltage']
      }

      mock.onPost(`/api/transistors/${name}/validate`).reply(200, validationResult)

      const result = await transistorApi.validate(name)

      expect(result.valid).toBe(false)
      expect(result.errors).toHaveLength(1)
    })
  })

  describe('compare()', () => {
    it('should compare multiple transistors', async () => {
      const transistorIds = ['CREE_C3M0060065J', 'Infineon_FF300R12KE3']
      const comparisonResult = {
        transistors: mockTransistors.slice(0, 2),
        comparison: { v_abs_max: [650, 1200] }
      }

      mock.onPost('/api/transistors/compare').reply(200, comparisonResult)

      const result = await transistorApi.compare(transistorIds)

      expect(result).toEqual(comparisonResult)
      expect(result.transistors).toHaveLength(2)
    })
  })

  describe('export()', () => {
    it('should export transistor in specified format', async () => {
      const name = 'CREE_C3M0060065J'
      const format = 'json'
      const blob = new Blob(['mock data'], { type: 'application/json' })

      mock.onPost(`/api/transistors/${name}/export/${format}`).reply(200, blob)

      const result = await transistorApi.export(name, format)

      expect(result).toBeInstanceOf(Blob)
    })

    it('should handle unsupported format', async () => {
      const name = 'CREE_C3M0060065J'
      const format = 'unsupported'

      mock.onPost(`/api/transistors/${name}/export/${format}`).reply(400, {
        detail: 'Unsupported format'
      })

      await expect(transistorApi.export(name, format)).rejects.toThrow()
    })
  })

  describe('upload()', () => {
    it('should upload transistor file', async () => {
      const file = new File(['{}'], 'transistor.json', { type: 'application/json' })

      mock.onPost('/api/transistors/upload').reply(200, mockTransistor1)

      const result = await transistorApi.upload(file)

      expect(result).toEqual(mockTransistor1)
    })

    it('should handle invalid file format', async () => {
      const file = new File(['invalid'], 'transistor.txt', { type: 'text/plain' })

      mock.onPost('/api/transistors/upload').reply(400, {
        detail: 'Invalid file format'
      })

      await expect(transistorApi.upload(file)).rejects.toThrow()
    })
  })

  describe('Error Handling', () => {
    it('should handle timeout errors', async () => {
      mock.onGet('/api/transistors').timeout()

      await expect(transistorApi.getAll()).rejects.toThrow()
    })

    it('should handle CORS errors', async () => {
      mock.onGet('/api/transistors').networkError()

      await expect(transistorApi.getAll()).rejects.toThrow()
    })

    it('should handle malformed JSON response', async () => {
      mock.onGet('/api/transistors').reply(200, 'not json')

      await expect(transistorApi.getAll()).rejects.toThrow()
    })
  })
})
