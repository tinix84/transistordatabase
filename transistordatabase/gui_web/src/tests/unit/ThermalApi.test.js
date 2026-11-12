import { afterEach, expect, it } from 'vitest'
import MockAdapter from 'axios-mock-adapter'
import api, { thermalApi } from '../../services/api.js'
const mock = new MockAdapter(api)
afterEach(() => mock.reset())
it('preserves advanced thermal numeric lexemes until backend validation', async () => {
  const raw = '{"interfaces":[{"resistivity_k_m2_w":-1e-400}]}'
  mock.onPost('/api/thermal/analyze').reply(config => {
    expect(config.data).toBe(raw)
    expect(config.headers['Content-Type']).toBe('application/json')
    return [422, { detail: 'Original numeric literal is outside the supported range' }]
  })
  await expect(thermalApi.analyze(raw)).rejects.toMatchObject({ response: { data: { detail: 'Original numeric literal is outside the supported range' } } })
})
it('keeps existing thermal object requests in ordinary JSON transport', async () => {
  const request = { operation: 'steady', source: 'Synthetic fixture', interface: null }
  mock.onPost('/api/thermal/analyze').reply(config => {
    expect(JSON.parse(config.data)).toEqual(request)
    return [200, { operation: 'steady' }]
  })
  expect(await thermalApi.analyze(request)).toEqual({ operation: 'steady' })
})
