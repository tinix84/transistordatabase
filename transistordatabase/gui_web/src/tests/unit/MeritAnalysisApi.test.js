import { afterEach, expect, it } from 'vitest'
import MockAdapter from 'axios-mock-adapter'
import api, { analysisApi } from '../../services/api.js'

const mock = new MockAdapter(api)
afterEach(() => mock.reset())
it('posts advanced grid JSON text unchanged with its numeric literal and JSON content type', async () => {
  const raw = '\r\n{"objectives":[{"weight":1e-999}],"zero":0}\r\n'
  mock.onPost('/api/analysis/grid').reply(config => {
    expect(config.data).toBe(raw)
    expect(config.headers.getContentType()).toBe('application/json')
    return [422, { detail: 'Numeric value is outside representable precision' }]
  })
  await expect(analysisApi.grid(raw)).rejects.toMatchObject({ response: { status: 422 } })
})
it('keeps guided grid objects on the existing JSON request path', async () => {
  const request = { objectives: [{ name: 'conduction_power_w', unit: 'W', direction: 'minimize', weight: 0 }] }
  mock.onPost('/api/analysis/grid').reply(config => {
    expect(JSON.parse(config.data)).toEqual(request)
    return [200, { inputs: request }]
  })
  expect(await analysisApi.grid(request)).toEqual({ inputs: request })
})
