import { afterEach, expect, it } from 'vitest'
import MockAdapter from 'axios-mock-adapter'
import api, { analysisApi } from '../../services/api.js'
const mock = new MockAdapter(api)
afterEach(() => mock.reset())
it('preserves exact XYZ HTTP response text and structured parser diagnostics', async () => {
  const rawJson = '{"energy_j":0.000000,"unknown":null,"row_ids":["line-1"]}'
  mock.onPost('/api/analysis/energy-map/xyz').replyOnce(200, rawJson)
  expect(await analysisApi.importXyzEnergy({ content_base64: 'MCwwLDA=' })).toEqual({ data: { energy_j: 0, unknown: null, row_ids: ['line-1'] }, rawJson })
  mock.onPost('/api/analysis/energy-map/xyz').replyOnce(422, '{"detail":"Line 1: invalid energy"}')
  await expect(analysisApi.importXyzEnergy({})).rejects.toMatchObject({ response: { data: { detail: 'Line 1: invalid energy' } } })
})
