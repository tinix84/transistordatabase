import { afterEach, expect, it } from 'vitest'
import MockAdapter from 'axios-mock-adapter'
import api, { deviceModelApi } from '../../services/api.js'
const mock = new MockAdapter(api)
afterEach(() => mock.reset())
it.each(['train', 'evaluate'])('preserves exact model %s review text and structured validation errors', async operation => {
  const path = `/api/models/${operation}`, rawJson = '{"value":0.01000,"unknown":null}'
  mock.onPost(path).replyOnce(200, rawJson)
  expect(await deviceModelApi[operation]({})).toEqual({ data: { value: .01, unknown: null }, rawJson })
  mock.onPost(path).replyOnce(422, '{"detail":"Unsupported model"}')
  await expect(deviceModelApi[operation]({})).rejects.toMatchObject({ response: { data: { detail: 'Unsupported model' } } })
})
it('sends imported model number tokens without browser normalization', async () => {
  const raw = '{"family":"ron","coefficient":1e-999,"negative_zero":-0.0}', queries = [{ id: 'q', output: 'resistance', coordinates: { t_j_c: 50 } }]
  mock.onPost('/api/models/evaluate').reply(config => {
    expect(config.data).toBe(`{"model":${raw},"queries":${JSON.stringify(queries)}}`)
    return [422, '{"detail":"Underflow is invalid"}']
  })
  await expect(deviceModelApi.evaluateModelJson(raw, queries)).rejects.toMatchObject({ response: { data: { detail: 'Underflow is invalid' } } })
})
