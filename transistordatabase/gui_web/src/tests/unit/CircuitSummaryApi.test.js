import { expect, it, vi } from 'vitest'
const { post } = vi.hoisted(() => ({ post: vi.fn() }))
vi.mock('../../services/api.js', () => ({ default: { post } }))
import { summarizeCircuit } from '../../services/circuitSummaryApi.js'

it('transports advanced JSON bytes unchanged for the server numeric guard', async () => {
  post.mockResolvedValueOnce({ data: { ok: true } })
  const raw = '\r\n{"members":[{"rtheta_k_w":1e-999}],"zero":0}\r\n'
  expect(await summarizeCircuit(raw)).toEqual({ ok: true })
  expect(post).toHaveBeenCalledWith('/api/converters/circuit-summary', raw, { headers: { 'Content-Type': 'application/json' } })
})
