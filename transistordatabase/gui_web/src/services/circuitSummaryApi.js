import api from './api.js'

export async function summarizeCircuit(request) {
  return (await api.post('/api/converters/circuit-summary', request,
    typeof request === 'string' ? { headers: { 'Content-Type': 'application/json' } } : undefined)).data
}
