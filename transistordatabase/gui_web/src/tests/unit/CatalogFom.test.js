import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
const { sources, screen } = vi.hoisted(() => ({ sources: vi.fn(), screen: vi.fn() }))
vi.mock('../../services/api.js', () => ({ catalogApi: { sources, screen } }))
import CatalogFom from '../../components/CatalogFom.vue'
import { catalogCsv } from '../../components/catalogPresentation.js'

const metadata = { id: 'A', sha256: 'a'.repeat(64), currency: 'CHF', file: 'source.csv', sheet: 'CSV', snapshot: 'historical-A' }
const metrics = { voltage_v: { label: 'Voltage', unit: 'V', direction: 'max' }, current_a: { label: 'Current', unit: 'A', direction: 'max' } }
const row = { id: 'A1', source_id: 'A', row: 2, part_number: '=literal', technology: 'SiC', manufacturer: 'maker', pareto: null, metrics: { voltage_v: 4, current_a: 8 }, conditions: { resistance: '25 C', gate_charge: '15 V' }, normalization: { voltage_v: { relative: 2 } } }
const result = { source: metadata, sources: [metadata], candidates: [row], panels: [{ id: 'default-0', x_metric: 'voltage_v', y_metric: 'current_a', points: [{ id: 'A1', x: 4, y: 8 }], matched_count: 1, excluded_count: 0 }], matched_count: 1, plotted_count: 0, pareto_count: 0, excluded_count: 1, assumptions: [], coverage: {}, normalization: { mode: 'fixed', metrics: { voltage_v: { denominator: 2, denominator_ids: ['A0'], population_ids: ['A0', 'A1'], population_hash: 'hash', status: 'available' } } } }
describe('catalog FOM source and population controls', () => {
  let wrapper
  beforeEach(() => {
    vi.useFakeTimers()
    sources.mockReset().mockResolvedValue({ sources: [metadata], metrics, facets: { A: { technology: ['SiC'], manufacturer: ['maker'] } } })
    screen.mockReset().mockResolvedValue(result)
  })
  afterEach(() => { wrapper?.unmount(); vi.useRealTimers() })
  async function load(props = {}) {
    wrapper = mount(CatalogFom, { props, global: { stubs: { CatalogScatter: { name: 'CatalogScatter', props: ['points'], template: '<div class="scatter-stub" />' } } } })
    await flushPromises(); await vi.advanceTimersByTimeAsync(200); await flushPromises()
  }
  it('resolves compact panel references and keeps chart data stable during row inspection', async () => {
    await load()
    const chart = wrapper.findComponent({ name: 'CatalogScatter' })
    const points = chart.props('points')
    expect(points[0].id).toBe('A1')
    expect(points[0].pareto).toBe(null)
    await wrapper.get('.part-button').trigger('click')
    expect(chart.props('points')).toBe(points)
    expect(wrapper.get('[aria-label="Selected catalog device"]').text()).toContain('Unavailable for chosen objectives')
  })
  it('forwards supplied snapshots and explicit fixed normalization without mutating the source', async () => {
    const snapshot = { schema_version: 1, sources: [metadata], records: [{ ...row }] }
    const original = JSON.stringify(snapshot)
    await load({ suppliedCatalog: snapshot })
    expect(screen.mock.calls[0][0].snapshot).toEqual(snapshot)
    await wrapper.get('[aria-label="Axis values"]').setValue('fixed')
    await vi.advanceTimersByTimeAsync(200)
    expect(screen.mock.lastCall[0].normalization).toEqual({ mode: 'fixed', metrics: ['voltage_v', 'current_a'] })
    expect(JSON.stringify(snapshot)).toBe(original)
  })
  it('invalidates held responses as soon as a filter changes', async () => {
    let release
    screen.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await load()
    await wrapper.get('[aria-label="Part or manufacturer"]').setValue('changed')
    release(result); await flushPromises()
    expect(wrapper.find('.summary-cards').exists()).toBe(false)
    await vi.advanceTimersByTimeAsync(200); await flushPromises()
    expect(wrapper.find('.summary-cards').exists()).toBe(true)
  })
  it('keeps thermal assumptions, currency and chip area unknown until each explicit declaration', async () => {
    await load()
    expect(screen.mock.lastCall[0]).not.toHaveProperty('workbook_derivation')
    expect(wrapper.get('[aria-label="Declared currency A"]').element.value).toBe('')
    expect(wrapper.get('[aria-label="Declared area unit A"]').element.value).toBe('')
    await wrapper.get('[aria-label="Acknowledge thermal screening assumptions"]').setValue(true)
    await wrapper.get('[aria-label="Declared currency A"]').setValue('chf')
    await wrapper.get('[aria-label="Declared area unit A"]').setValue('mm2')
    await vi.advanceTimersByTimeAsync(200); await flushPromises()
    expect(screen.mock.lastCall[0].workbook_derivation).toEqual({ thermal_convention: 'source-constant-model-100C', source_declarations: [{ source_id: 'A', currency: 'CHF', area_unit: 'mm2' }] })
    expect(metadata.currency).toBe('CHF')
    expect(wrapper.text()).toContain('not a maximum switching frequency')
    await wrapper.get('[aria-label="Catalog sources"]').setValue([])
    await vi.advanceTimersByTimeAsync(200)
    expect(wrapper.find('.summary-cards').exists()).toBe(false)
  })
  it('shows true zero separately from unavailable benchmark inputs and exports the calculation proof', async () => {
    const derivedMetrics = { ...metrics, thermal_current_a: { label: 'Thermal current screening estimate', unit: 'A', direction: 'max' }, voltage_energy_benchmark_hz: { label: 'Voltage / energy coefficient benchmark', unit: 'Hz', direction: 'max' } }
    const evidence = { declarations: { area_unit: null }, source_conditions: { conduction_temperature_c: 175, gate_voltage_v: 15 }, metric_evidence: { thermal_current_a: { status: 'available', value: 0 }, voltage_energy_benchmark_hz: { status: 'unavailable', reason: 'missing recovery coefficient' } } }
    const derived = { ...row, metrics: { ...row.metrics, thermal_current_a: 0, voltage_energy_benchmark_hz: null }, metric_exclusions: { voltage_energy_benchmark_hz: 'missing recovery coefficient' }, workbook_evidence: evidence }
    sources.mockResolvedValue({ sources: [metadata], metrics: derivedMetrics, facets: {} })
    screen.mockResolvedValue({ ...result, candidates: [derived] })
    await load()
    await wrapper.get('[aria-label="X metric"]').setValue('thermal_current_a')
    await wrapper.get('[aria-label="Y metric"]').setValue('voltage_energy_benchmark_hz')
    await vi.advanceTimersByTimeAsync(200); await flushPromises()
    await wrapper.get('.part-button').trigger('click')
    const detail = wrapper.get('[aria-label="Selected metric evidence"]')
    expect(detail.findAll('tbody tr')[0].findAll('td')[1].text()).toBe('0')
    expect(detail.findAll('tbody tr')[1].findAll('td')[1].text()).toBe('Missing')
    expect(detail.text()).toContain('missing recovery coefficient')
    expect(detail.text()).toContain('conduction_temperature_c')
    expect(wrapper.get('[aria-label="X objective"]').element.value).toBe('max')
    const csv = catalogCsv({ ...result, workbook_derivation: { thermal_convention: 'source-constant-model-100C' } }, derivedMetrics, [derived])
    expect(csv).toContain('"workbook_evidence_json","metric_exclusions_json","workbook_derivation_json"')
    expect(csv).toContain('missing recovery coefficient')
    expect(csv).toContain('source-constant-model-100C')
  })
  it('exports original source conditions and reproducible normalization evidence', () => {
    const csv = catalogCsv(result, metrics, [row])
    expect(csv).toContain('"\'=literal"')
    expect(csv).toContain('"voltage_v_population_ids"')
    expect(csv).toContain('"[""A0"",""A1""]"')
    expect(csv).toContain('25 C')
    expect(csv).toContain(metadata.sha256)
    expect(csv).toContain('"4","8","2","2","hash"')
    const unknownCurrency = catalogCsv(result, metrics, [{ ...row, currency: null }])
    expect(unknownCurrency).toContain(`${metadata.sha256}\",\"\",\"historical-A`)
  })
  it('stores shared normalization lists once for thousands of rows and retains empty export evidence', () => {
    const rows = Array.from({ length: 3000 }, (_, i) => ({ ...row, id: `unique-population-id-${i}` }))
    const keys = Object.fromEntries(Array.from({ length: 13 }, (_, i) => [`metric_${i}`, {}]))
    const evidence = { denominator: 2, denominator_ids: rows.map(r => r.id), population_ids: rows.map(r => r.id), population_hash: 'a'.repeat(64) }
    const large = { ...result, normalization: { mode: 'fixed', metrics: Object.fromEntries(Object.keys(keys).map(k => [k, evidence])) } }
    const csv = catalogCsv(large, keys, rows)
    expect(csv.length).toBeLessThan(8_000_000)
    expect(csv.match(/unique-population-id-2999/g)).toHaveLength(27) // One candidate ID plus two lists per metric.
    const empty = catalogCsv(large, keys, [])
    expect(empty).toContain('"evidence_only","full_ids"')
    expect(empty).toContain('unique-population-id-2999')
  })
})
