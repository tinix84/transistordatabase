import { afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { grid } = vi.hoisted(() => ({ grid: vi.fn() }))
vi.mock('../../services/api.js', () => ({ analysisApi: { grid } }))
import GridWorkbench from '../../components/GridWorkbench.vue'
import MeritAnalysisEvidence from '../../components/MeritAnalysisEvidence.vue'

const objective = (name, weight, minimum, maximum) => ({ name, unit: 'W', direction: 'minimize', weight,
  normalized_weight: weight / 4, active: true, minimum, maximum,
  minimum_configuration_ids: [name === 'conduction_power_w' ? 'B' : 'A'], maximum_configuration_ids: [name === 'conduction_power_w' ? 'A' : 'B'] })
const metric = (name, value, penalty, contribution) => ({ name, unit: 'W', full_mean: value, complete: true,
  coverage: 1, retained_weight: 4, total_weight: 4, missing_operating_point_ids: [], penalty, contribution })
const merit = {
  method_version: 'waveform-grid-merit-v1', grid_input_sha256: 'grid-hash', population_sha256: 'population-hash', status: 'computed',
  population: { configuration_ids: ['A', 'B'], operating_point_ids: ['high', 'low'], zero_weight_operating_point_ids: ['zero'],
    configuration_coverage: 1, configuration_count: 2, eligible_configuration_count: 2 },
  objectives: [objective('conduction_power_w', 1, 20, 25), objective('turn_on_power_w', 3, 4, 8)],
  rows: [{ configuration_id: 'A', status: 'computed', score: .25, rank: 1, exclusion_reasons: [],
    objectives: [metric('conduction_power_w', 25, 1, .25), metric('turn_on_power_w', 4, 0, 0)] },
  { configuration_id: 'B', status: 'computed', score: .75, rank: 2, exclusion_reasons: [],
    objectives: [metric('conduction_power_w', 20, 0, 0), metric('turn_on_power_w', 8, 1, .75)] }],
  ranking: [{ configuration_id: 'A', score: .25, rank: 1 }, { configuration_id: 'B', score: .75, rank: 2 }],
  rows_json: '[{"score":-0.0,"missing":null}]', rows_csv: '"score","missing"\r\n"0","NULL"\r\n', null_csv: 'NULL',
}
let wrapper
afterEach(() => { wrapper?.unmount(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); vi.clearAllMocks() })
const options = { global: { stubs: { WaveformAnalysisView: true, MeritAnalysisEvidence: true } } }
async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }

describe('weighted merit comparison', () => {
  it('omits merit by default and validates explicit guided objectives before transport', async () => {
    grid.mockRejectedValue(new Error('Request inspected without a response fixture'))
    wrapper = mount(GridWorkbench, options)
    await submit(); expect(grid.mock.calls[0][0]).not.toHaveProperty('objectives')
    await wrapper.findAll('label').find(label => label.text() === 'Compare weighted merit penalties').get('input').setValue(true)
    await submit()
    expect(grid.mock.calls.at(-1)[0].objectives).toEqual([
      { name: 'conduction_power_w', unit: 'W', direction: 'minimize', weight: 1 },
      { name: 'turn_on_power_w', unit: 'W', direction: 'minimize', weight: 3 },
    ])
    await wrapper.get('[aria-label="Merit objective 1 weight"]').setValue('1e-999'); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('too small for a nonzero numeric value')
    expect(grid).toHaveBeenCalledTimes(2)
    await wrapper.get('[aria-label="Merit objective 1 weight"]').setValue('-1'); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('nonnegative')
    await wrapper.get('[aria-label="Merit objective 1 weight"]').setValue('0')
    await wrapper.get('[aria-label="Merit objective 2 weight"]').setValue('0'); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('positive weight')
    await wrapper.get('[aria-label="Merit objective 2 weight"]').setValue('3')
    await submit(); expect(grid.mock.calls.at(-1)[0].objectives[0].weight).toBe(0)
    await wrapper.get('[aria-label="Merit objective 2 metric"]').setValue('conduction_power_w'); await submit()
    expect(wrapper.get('[role=alert]').text()).toContain('different power metric')
    expect(grid).toHaveBeenCalledTimes(3)
  })
  it('renders returned dimensionless penalties, raw means, full coverage, and population identity', () => {
    wrapper = mount(MeritAnalysisEvidence, { props: { merit } })
    expect(wrapper.get('[data-testid=merit-ranking]').text()).toContain('Rank 1 · A · penalty 0.250000')
    expect(wrapper.get('[data-testid=merit-ranking]').text()).toContain('Rank 2 · B · penalty 0.750000')
    expect(wrapper.get('[data-score="0.25"]').attributes('style')).toContain('25%')
    expect(wrapper.get('[data-testid=merit-configurations]').text()).toContain('25.0000 W')
    expect(wrapper.get('[data-testid=merit-configurations]').text()).toContain('contribution 0.250000')
    expect(wrapper.get('[data-testid=merit-population]').text()).toContain('high, low')
    expect(wrapper.text()).toContain('population-hash'); expect(wrapper.text()).toContain('grid-hash')
  })
  it('rejects a positive operating-point weight that underflows and preserves explicit zero', async () => {
    grid.mockRejectedValue(new Error('Request inspected without a response fixture'))
    wrapper = mount(GridWorkbench, options)
    await wrapper.findAll('button').find(button => button.text() === 'Add operating point').trigger('click')
    await wrapper.get('[aria-label="Point 1 weight"]').setValue('1e-999'); await submit()
    expect(grid).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('too small for a nonzero numeric value')
    await wrapper.get('[aria-label="Point 1 weight"]').setValue('0'); await submit()
    expect(grid.mock.calls[0][0].operating_points.map(point => point.weight)).toEqual([0, 1])
  })
  it('sends the advanced editor as raw JSON without rounding a tiny positive weight to zero', async () => {
    grid.mockRejectedValue({ response: { data: { detail: 'Numeric value is outside representable precision' } } })
    wrapper = mount(GridWorkbench, options)
    await wrapper.findAll('label').find(label => label.text() === 'Use grid JSON as input').get('input').setValue(true)
    const raw = '{"objectives":[{"weight":1e-999}]}'
    await wrapper.get('[aria-label="Grid request JSON"]').setValue(raw); await submit()
    expect(grid.mock.calls[0][0]).toBe(raw)
    expect(wrapper.get('[role=alert]').text()).toContain('outside representable precision')
  })
  it('keeps excluded rows and missing points visible when no population is eligible', () => {
    const excluded = { ...merit.rows[0], status: 'excluded', score: null, rank: null,
      exclusion_reasons: [{ name: 'turn_on_power_w', missing_operating_point_ids: ['high'] }],
      objectives: [{ ...merit.rows[0].objectives[1], full_mean: null, complete: false, penalty: null, contribution: null, coverage: .25, missing_operating_point_ids: ['high'] }] }
    wrapper = mount(MeritAnalysisEvidence, { props: { merit: { ...merit, status: 'unavailable', ranking: [], rows: [excluded] } } })
    expect(wrapper.find('[data-testid=merit-ranking]').exists()).toBe(false)
    expect(wrapper.text()).toContain('Scores and ranks are unavailable')
    expect(wrapper.get('[data-testid=merit-configurations]').text()).toContain('missing high')
    expect(wrapper.get('[data-testid=merit-configurations]').text()).toContain('Rank Unavailable')
  })
  it('bounds table rows and downloads exact server text including zero and null', async () => {
    vi.useFakeTimers()
    const blobs = []; vi.stubGlobal('Blob', class { constructor(parts, options) { this.parts = parts; this.type = options.type } })
    vi.stubGlobal('URL', { createObjectURL: vi.fn(blob => { blobs.push(blob); return 'blob:test' }), revokeObjectURL: vi.fn() })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    wrapper = mount(MeritAnalysisEvidence, { props: { merit: { ...merit, rows: Array.from({ length: 33 }, (_, index) => ({ ...merit.rows[0], configuration_id: `A-${index}` })) } } })
    expect(wrapper.get('[data-testid=merit-configurations]').findAll('tbody tr')).toHaveLength(32)
    await wrapper.findAll('button').find(button => button.text() === 'Next merit rows').trigger('click')
    expect(wrapper.get('[data-testid=merit-configurations]').findAll('tbody tr')).toHaveLength(1)
    await wrapper.findAll('button').find(button => button.text() === 'Download merit JSON').trigger('click')
    await wrapper.findAll('button').find(button => button.text() === 'Download merit CSV').trigger('click')
    expect(blobs.map(blob => blob.parts[0])).toEqual([merit.rows_json, merit.rows_csv])
    expect(blobs.map(blob => blob.type)).toEqual(['application/json', 'text/csv'])
    vi.runAllTimers()
  })
  it('invalidates merit controls and discards late replies after objective edits', async () => {
    const response = { metric: { name: 'conduction_power_w' }, inputs: { configurations: [], operating_points: [] }, rows: [], ranking: [], summaries: [], merit }
    let resolve; grid.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    wrapper = mount(GridWorkbench, options)
    await wrapper.findAll('label').find(label => label.text() === 'Compare weighted merit penalties').get('input').setValue(true)
    await wrapper.get('form').trigger('submit')
    await wrapper.get('[aria-label="Merit objective 1 direction"]').setValue('maximize')
    resolve(response); await flushPromises()
    expect(wrapper.find('[data-testid=grid-results]').exists()).toBe(false)
    grid.mockResolvedValueOnce(response); await submit()
    expect(wrapper.findComponent(MeritAnalysisEvidence).exists()).toBe(true)
    await wrapper.get('[aria-label="Merit objective 1 weight"]').setValue('0')
    expect(wrapper.findComponent(MeritAnalysisEvidence).exists()).toBe(false)
  })
})
