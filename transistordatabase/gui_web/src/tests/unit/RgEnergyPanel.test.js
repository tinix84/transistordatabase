import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { inventory, calculate, getAll } = vi.hoisted(() => ({ inventory: vi.fn(), calculate: vi.fn(), getAll: vi.fn() }))
vi.mock('../../services/api.js', () => ({ rgEnergyApi: { inventory, calculate }, transistorApi: { getAll } }))
import RgEnergyPanel from '../../components/RgEnergyPanel.vue'

const conditions = { v_supply: 400, v_g: 12, v_g_off: null, load_inductance: null, commutation_inductance: null, commutation_device: null, convention: null, event: null, method: 'unknown' }
const curve = (index, changes = {}) => ({ source_id: `sample:switch.e_on_data[${index}]`, source_kind: 'repository-locator', branch: 'e_on', index, representation: 'graph_r_e', t_j: 75, i_x: 10, r_g: null,
  conditions: { ...conditions }, source_filename: null, source_sha256: null, snapshot_sha256: `snapshot-${index}`,
  coverage: { axis: 'r_g', axis_unit: 'ohm', lower: 1, upper: 9, energy_min: .002, energy_max: .008, sample_count: 3 },
  preview: { axis: [1, 4, 9], energy_j: [.008, .002, .006], truncated: false }, ...changes })
const baseInventory = { device_id: 'sample', curves: [curve(0), curve(1, { t_j: 125 }), curve(2, { conditions: { ...conditions, v_g: 0, method: null } }), curve(3, { conditions: { ...conditions, v_g: null, method: null } })] }
const response = { device_id: 'sample', operation: 'energy-at-rg', value: .007, unit: 'J', evidence_tag: 'bounded-interpolation-estimate',
  query: { operation: 'energy-at-rg', branch: 'e_on', representation: 'graph_r_e', i_channel: 10, t_j: 75, conditions, r_g: 1.5 }, conditions,
  source_scope: 'considered-branch', sources: [curve(0), curve(1, { t_j: 125 })], method: 'piecewise-linear; no extrapolation' }

describe('repository-backed Rg energy child', () => {
  let wrapper
  const field = name => wrapper.get(`[data-testid=rg-${name}]`)
  beforeEach(async () => {
    getAll.mockReset().mockResolvedValue([{ metadata: { name: 'sample' } }, { id: 'other', name: 'Other sample' }])
    inventory.mockReset().mockResolvedValue(structuredClone(baseInventory)); calculate.mockReset().mockResolvedValue({ data: response, rawJson: JSON.stringify(response) })
    wrapper = mount(RgEnergyPanel, { global: { stubs: { CurvePlotter: true } } }); await flushPromises()
  })
  afterEach(() => wrapper.unmount())
  async function selectDevice() { await field('device').setValue('sample'); await flushPromises() }
  async function ready() {
    await selectDevice(); await field('context').setValue(field('context').findAll('option')[1].element.value)
    await field('current').setValue('10'); await field('temperature').setValue('75'); await field('query').setValue('1.5')
  }
  async function submit() { await wrapper.get('form').trigger('submit'); await flushPromises() }
  it('requires device and context selection without selecting the first recorded curve', async () => {
    expect(inventory).not.toHaveBeenCalled(); expect(field('device').element.value).toBe('')
    await selectDevice(); expect(field('context').element.value).toBe(''); expect(field('context').findAll('option')).toHaveLength(4)
    await submit(); expect(calculate).not.toHaveBeenCalled(); expect(field('error').text()).toContain('exact recorded context')
  })
  it('groups all records by exact conditions and preserves null, known zero and literal unknown', async () => {
    await ready(); expect(wrapper.findAll('table tbody tr')).toHaveLength(2)
    expect(field('conditions').text()).toContain('"unknown"'); expect(field('conditions').text()).toContain('Unknown / not recorded (null)')
    await submit(); expect(calculate).toHaveBeenCalledWith('sample', response.query)
    expect(Object.keys(calculate.mock.calls[0][1].conditions)).toHaveLength(9)
    expect(wrapper.text()).toContain('Source SHA-256: Not recorded'); expect(wrapper.text()).toContain('repository-locator'); expect(wrapper.text()).toContain('snapshot-0')
    await field('context').setValue(field('context').findAll('option')[2].element.value); await submit()
    expect(calculate.mock.calls.at(-1)[1].conditions).toMatchObject({ v_g: 0, method: null })
    await field('context').setValue(field('context').findAll('option')[3].element.value); await submit()
    expect(calculate.mock.calls.at(-1)[1].conditions).toMatchObject({ v_g: null, method: null })
  })
  it('plots only source points and the returned point with padded marker bounds', async () => {
    await ready(); await submit(); const props = wrapper.findComponent({ name: 'CurvePlotter' }).props()
    expect(props.datasets).toHaveLength(3); expect(props.datasets[0].data).toEqual([{ x: 1, y: .008 }, { x: 4, y: .002 }, { x: 9, y: .006 }])
    expect(props.datasets.at(-1).data).toEqual([{ x: 1.5, y: .007 }]); expect(props.allowZoom).toBe(false)
    expect(props.xMin).toBeLessThan(1); expect(props.xMax).toBeGreaterThan(9); expect(props.yMin).toBeLessThan(.002); expect(props.yMax).toBeGreaterThan(.008)
    expect(field('results').text()).toContain('bounded-interpolation-estimate'); expect(field('value').text()).toContain('0.00700000 J')
  })
  it('sends inverse SI targets and preserves a genuine zero response', async () => {
    await ready(); await field('operation').setValue('rg-at-energy'); await field('query').setValue('.007'); await submit()
    expect(calculate.mock.calls.at(-1)[1]).toMatchObject({ operation: 'rg-at-energy', e_target: .007 }); expect(calculate.mock.calls.at(-1)[1]).not.toHaveProperty('r_g')
    calculate.mockResolvedValueOnce({ data: { ...response, value: 0 }, rawJson: 'zero' }); await submit(); expect(field('value').text()).toBe('0.00000 J')
  })
  it('replaces inventory previews and source hashes with the accepted result snapshots', async () => {
    await ready()
    const updated = curve(0, { snapshot_sha256: 'updated-snapshot', source_filename: 'updated.csv', preview: { axis: [1, 4, 9], energy_j: [.009, .003, .007], truncated: false } })
    calculate.mockResolvedValueOnce({ data: { ...response, sources: [updated] }, rawJson: 'updated' }); await submit()
    expect(wrapper.get('table').text()).toContain('updated-snapshot'); expect(wrapper.get('table').text()).toContain('updated.csv')
    expect(wrapper.get('table').text()).not.toContain('snapshot-1'); expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')[0].data[0]).toEqual({ x: 1, y: .009 })
    expect(wrapper.findAll('table tbody tr')).toHaveLength(1)
  })
  it('retains returned nearby recorded contexts without reimplementing core matching tolerances', async () => {
    await ready()
    const near = curve(4, { conditions: { ...conditions, v_supply: 400.0000000005 }, snapshot_sha256: 'nearby-context' })
    calculate.mockResolvedValueOnce({ data: { ...response, sources: [...response.sources, near] }, rawJson: 'nearby' }); await submit()
    expect(wrapper.get('table').text()).toContain('nearby-context'); expect(wrapper.get('table').text()).toContain('400.0000000005')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')).toHaveLength(4)
    expect(wrapper.findAll('table tbody tr')).toHaveLength(3)
  })
  it('keeps unavailable source metadata and diagnostics without selectable contexts, false zero, or plot points', async () => {
    const unavailable = curve(5, { status: 'unavailable', diagnostic: 'Invalid recorded graph at Tj=100', conditions: null, t_j: 100,
      preview: null, coverage: null, snapshot_sha256: null, unavailable_fields: ['conditions', 'coverage', 'preview', 'snapshot_sha256'] })
    inventory.mockResolvedValueOnce({ device_id: 'sample', curves: [...baseInventory.curves, unavailable] }); await ready()
    expect(field('context').findAll('option')).toHaveLength(4)
    expect(field('unavailable-records').text()).toContain('Invalid recorded graph at Tj=100')
    calculate.mockResolvedValueOnce({ data: { ...response, sources: [...response.sources, unavailable] }, rawJson: 'unavailable-proof' }); await submit()
    expect(wrapper.findAll('table tbody tr')).toHaveLength(3); expect(wrapper.get('table').text()).toContain('Context unavailable')
    expect(wrapper.get('table').text()).toContain('Unavailable: Invalid recorded graph'); expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')).toHaveLength(3)
    inventory.mockResolvedValueOnce({ device_id: 'other', curves: [unavailable] }); await field('device').setValue('other'); await flushPromises()
    expect(wrapper.find('[data-testid=rg-context]').exists()).toBe(false); expect(wrapper.text()).toContain('No selectable context')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).exists()).toBe(false)
  })
  it('keeps a usable curve selectable when only its source metadata is unavailable', async () => {
    const warning = curve(0, { status: 'available', diagnostic: 'Invalid source SHA-256', source_sha256: null, unavailable_fields: ['source_sha256'] })
    inventory.mockResolvedValueOnce({ device_id: 'sample', curves: [warning] }); await ready()
    expect(wrapper.text()).toContain('Source metadata diagnostic'); expect(wrapper.text()).toContain('Invalid source SHA-256')
    expect(wrapper.text()).toContain('Unavailable fields: source_sha256'); expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')).toHaveLength(1)
    await submit(); expect(calculate).toHaveBeenCalled()
  })
  it('rejects blank, nondecimal, nonfinite and underflowing raw inputs before transport', async () => {
    await ready()
    for (const [name, value] of [['current', ''], ['current', '0x10'], ['current', 'Infinity'], ['current', '1e-999'], ['temperature', ''], ['query', '1e-999']]) {
      await field('current').setValue('10'); await field('temperature').setValue('75'); await field('query').setValue('1.5'); calculate.mockClear()
      await field(name).setValue(value); await submit(); expect(calculate).not.toHaveBeenCalled(); expect(field('error').exists()).toBe(true)
    }
    await field('query').setValue('0'); await field('temperature').setValue('0'); await field('current').setValue('0'); await submit()
    expect(calculate.mock.calls.at(-1)[1]).toMatchObject({ i_channel: 0, t_j: 0, r_g: 0 })
  })
  it('retains core ambiguity and bounded-domain diagnostics without choosing a first source', async () => {
    await ready(); calculate.mockRejectedValueOnce({ response: { status: 422, data: { detail: 'Multiple Rg crossings exist.' } } }); await submit()
    expect(field('error').text()).toContain('Multiple Rg crossings'); expect(wrapper.find('[data-testid=rg-results]').exists()).toBe(false)
    await field('query').setValue('10'); expect(wrapper.find('[data-testid=rg-error]').exists()).toBe(false)
    calculate.mockRejectedValueOnce({ response: { data: { detail: [{ msg: 'Outside recorded domain' }] } } }); await submit(); expect(field('error').text()).toContain('Outside recorded domain')
  })
  it('invalidates result and exports on query, context, branch, representation, and inventory refresh changes', async () => {
    await ready()
    for (const [name, value] of [['current', '11'], ['temperature', '80'], ['query', '2'], ['context', '']]) {
      await submit(); expect(field('results').exists()).toBe(true); await field(name).setValue(value)
      expect(wrapper.find('[data-testid=rg-results]').exists()).toBe(false); expect(wrapper.find('[data-testid=rg-json]').exists()).toBe(false)
    }
    await field('context').setValue(field('context').findAll('option')[1].element.value); await submit()
    await wrapper.findAll('button').find(button => button.text() === 'Refresh repository data').trigger('click'); await flushPromises()
    expect(field('context').element.value).toBe(''); expect(wrapper.find('[data-testid=rg-results]').exists()).toBe(false)
    await field('context').setValue(field('context').findAll('option')[1].element.value); await submit(); await field('representation').setValue('graph_i_e')
    expect(wrapper.find('[data-testid=rg-results]').exists()).toBe(false); expect(wrapper.text()).toContain('exact recorded Rg')
    await field('branch').setValue('e_rr'); expect(wrapper.text()).toContain('No recorded energy curves')
  })
  it('discards a pending calculation after context or device selection changes', async () => {
    await ready(); let resolve
    calculate.mockImplementationOnce(() => new Promise(done => { resolve = done })); await wrapper.get('form').trigger('submit')
    await field('context').setValue(''); resolve({ data: response, rawJson: 'old' }); await flushPromises()
    expect(wrapper.find('[data-testid=rg-results]').exists()).toBe(false); expect(wrapper.find('[data-testid=rg-json]').exists()).toBe(false)
  })
  it('discards an old device inventory after the newer device completes', async () => {
    let resolve
    inventory.mockImplementationOnce(() => new Promise(done => { resolve = done })); await field('device').setValue('sample')
    inventory.mockResolvedValueOnce({ device_id: 'other', curves: [] }); await field('device').setValue('other'); await flushPromises()
    resolve(baseInventory); await flushPromises(); expect(wrapper.find('[data-testid=rg-context]').exists()).toBe(false); expect(wrapper.text()).toContain('No recorded energy curves')
  })
})
