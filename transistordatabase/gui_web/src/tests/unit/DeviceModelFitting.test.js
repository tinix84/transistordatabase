import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import DeviceModelFitting from '../../components/DeviceModelFitting.vue'
import DeviceModelFitPlot from '../../components/DeviceModelFitPlot.vue'
import { fittingFixture, modelHeader, parseObservations } from '../../components/deviceModelInput.js'

const { train, evaluate, evaluateModelJson, download } = vi.hoisted(() => ({ train: vi.fn(), evaluate: vi.fn(), evaluateModelJson: vi.fn(), download: vi.fn() }))
vi.mock('../../services/api.js', () => ({ deviceModelApi: { train, evaluate, evaluateModelJson } }))
vi.mock('../../components/analysisFiles.js', async original => ({ ...await original(), downloadAnalysis: download }))
const row = { id: 'r1', source: '=source', series: 'series', observed: .01, predicted: .01, residual: 0, status: 'predicted', reason: null }
const data = () => ({ inputs: { family: 'ron' }, model: { family: 'ron', degree: 1, slices: [], coefficients: [1, 2] }, model_json: '{"family":"ron","coefficients":[1.0,2.0]}',
  diagnostics: [{ output: 'resistance', slice: 'ron', train: { count: 3, rmse: 0, mae: 0, r2: 1, rows: [row] }, holdout: { count: 1, rmse: .00025, mae: .00025, r2: null, rows: [{ ...row, id: 'rh', observed: .0115, predicted: .01125, residual: -.00025 }] } }],
  plots: [{ id: 'ron', output: 'resistance', axis: 't_j_c', x_label: 'Temperature (°C)', y_label: 'Resistance (ohm)', conditions: { v_g_v: null, current_a: null }, support: [25, 125], normalization: null,
    samples: [{ x: 25, y: .01 }, { x: 125, y: .015 }], observations: [{ ...row, x: 25, y: .01, split: 'train' }, { ...row, id: 'rh', x: 50, y: .0115, split: 'holdout' }] }],
  scope: 'Bounded estimate', units: { resistance: 'ohm', voltage: 'V', current: 'A', energy: 'J' }, evidence: { model_sha256: 'a'.repeat(64), result_sha256: 'b'.repeat(64) } })
const envelope = value => ({ data: value, rawJson: JSON.stringify(value) })
let wrapper
const field = name => wrapper.get(`[data-testid=fit-${name}]`)
async function fixture(family = 'ron') { wrapper = mount(DeviceModelFitting); await field('family').setValue(family); await field('example').trigger('click') }
async function submit() { await wrapper.findAll('form')[0].trigger('submit'); await flushPromises() }
async function predict() { await wrapper.findAll('form')[1].trigger('submit'); await flushPromises() }
beforeEach(() => { train.mockReset(); evaluate.mockReset(); evaluateModelJson.mockReset(); download.mockReset(); train.mockResolvedValue(envelope(data())) })
afterEach(() => wrapper?.unmount())

describe('device model raw observation input', () => {
  it('keeps explicit per-row provenance and train/holdout labels', () => {
    const fixture = fittingFixture('ron'), parsed = parseObservations(fixture.csv, 'ron', fixture.source, fixture.series)
    expect(parsed).toHaveLength(4); expect(parsed[3]).toMatchObject({ id: 'rh', split: 'holdout', coordinates: { t_j_c: 50 }, value: .0115 })
    const csv = `${modelHeader('ron').join(',')}\nx,"Source, manual",separate,train,-5,0e-400`
    expect(parseObservations(csv, 'ron', '', '')[0]).toMatchObject({ source: 'Source, manual', series: 'separate', coordinates: { t_j_c: -5 }, value: 0 })
  })
  it.each(['', 'NaN', 'Infinity', '0x10', '0b10', '1e-400', '1.2bad'])('rejects invalid numeric targets without coercion: %s', value => {
    expect(() => parseObservations(`id,source,series,split,t_j_c,value\nx,s,z,train,25,${value}`, 'ron', '', '')).toThrow()
  })
  it('rejects missing labels, duplicate IDs and implicit splits', () => {
    for (const rows of ['x,,,train,25,.01', 'x,s,z,,25,.01', 'x,s,z,train,25,.01\nx,s,z,holdout,50,.02']) {
      expect(() => parseObservations(`id,source,series,split,t_j_c,value\n${rows}`, 'ron', '', '')).toThrow()
    }
  })
  it('keeps unknown diode temperature separate from zero and rejects unknown variable temperatures', () => {
    const header = modelHeader('diode').join(',')
    expect(parseObservations(`${header}\nx,s,z,train,1,,null,.5`, 'diode', '', '')[0].coordinates).toEqual({ current_a: 1, t_j_c: null, v_g_v: null })
    expect(parseObservations(`${header}\nx,s,z,train,1,0,0,.5`, 'diode', '', '')[0].coordinates).toEqual({ current_a: 1, t_j_c: 0, v_g_v: 0 })
    expect(() => parseObservations('id,source,series,split,t_j_c,value\nx,s,z,train,,.01', 'ron', '', '')).toThrow()
  })
})

describe('model fitting workbench', () => {
  it('sends explicit degree, raw observations and unknown fixed conditions, with no repository mutation', async () => {
    await fixture(); await submit()
    expect(train.mock.calls[0][0]).toMatchObject({ family: 'ron', degree: 1, event: null, anchor: null, fixed_conditions: { v_g_v: null, current_a: null } })
    expect(train.mock.calls[0][0].observations).toHaveLength(4)
    expect(field('metrics-holdout').text()).toContain('0.000250000'); expect(field('metrics-holdout').text()).toContain('Unavailable')
    expect(field('residuals').text()).toContain('rh / holdout'); expect(wrapper.findAll('form form')).toHaveLength(0)
    expect(wrapper.findComponent(DeviceModelFitPlot).props('plot').samples).toEqual(data().plots[0].samples)
  })
  it('uses separate energy events and explicit anchor without synthetic observations', async () => {
    await fixture('energy'); await field('event').setValue('e_off'); await submit()
    expect(train.mock.calls[0][0]).toMatchObject({ family: 'energy', event: 'e_off', anchor: { current_a: 20, voltage_v: 400, t_j_c: 25, r_g_ohm: 4 }, fixed_conditions: { v_g_v: null } })
    expect(train.mock.calls[0][0].observations).toHaveLength(10)
    expect(train.mock.calls[0][0].observations.at(-1).value).toBe(.002678125)
  })
  it('distinguishes null and exact zero and keeps raw malformed condition text for validation', async () => {
    await fixture(); await field('condition-v_g_v').setValue('0'); await submit()
    expect(train.mock.calls[0][0].fixed_conditions.v_g_v).toBe(0)
    await field('condition-v_g_v').setValue('0x10'); await submit(); expect(train).toHaveBeenCalledTimes(1)
    expect(wrapper.get('[role=alert]').text()).toContain('finite decimal')
  })
  it('evaluates only through the API and displays rejected null without a zero or query marker', async () => {
    await fixture(); await submit()
    const coordinates = { t_j_c: 50, v_g_v: null, current_a: null }
    evaluate.mockResolvedValue(envelope({ inputs: { queries: [{ id: 'query-1', coordinates }] }, result: { predictions: [{ id: 'query-1', output: 'resistance', status: 'rejected', value: null, unit: 'ohm', reason: 'Outside support' }] } }))
    await predict(); expect(evaluate).toHaveBeenCalledWith({ model: data().model, queries: [{ id: 'query-1', output: 'resistance', coordinates }] })
    expect(field('prediction').text()).toContain('Unavailable ohm · rejected'); expect(wrapper.findComponent(DeviceModelFitPlot).props('queries')).toEqual([])
  })
  it('uses a separate inverse diode output with exact temperature and gate slice', async () => {
    await fixture('diode'); await submit(); await field('query-output').setValue('current'); await field('query-v_g_v').setValue('0')
    evaluate.mockResolvedValue(envelope({ inputs: {}, result: { predictions: [] } })); await predict()
    expect(evaluate.mock.calls[0][0].queries[0]).toEqual({ id: 'query-1', output: 'current', coordinates: { voltage_v: .7, t_j_c: 25, v_g_v: 0 } })
    await field('query-t_j_c').setValue(''); await predict(); expect(evaluate.mock.calls[1][0].queries[0].coordinates.t_j_c).toBeNull()
  })
  it('clears fit exports and discards late fit responses on edits', async () => {
    await fixture(); await submit(); await field('source').setValue('Changed'); expect(wrapper.find('[data-testid=fit-results]').exists()).toBe(false)
    let release; train.mockImplementationOnce(() => new Promise(resolve => { release = resolve })); await wrapper.findAll('form')[0].trigger('submit')
    await field('degree').setValue('2'); release(envelope(data())); await flushPromises()
    expect(wrapper.find('[data-testid=fit-results]').exists()).toBe(false); expect(field('stale').text()).toContain('Inputs changed')
  })
  it('discards stale query responses when coordinates or training context change', async () => {
    await fixture(); await submit(); let release
    evaluate.mockImplementationOnce(() => new Promise(resolve => { release = resolve })); await wrapper.findAll('form')[1].trigger('submit')
    await field('query-t_j_c').setValue('60'); release(envelope({ result: { predictions: [] } })); await flushPromises()
    expect(wrapper.find('[data-testid=fit-predictions]').exists()).toBe(false)
    evaluate.mockImplementationOnce(() => new Promise(resolve => { release = resolve })); await wrapper.findAll('form')[1].trigger('submit')
    await field('condition-v_g_v').setValue('15'); release(envelope({ result: { predictions: [] } })); await flushPromises()
    expect(wrapper.find('[data-testid=fit-results]').exists()).toBe(false)
  })
  it('downloads exact review bytes, model, and safely quoted residual provenance', async () => {
    await fixture(); const accepted = data(), rawJson = JSON.stringify(accepted).replace('0.01', '0.01000'); train.mockResolvedValueOnce({ data: accepted, rawJson }); await submit()
    await field('json').trigger('click'); expect(download).toHaveBeenLastCalledWith(rawJson, 'device-model-training-review.json', 'application/json;charset=utf-8')
    await field('model-json').trigger('click'); expect(download).toHaveBeenLastCalledWith(accepted.model_json, 'device-model.json', 'application/json;charset=utf-8')
    await field('csv').trigger('click'); expect(download.mock.calls.at(-1)[0]).toContain('"\'=source"'); expect(download.mock.calls.at(-1)[0]).toContain(accepted.evidence.model_sha256)
  })
  it('imports only data, clears fitting claims, and preserves untrusted model numeric text for backend validation', async () => {
    await fixture(); await submit()
    const rawModel = '{"family":"ron","coefficient":1e-999,"negative_zero":-0.0}'
    await field('import-json').setValue(rawModel); expect(wrapper.find('[data-testid=fit-results]').exists()).toBe(false)
    await field('import').trigger('click'); expect(field('import-status').text()).toContain('next request validates')
    evaluateModelJson.mockRejectedValueOnce({ response: { data: { detail: 'Model numeric value underflows' } } }); await predict()
    expect(evaluateModelJson).toHaveBeenCalledWith(rawModel, [{ id: 'query-1', output: 'resistance', coordinates: { t_j_c: 50, v_g_v: null, current_a: null } }])
    expect(wrapper.get('[role=alert]').text()).toContain('underflows'); expect(wrapper.find('[data-testid=fit-predictions]').exists()).toBe(false)
    await field('import-json').setValue('{"family":"diode"}'); expect(wrapper.find('[data-testid=fit-import-status]').exists()).toBe(false)
  })
})

it('renders backend samples, distinct split markers and padded query coordinates without inventing rejected points', () => {
  wrapper = mount(DeviceModelFitPlot, { props: { plot: data().plots[0], queries: [{ x: 50, y: .01125 }, { x: null, y: null }] } })
  expect(wrapper.findAll('[data-kind=train]')).toHaveLength(1); expect(wrapper.findAll('[data-kind=holdout]')).toHaveLength(1); expect(wrapper.findAll('[data-kind=query]')).toHaveLength(1)
  expect(wrapper.get('[data-kind=train]').attributes('cx')).not.toBe('88')
  expect(wrapper.get('svg').attributes('aria-label')).toContain('Temperature')
})
