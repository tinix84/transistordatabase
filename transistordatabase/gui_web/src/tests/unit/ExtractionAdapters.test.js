import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { post, calculateBiela } = vi.hoisted(() => ({ post: vi.fn(), calculateBiela: vi.fn() }))
vi.mock('../../services/api', () => ({ default: { post }, transistorApi: { calculateBiela } }))
import DeviceParameterCalculator from '../../components/DeviceParameterCalculator.vue'
import BielaCalculator from '../../components/BielaCalculator.vue'
import AnalyticalCalculator from '../../components/AnalyticalCalculator.vue'

const transferInput = { operation: 'transfer-fit', source: 'real transfer source',
  conditions: { v_ds_v: 10, t_j_c: 25 }, v_th_v: 3,
  curves: [2.1, 8.1, 18.1, 32.1, 50.1].map((current, i) => ({ v_g_v: i + 4, t_j_c: 25, voltage_v: [10], current_a: [current] })) }
const transferParameters = { k_1: 2, k_2: .1, x: 2, v_th: 3, gm_i_min: 2.1, gm_i_max: 50.1 }
const transfer = { operation: 'transfer-fit', inputs: transferInput,
  context: { source: transferInput.source, conditions: transferInput.conditions }, units: 'SI', condition_policy: 'Exact',
  result: { ...transferParameters, current_min: 2.1, current_max: 50.1,
    biela_parameters: transferParameters, gate_voltage_v: [4, 5], current_a: [2.1, 8.1],
    fitted_current_a: [2.100000001, 8.1], residual_a: [1e-9, 0], secant_gm_s: [2.1, 4.05],
    diagnostics: { rank: 3, rmse_a: 1e-9, v_ds_v: 10, t_j_c: 25 } },
  plots: [{ title: 'Returned transfer fit', x_label: 'Vgs (V)', y_label: 'Current (A)', x: [4, 5], series: { Fitted: [2.100000001, 8.1] } }] }
const recoveryInput = { operation: 'reverse-recovery-extraction', source: 'real recovery source',
  conditions: { v_supply_v: 100, t_j_c: 25, q_oss_v_ref_v: 100, q_oss_t_j_c: 25 },
  q_rr_c: 100e-9, q_oss_c: 12e-9, i_rr_a: 10, di_dt_a_s: 2.4e9, i_forward_test_a: 20, t_rr_s: null }
const rrParameters = { tau_rr: 6.716666666667e-9, tau_c: 13.669756359629e-9, t_m: 13.204949301187e-9, q_rr_star: 88e-9 }
const recovery = { operation: recoveryInput.operation, inputs: recoveryInput,
  context: { source: recoveryInput.source, conditions: recoveryInput.conditions }, units: 'SI', condition_policy: 'Exact',
  result: { ...rrParameters, biela_parameters: rrParameters, t_rr_used: false, peak_residual_a: 0 }, plots: [] }

describe('transfer and recovery extraction workflows', () => {
  let wrapper
  const options = { global: { stubs: { CurvePlotter: true, BielaEnergyBreakdown: true, GateChargeCalculator: true } } }
  beforeEach(() => { post.mockReset(); calculateBiela.mockReset() })
  afterEach(() => { wrapper?.unmount(); vi.restoreAllMocks() })
  async function calculate() { await wrapper.findAll('button').find(button => button.text() === 'Calculate device parameters').trigger('click'); await flushPromises() }
  it('edits real channel curves without JSON and uses returned fit rows for charts, tables and application', async () => {
    post.mockResolvedValue({ data: transfer })
    wrapper = mount(DeviceParameterCalculator, options)
    await wrapper.get('[aria-label="Parameter calculation"]').setValue('transfer-fit')
    await wrapper.get('[name=source]').setValue('real transfer source')
    await wrapper.get('[aria-label="Curve 1 Vds samples (V)"]').setValue('5, 15')
    await wrapper.get('[aria-label="Curve 1 current samples (A)"]').setValue('1.6, 2.6')
    await calculate()
    expect(post.mock.lastCall[1].curves[0]).toEqual({ v_g_v: 4, t_j_c: 25, voltage_v: [5, 15], current_a: [1.6, 2.6] })
    expect(wrapper.get('[aria-label="Exact transfer fit samples"]').text()).toContain('2.1')
    expect(wrapper.findComponent({ name: 'CurvePlotter' }).props('datasets')[0].data).toEqual([{ x: 4, y: 2.100000001 }, { x: 5, y: 8.1 }])
    expect(wrapper.emitted('apply-biela')).toBeUndefined()
    await wrapper.findAll('button').find(button => button.text().includes('Review extracted')).trigger('click')
    expect(wrapper.emitted('apply-biela')[0][0]).toEqual(transfer)
    await wrapper.get('[name=source]').setValue('changed source')
    expect(wrapper.find('[aria-label="Device parameter results"]').exists()).toBe(false)
  })
  it('converts nC, ns and GA/s fields to SI and exports the complete authoritative envelope', async () => {
    post.mockResolvedValue({ data: recovery })
    wrapper = mount(DeviceParameterCalculator, options)
    await wrapper.get('[aria-label="Parameter calculation"]').setValue('reverse-recovery-extraction')
    await wrapper.get('[name=q_rr_c]').setValue('100')
    await wrapper.get('[name=di_dt_a_s]').setValue('2.4')
    expect(wrapper.get('[name=di_dt_a_s]').element.closest('label').textContent).toContain('Positive current-slope magnitude (GA/s)')
    await wrapper.get('[name=t_rr_s]').setValue('30')
    await calculate()
    expect(post.mock.lastCall[1]).toEqual(expect.objectContaining({ q_rr_c: 100e-9, q_oss_c: 12e-9, di_dt_a_s: 2.4e9, t_rr_s: 30e-9, i_forward_test_a: 20 }))
    expect(wrapper.get('[aria-label="Recovery readable units"]').text()).toContain('6.7166667')
    let blob
    vi.spyOn(URL, 'createObjectURL').mockImplementation(value => { blob = value; return 'blob:fixture' })
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    await wrapper.findAll('button').find(button => button.text() === 'Download reproducible result').trigger('click')
    expect(JSON.parse(await blob.text())).toEqual(recovery)
  })
  it('keeps signed slew result labels in A/s without recovery display scaling', async () => {
    post.mockResolvedValue({ data: { operation: 'gate-slew-estimate', result: { di_dt_a_s: 9e9 }, plots: [] } })
    wrapper = mount(DeviceParameterCalculator, options)
    await wrapper.get('[aria-label="Parameter calculation"]').setValue('gate-slew-estimate')
    await wrapper.get('.estimate-choice input').setValue(true)
    await calculate()
    const result = wrapper.get('[aria-label="Device parameter results"]')
    expect(result.text()).toContain('Signed collector current slew (A/s)')
    expect(result.text()).toContain('9000000000')
    expect(result.text()).not.toContain('GA/s')
  })
  it('downloads exact server CSV and keeps chart dataset identity during unrelated renders', async () => {
    const csv = 'gate_voltage_v,current_a,fitted_current_a,residual_a,secant_gm_s,source\r\n4,2.1,2.100000001,1e-9,2.1,source\r\n'
    post.mockResolvedValue({ data: { ...transfer, csv } })
    wrapper = mount(DeviceParameterCalculator, options)
    await wrapper.get('[aria-label="Parameter calculation"]').setValue('transfer-fit')
    await calculate()
    const chart = wrapper.findComponent({ name: 'CurvePlotter' })
    const datasets = chart.props('datasets')
    wrapper.vm.$forceUpdate(); await flushPromises()
    expect(chart.props('datasets')).toBe(datasets)
    let blob
    vi.spyOn(URL, 'createObjectURL').mockImplementation(value => { blob = value; return 'blob:fixture' })
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    await wrapper.findAll('button').find(button => button.text().includes('CSV')).trigger('click')
    expect(await blob.text()).toBe(csv)
  })
  it('uses compact scientific y ticks only for the transfer residual plot', async () => {
    post.mockResolvedValue({ data: { ...transfer, plots: [...transfer.plots,
      { title: 'Transfer residual', x_label: 'Gate voltage (V)', y_label: 'Residual (A)', x: [4, 5], series: { Residual: [1e-15, -1e-15] } }] } })
    wrapper = mount(DeviceParameterCalculator, options)
    await wrapper.get('[aria-label="Parameter calculation"]').setValue('transfer-fit')
    await calculate()
    const charts = wrapper.findAllComponents({ name: 'CurvePlotter' })
    expect(charts.map(chart => chart.props('compactScientificYTicks'))).toEqual([false, true])
    expect(charts[1].props('datasets')[0].data).toEqual([{ x: 4, y: 1e-15 }, { x: 5, y: -1e-15 }])
  })
  it('keeps malformed advanced curve shapes editable for typed API rejection', async () => {
    const renderErrors = []
    post.mockRejectedValue({ response: { data: { detail: [{ loc: ['body', 'curves'], msg: 'Invalid curve shape' }] } } })
    wrapper = mount(DeviceParameterCalculator, { global: { ...options.global, config: { errorHandler: error => renderErrors.push(error) } } })
    await wrapper.get('[aria-label="Parameter calculation"]').setValue('transfer-fit')
    for (const curves of [[{}], null, ['wrong'], [{ voltage_v: null, current_a: 'wrong' }]]) {
      await wrapper.get('[aria-label="Parameter input data"]').setValue(JSON.stringify({ ...transferInput, curves }))
      await calculate()
      expect(wrapper.get('[role=alert]').text()).toContain('Invalid curve shape')
    }
    expect(renderErrors).toEqual([])
  })
  it('discards an extraction response after a curve edit', async () => {
    let resolve
    post.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    wrapper = mount(DeviceParameterCalculator, options)
    await wrapper.get('[aria-label="Parameter calculation"]').setValue('transfer-fit')
    await calculate()
    await wrapper.get('[aria-label="Curve 1 gate voltage (V)"]').setValue('4.1')
    resolve({ data: transfer }); await flushPromises()
    expect(wrapper.find('[aria-label="Device parameter results"]').exists()).toBe(false)
  })
  it('reviews application, preserves independent conditions and marks subsequent parameter edits', async () => {
    wrapper = mount(BielaCalculator, { ...options, props: { extraction: transfer } })
    expect(wrapper.get('[name=k_1]').element.value).toBe('10')
    expect(wrapper.get('[aria-label="Review extraction application"]').text()).toContain('real transfer source')
    await wrapper.findAll('button').find(button => button.text() === 'Apply these extracted parameters').trigger('click')
    expect(wrapper.get('[name=k_1]').element.value).toBe('2')
    expect(wrapper.get('[name=v_dc]').element.value).toBe('100')
    expect(wrapper.get('[name=q_oss]').element.value).toBe('2e-7')
    expect(wrapper.get('[name=source]').element.value).toContain('algebra')
    await wrapper.get('[name=k_1]').setValue('3')
    expect(wrapper.get('[aria-label="Applied extraction evidence"]').text()).toContain('Edited manual parameters')
    await wrapper.findAll('input[type=checkbox]')[1].setValue(true)
    calculateBiela.mockRejectedValueOnce(new Error('test boundary'))
    await wrapper.findAll('button').find(button => button.text() === 'Calculate half-bridge estimate').trigger('click')
    await flushPromises()
    expect(calculateBiela.mock.lastCall[0].extraction_evidence).toEqual([{ inputs: transferInput, status: 'edited' }])
  })
  it('rejects recovery Qoss context conflicts without inventing compatible capacitances', async () => {
    wrapper = mount(BielaCalculator, { ...options, props: { extraction: recovery } })
    await wrapper.findAll('button').find(button => button.text() === 'Apply these extracted parameters').trigger('click')
    expect(wrapper.get('[name=recovery_mode]').element.value).toBe('charge_control')
    expect(wrapper.get('[name=tau_rr]').element.value).toBe(String(rrParameters.tau_rr))
    expect(wrapper.get('[name=c_ds]').element.value).toBe('1e-9')
    expect(wrapper.get('[name=q_oss]').element.value).toBe('2e-7')
    await wrapper.findAll('input[type=checkbox]')[1].setValue(true)
    await wrapper.findAll('button').find(button => button.text() === 'Calculate half-bridge estimate').trigger('click')
    await flushPromises()
    expect(calculateBiela).not.toHaveBeenCalled()
    expect(wrapper.get('[role=alert]').text()).toContain('Qoss or voltage differs')
  })
  it('retains model edits across the parameter handoff and stages before application', async () => {
    wrapper = mount(AnalyticalCalculator, options)
    await wrapper.get('[name=r_g]').setValue('22')
    await wrapper.findAll('button').find(button => button.text() === 'Device Parameter Operations').trigger('click')
    const device = wrapper.findComponent(DeviceParameterCalculator)
    device.vm.$emit('apply-biela', transfer); await flushPromises()
    expect(wrapper.get('[name=r_g]').element.value).toBe('22')
    expect(wrapper.get('[name=k_1]').element.value).toBe('10')
    expect(wrapper.find('[aria-label="Review extraction application"]').exists()).toBe(true)
    await wrapper.findAll('button').find(button => button.text() === 'Device Parameter Operations').trigger('click')
    await device.get('[name=source]').setValue('A new source invalidates pending application')
    await wrapper.findAll('button').find(button => button.text() === 'Biela Switching Loss Model').trigger('click')
    expect(wrapper.find('[aria-label="Review extraction application"]').exists()).toBe(false)
  })
})
