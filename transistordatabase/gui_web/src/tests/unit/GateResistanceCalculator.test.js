import { afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

const { post } = vi.hoisted(() => ({ post: vi.fn() }))
vi.mock('../../services/api', () => ({ default: { post } }))
import DeviceParameterCalculator from '../../components/DeviceParameterCalculator.vue'

describe('inverse gate resistance estimate', () => {
  let wrapper
  afterEach(() => { wrapper?.unmount(); post.mockReset() })

  it('edits SI fields, acknowledges the estimate and displays sourced ohm results', async () => {
    wrapper = mount(DeviceParameterCalculator, { global: { stubs: { CurvePlotter: true } } })
    await wrapper.get('[aria-label="Parameter calculation"]').setValue('gate-resistance-estimate')
    expect(wrapper.get('button').attributes('disabled')).toBeDefined()
    for (const [name, value] of Object.entries({ target_dv_dt_v_s: '20000000000', c_rss_min_f: '1e-10', v_drive_high_v: '15', v_drive_low_v: '-5', v_plateau_low_v: '5', v_plateau_high_v: '7' })) {
      await wrapper.get(`[name="${name}"]`).setValue(value)
    }
    await wrapper.get('[name=source]').setValue('Measured capacitance and plateau source')
    await wrapper.get('.estimate-choice [type=checkbox]').setValue(true)
    post.mockResolvedValueOnce({ data: { operation: 'gate-resistance-estimate', units: 'SI', condition_policy: 'Exact recorded conditions', plots: [],
      context: { source: 'Measured capacitance and plateau source' },
      result: { r_on_ohm: 5, r_off_ohm: 6, r_on_min_ohm: 4, r_on_max_ohm: 5, r_off_min_ohm: 5, r_off_max_ohm: 6, gate_current_magnitude_a: 2, unit: 'ohm',
        scope: 'Total gate-loop series resistance, including driver, internal and external resistance',
        assumptions: ['Constant Miller capacitance and supplied plateau bounds'] } } })
    await wrapper.get('button').trigger('click')
    await flushPromises()
    expect(post).toHaveBeenCalledWith('/api/parameters/calculate', expect.objectContaining({ operation: 'gate-resistance-estimate', selected: true, target_dv_dt_v_s: 20e9, c_rss_min_f: 100e-12,
      v_drive_high_v: 15, v_drive_low_v: -5, v_plateau_low_v: 5, v_plateau_high_v: 7, source: 'Measured capacitance and plateau source' }))
    const result = wrapper.get('[aria-label="Device parameter results"]')
    expect(result.text()).toContain('Minimum turn-on total series resistance (Ω)')
    expect(result.text()).toContain('Minimum turn-off total series resistance (Ω)')
    expect(result.text()).toContain('Measured capacitance and plateau source')
    expect(result.text()).toContain('Constant Miller capacitance and supplied plateau bounds')
    expect(result.text()).toContain('5')
    expect(result.text()).toContain('6')
    expect(wrapper.emitted('apply-biela')).toBeUndefined()
    await wrapper.get('[name=v_plateau_low_v]').setValue('6')
    expect(wrapper.find('[aria-label="Device parameter results"]').exists()).toBe(false)
  })
})
