import { afterEach, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import AnalyticalCalculator from '../../components/AnalyticalCalculator.vue'

afterEach(() => window.history.replaceState({}, '', '/'))
it.each(['biela', 'gate_charge', 'parameters', 'device-fitting', 'invalid-model'])('validates the initial analytical model: %s', model => {
  window.history.replaceState({}, '', `/?model=${model}`)
  const wrapper = mount(AnalyticalCalculator, { global: { stubs: { BielaCalculator: true, GateChargeCalculator: true, DeviceParameterCalculator: true, DeviceModelFitting: true } } })
  const active = wrapper.get('button.active').text()
  expect(active).toBe(model === 'device-fitting' ? 'Device model fitting' : model === 'parameters' ? 'Device Parameter Operations' : model === 'gate_charge' ? 'Gate Charge Timing Model' : 'Biela Switching Loss Model')
  wrapper.unmount()
})
