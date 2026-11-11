import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
const { inspectImage, transform } = vi.hoisted(() => ({ inspectImage: vi.fn(), transform: vi.fn() }))
vi.mock('../../services/api.js', () => ({ digitizerApi: { image: inspectImage, transform } }))
import DigitizerWorkbench from '../../components/DigitizerWorkbench.vue'

const metadata = { sha256: 'ab'.repeat(32), width: 1000, height: 600, format: 'PNG', byte_size: 4, orientation: 1 }
const response = {
  image: metadata, calibration: { image: metadata }, source: 'Fixture figure, conditions unknown',
  points: [{ index: 1, pixel_x: 500, pixel_y: 300, x: 10, y: 10 }],
  csv: 'index,pixel_x,pixel_y,x,y\r\n1,500,300,10,10\r\n', units: { x: 'A', y: 'V' },
}

describe('calibrated image digitizer', () => {
  let wrapper
  const field = id => wrapper.get(`[data-testid=digitizer-${id}]`)
  async function upload(bytes = [137, 80, 78, 71], name = 'fixture.png') {
    const file = { name, size: bytes.length, arrayBuffer: async () => Uint8Array.from(bytes).buffer }
    Object.defineProperty(field('file').element, 'files', { configurable: true, value: [file] })
    await field('file').trigger('change'); await flushPromises()
    await vi.waitFor(() => {
      expect(wrapper.find('[data-testid=digitizer-image]').exists()).toBe(true)
      expect(field('map').element.disabled).toBe(false)
      expect(wrapper.text()).not.toContain('Checking image bytes and dimensions…')
    })
  }
  async function calibrate() {
    for (const [key, value] of Object.entries({ 'x-pixel-1': 100, 'x-pixel-2': 900, 'x-value-1': 1, 'x-value-2': 100, 'y-pixel-1': 500, 'y-pixel-2': 100, 'y-value-1': 0, 'y-value-2': 20, 'x-unit': 'A', 'y-unit': 'V', source: response.source })) await field(key).setValue(String(value))
    await field('x-scale').setValue('log')
  }
  async function add(x = 500, y = 300) {
    await field('manual-x').setValue(x); await field('manual-y').setValue(y)
    await wrapper.get('.point-entry').trigger('submit')
  }
  async function map() {
    const previousCalls = transform.mock.calls.length
    await vi.waitFor(() => expect(field('map').element.disabled).toBe(false))
    await field('map').trigger('click')
    await vi.waitFor(() => expect(transform).toHaveBeenCalledTimes(previousCalls + 1))
    await flushPromises()
  }
  beforeEach(() => {
    inspectImage.mockReset().mockResolvedValue(metadata)
    transform.mockReset().mockResolvedValue(structuredClone(response))
    wrapper = mount(DigitizerWorkbench)
  })
  afterEach(() => { wrapper.unmount(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

  it('uses precise pointer coordinates only for a completed primary click and clears canceled pointers', async () => {
    await upload()
    vi.stubGlobal('DOMPoint', class {
      constructor(x, y) { this.x = x; this.y = y }
      matrixTransform(matrix) { return matrix.transformPoint(this) }
    })
    const svg = field('image')
    svg.element.getScreenCTM = () => ({ inverse: () => ({ transformPoint: point => ({ x: point.x * 2, y: point.y * 2 }) }) })
    async function dispatch(type, properties) {
      const event = new Event(type, { bubbles: true })
      for (const [key, value] of Object.entries(properties)) Object.defineProperty(event, key, { value })
      svg.element.dispatchEvent(event); await flushPromises()
    }
    const precise = { isPrimary: true, button: 0, clientX: 250.25, clientY: 150.25 }
    await dispatch('pointerdown', precise); await dispatch('pointerup', precise)
    expect(wrapper.findAll('[data-testid=digitizer-point-row]')).toHaveLength(0)
    await dispatch('click', { button: 0, clientX: 250, clientY: 150 })
    expect(wrapper.get('[aria-label="Point 1 X pixel"]').element.value).toBe('500.5')
    expect(wrapper.get('[aria-label="Point 1 Y pixel"]').element.value).toBe('300.5')
    await dispatch('pointerdown', precise); await dispatch('pointercancel', {})
    await dispatch('click', { button: 0, clientX: 200, clientY: 100 })
    expect(wrapper.findAll('[data-testid=digitizer-point-row]')).toHaveLength(1)
    const next = { ...precise, clientX: 200, clientY: 100 }
    await dispatch('pointerdown', next); await dispatch('pointerup', next)
    await dispatch('click', next)
    expect(wrapper.get('[aria-label="Point 2 X pixel"]').element.value).toBe('400')
    await dispatch('pointerdown', precise)
    await dispatch('pointermove', { ...precise, clientX: 350 })
    await dispatch('pointermove', precise); await dispatch('pointerup', precise)
    await dispatch('click', precise)
    expect(wrapper.findAll('[data-testid=digitizer-point-row]')).toHaveLength(2)
    await dispatch('pointerdown', { ...precise, button: 2 }); await dispatch('pointerup', { ...precise, button: 2 })
    await dispatch('click', { button: 2, clientX: 250, clientY: 150 })
    expect(wrapper.findAll('[data-testid=digitizer-point-row]')).toHaveLength(2)
  })

  it('retains exact bytes and submits independent axis calibration to Python', async () => {
    await upload(); await calibrate(); await add(); await map()
    expect(inspectImage).toHaveBeenCalledWith({ filename: 'fixture.png', content_base64: 'iVBORw==' })
    expect(transform).toHaveBeenCalledWith({ filename: 'fixture.png', content_base64: 'iVBORw==', source: response.source,
      calibration: { image: { sha256: metadata.sha256, width: 1000, height: 600 },
        x: { pixel_1: 100, value_1: 1, pixel_2: 900, value_2: 100, scale: 'log', unit: 'A' },
        y: { pixel_1: 500, value_1: 0, pixel_2: 100, value_2: 20, scale: 'linear', unit: 'V' }, allow_extrapolation: false }, points: [{ x: 500, y: 300 }] })
    expect(field('physical-x').text()).toBe('10'); expect(field('physical-y').text()).toBe('10')
    expect(field('csv').attributes('disabled')).toBeUndefined()
    await field('x-value-2').setValue('1000')
    expect(field('y-value-2').element.value).toBe('20')
    expect(field('csv').attributes('disabled')).toBeDefined()
    expect(field('physical-y').text()).toBe('Map required')
  })

  it('preserves selection order and supports point edit, removal and undo', async () => {
    await upload(); await add(500, 300); await add(600, 350)
    await wrapper.get('[aria-label="Point 1 X pixel"]').setValue('450')
    await field('undo').trigger('click')
    expect(wrapper.get('[aria-label="Point 1 X pixel"]').element.value).toBe('500')
    await wrapper.get('[aria-label="Remove point 1"]').trigger('click')
    expect(wrapper.get('[aria-label="Point 1 X pixel"]').element.value).toBe('600')
    await field('undo').trigger('click'); await map()
    expect(transform.mock.calls.at(-1)[0].points).toEqual([{ x: 500, y: 300 }, { x: 600, y: 350 }])
  })

  it('labels a blank source as unknown without preventing a valid calibrated export', async () => {
    await upload(); await calibrate(); await add(); await field('source').setValue(''); await map()
    expect(transform.mock.calls.at(-1)[0].source).toBe('Unknown')
    expect(field('csv').attributes('disabled')).toBeUndefined()
    await field('source').setValue(' \t@source\n'); await map()
    expect(transform.mock.calls.at(-1)[0].source).toBe(' \t@source\n')
  })

  it('ignores held transform responses after calibration changes and image replacement', async () => {
    await upload(); await calibrate(); await add()
    let release; transform.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await field('map').trigger('click'); await field('source').setValue('New source')
    release(response); await flushPromises()
    expect(wrapper.find('[data-testid=digitizer-results]').exists()).toBe(false)
    transform.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    await field('map').trigger('click')
    inspectImage.mockResolvedValueOnce({ ...metadata, sha256: 'cd'.repeat(32) })
    await upload([1, 2, 3, 4], 'fixture.png')
    release(response); await flushPromises()
    expect(wrapper.findAll('[data-testid=digitizer-point-row]')).toHaveLength(0)
    expect(field('x-pixel-1').element.value).toBe('')
    expect(field('hash').text()).toBe('cd'.repeat(32))
    expect(field('csv').attributes('disabled')).toBeDefined()
  })

  it('retains calibration only for an identical image hash and rejects stale image reads', async () => {
    await upload(); await calibrate(); await add(); await map()
    await upload(undefined, 'renamed.png')
    expect(field('x-pixel-1').element.value).toBe('100')
    expect(wrapper.findAll('[data-testid=digitizer-point-row]')).toHaveLength(1)
    expect(wrapper.text()).toContain('image bytes are identical')
    expect(field('csv').attributes('disabled')).toBeDefined()
    let release
    const slow = { name: 'slow.png', size: 2, arrayBuffer: () => new Promise(resolve => { release = resolve }) }
    Object.defineProperty(field('file').element, 'files', { configurable: true, value: [slow] })
    await field('file').trigger('change')
    inspectImage.mockResolvedValueOnce({ ...metadata, sha256: 'ef'.repeat(32) })
    await upload([5, 6], 'new.png')
    release(Uint8Array.from([1, 2]).buffer); await flushPromises()
    expect(field('hash').text()).toBe('ef'.repeat(32))
    expect(wrapper.text()).toContain('new.png')
    expect(inspectImage).toHaveBeenCalledTimes(3)
  })

  it('exports the exact Python CSV and JSON values and displays backend validation errors', async () => {
    await upload(); await calibrate(); await add(); await map()
    const blobs = [], names = []
    vi.spyOn(URL, 'createObjectURL').mockImplementation(blob => { blobs.push(blob); return 'blob:digitizer' })
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function () { names.push(this.download) })
    await field('csv').trigger('click'); await field('json').trigger('click')
    expect(await blobs[0].text()).toBe(response.csv)
    expect(JSON.parse(await blobs[1].text())).toEqual(response)
    expect(names).toEqual(['digitized-points.csv', 'digitized-points.json'])
    await field('extrapolate').setValue(true)
    transform.mockRejectedValueOnce({ response: { data: { detail: 'Logarithmic anchors must be positive.' } } })
    await map()
    expect(transform.mock.calls.at(-1)[0].calibration.allow_extrapolation).toBe(true)
    expect(wrapper.get('[role=alert]').text()).toContain('Logarithmic anchors must be positive')
    expect(field('csv').attributes('disabled')).toBeDefined()
  })
})
