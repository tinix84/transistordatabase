import { afterEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import EnergyScene3D from '../../components/EnergyScene3D.vue'
import { buildScene, defaultCamera, FACE_GRID_LIMIT, moveCamera, normalize, pickPoint, POINT_LIMIT, projectScene, sourceObservations } from '../../components/energyScene.js'
import { xyzResult } from '../fixtures/xyzEnergy.js'

const xyz = () => sourceObservations(xyzResult(), 'xyz')
const complete = () => [...xyz(), { ...xyz()[0], id: 'line-6', current_a: 3, voltage_v: 600, energy_j: .000045 }]
const slice = (id, v, currents, energies) => ({ id, voltage_v: v, current_a: currents, energy_j: energies,
  source: `source-${id}`, source_kind: 'manual', conditions: { t_j_c: 25 } })
const prepared = slices => ({ source_slices: slices, slice_diagnostics: slices.map(value => ({ slice_id: value.id, status: 'selected' })),
  cells: [[{ current_a: 5, voltage_v: 400, energy_j: 9, status: 'available', mode: 'bounded-current-linear' }]], input_sha256: 'c'.repeat(64) })
let wrapper
afterEach(() => { wrapper?.unmount(); vi.restoreAllMocks() })

describe('original energy source geometry', () => {
  it('preserves shuffled SI source rows and never fills the missing internal coordinate', () => {
    const scene = buildScene(xyz())
    expect(scene.points.map(row => row.id)).toEqual(['line-1','line-2','line-3','line-4','line-5'])
    expect(scene.currents).toEqual([0,3,9]); expect(scene.voltages).toEqual([200,600])
    expect(scene.points.find(row => row.id === 'line-5').energy_j).toBe(0)
    expect(scene.faces).toHaveLength(0)
  })
  it('makes exactly two triangles per complete adjacent quad and no triangles through a conflict', () => {
    expect(buildScene(complete()).faces).toHaveLength(4)
    const conflict = [...complete(), { ...complete()[5], id: 'line-7', energy_j: .000046 }]
    expect(buildScene(conflict).points).toHaveLength(7); expect(buildScene(conflict).faces).toHaveLength(0)
    conflict[6].energy_j = .000045
    expect(buildScene(conflict).faces).toHaveLength(0)
  })
  it('keeps unknown rows without a fake zero point or a face', () => {
    const rows = complete(); rows[5] = { ...rows[5], energy_j: null, available: false, cell_status: 'unknown' }
    const scene = buildScene(rows)
    expect(scene.observations).toHaveLength(6); expect(scene.points).toHaveLength(5); expect(scene.faces).toHaveLength(0)
  })
  it('uses only selected original slice samples and their full knot axes, never interpolated cells', () => {
    const data = prepared([slice('a',200,[0,3,9],[0,.00003,.00006]),slice('b',600,[0,9],[.000012,.00009]),slice('excluded',800,[0,9],[1,2])])
    data.slice_diagnostics[2].status = 'excluded'
    const rows = sourceObservations(data,'prepared'), scene = buildScene(rows)
    expect(rows).toHaveLength(5); expect(scene.currents).toEqual([0,3,9]); expect(scene.voltages).toEqual([200,600])
    expect(scene.faces).toHaveLength(0); expect(rows.some(row => row.energy_j === 9)).toBe(false)
    expect(rows[1]).toMatchObject({ id:'a:point-2', source:'source-a', conditions:{t_j_c:25}, current_a:3 })
  })
  it('suppresses prepared corners with competing slice domains at different original knots', () => {
    const rows = sourceObservations(prepared([slice('a',200,[0,3,9],[0,3,9]),slice('b',600,[0,3,9],[0,3,9]),slice('competing',600,[1,8],[1,8])]),'prepared')
    expect(rows.find(row => row.id === 'b:point-2').cell_status).toBe('ambiguous')
    expect(buildScene(rows).faces).toHaveLength(0)
  })
  it('never connects disjoint prepared source domains across an unsupported current interval', () => {
    const rows = sourceObservations(prepared([slice('left',200,[0,1],[0,1]),slice('right',200,[9,10],[9,10]),slice('top',600,[0,1,9,10],[0,1,9,10])]),'prepared')
    const scene = buildScene(rows)
    expect(scene.faces).toHaveLength(4)
    expect(scene.faces.some(face => face.some(point => point.current_a === 1) && face.some(point => point.current_a === 9))).toBe(false)
  })
  it('handles empty selections, constants, tiny energies and maximum finite coordinates', () => {
    expect(buildScene([]).points).toHaveLength(0)
    expect(normalize(5,[5])).toBe(0)
    expect(normalize(5e-324,[0,1e-323])).toBe(0)
    expect(normalize(Number.MAX_VALUE,[0,Number.MAX_VALUE])).toBe(1)
    expect(normalize(0,[-Number.MAX_VALUE,Number.MAX_VALUE])).toBe(0)
    expect(normalize(1.5000000000000002,[1.5,1.5000000000000002,1.5000000000000004])).toBe(0)
    const row = { id:'extreme',current_a:Number.MAX_VALUE,voltage_v:0,energy_j:5e-324,available:true }
    const point = projectScene(buildScene([row]),defaultCamera(),390,430).points[0]
    expect(point.x).toBe(195); expect(point.y).toBe(215); expect(point.depth).toBe(0)
  })
  it('normalizes nonuniform coordinates numerically instead of by their indices', () => {
    expect(normalize(3,[0,3,9])).toBeCloseTo(-1/3)
    const camera = {yaw:0,pitch:0,zoom:1}, points = projectScene(buildScene(complete()),camera,1000,500).points
    const low = points.filter(row => row.voltage_v === 200).sort((a,b) => a.current_a-b.current_a)
    expect((low[1].x-low[0].x)/(low[2].x-low[1].x)).toBeCloseTo(.5)
  })
  it('rotates actual projected coordinates, depth-sorts faces and resets camera and zoom', () => {
    const scene = buildScene(complete()), camera = defaultCamera(), initial = projectScene(scene,camera,720,430)
    const moved = moveCamera(moveCamera(camera,'ArrowRight'),'+'), rotated = projectScene(scene,moved,720,430)
    expect(rotated.points.find(point => point.id === 'line-1').x).not.toBe(initial.points.find(point => point.id === 'line-1').x)
    const depths = rotated.faces.map(face => face.reduce((sum,point) => sum+point.depth,0))
    expect(depths).toEqual([...depths].sort((a,b) => a-b))
    expect(moveCamera(moved,'r')).toEqual(camera)
    let zoom = camera; for(let i=0;i<30;i++) zoom=moveCamera(zoom,'+')
    expect(zoom.zoom).toBe(1.7)
  })
  it('picks nearest rendered points and frontmost exact overlaps without losing duplicate row identities', () => {
    const points = [{id:'back',x:10,y:20,depth:0},{id:'front',x:10,y:20,depth:1},{id:'near',x:15,y:20,depth:2}]
    expect(pickPoint(points,10,20).id).toBe('front'); expect(pickPoint(points,15,20).id).toBe('near')
    expect(pickPoint(points,100,100)).toBe(null)
  })
  it('declares render and union-grid limits instead of silently truncating observations', () => {
    const rows = Array.from({length:POINT_LIMIT+1},(_,index) => ({id:String(index),current_a:index,voltage_v:0,energy_j:0,available:true}))
    const scene = buildScene(rows)
    expect(scene.points).toHaveLength(POINT_LIMIT+1); expect(scene.capped).toBe(true); expect(scene.renderedPoints).toHaveLength(0)
    const sparse = Array.from({length:300},(_,index) => ({id:String(index),current_a:index,voltage_v:index,energy_j:0,available:true}))
    expect(300*300).toBeGreaterThan(FACE_GRID_LIMIT); expect(buildScene(sparse).faceCapped).toBe(true)
  })
})

describe('linked accessible source view', () => {
  it('connects source row selection to complete source, condition and hash evidence', async () => {
    wrapper=mount(EnergyScene3D,{props:{result:xyzResult(),mode:'xyz'}})
    await wrapper.get('[data-source-row="line-5"]').trigger('click')
    expect(wrapper.get('[data-testid=energy-3d-selection]').text()).toContain('E 0 J')
    expect(wrapper.get('[data-testid=energy-3d-evidence]').text()).toContain('Manual fixture')
    expect(wrapper.get('[data-testid=energy-3d-evidence]').text()).toContain('Recorded switching energy')
    expect(wrapper.get('[data-testid=energy-3d-evidence]').text()).toContain('a'.repeat(64))
    expect(wrapper.get('[data-source-row="line-5"]').attributes('aria-pressed')).toBe('true')
    const canvas=wrapper.get('canvas'), initial=Number(canvas.attributes('data-yaw'))
    await canvas.trigger('keydown',{key:'ArrowRight'}); expect(Number(canvas.attributes('data-yaw'))).toBeCloseTo(initial+.15)
    await canvas.trigger('keydown',{key:'+'}); expect(Number(canvas.attributes('data-zoom'))).toBeGreaterThan(1)
    await canvas.trigger('keydown',{key:'Home'}); expect(Number(canvas.attributes('data-yaw'))).toBe(initial)
  })
  it('states all finite and unknown counts and pages every source row', async () => {
    const result=xyzResult(); result.rows=Array.from({length:101},(_,index) => ({...result.rows[0],id:`line-${index+1}`,energy_j:index === 100?null:0}))
    wrapper=mount(EnergyScene3D,{props:{result,mode:'xyz'}})
    expect(wrapper.get('[data-testid=energy-3d-counts]').text()).toContain('101 source rows · 100 finite points')
    expect(wrapper.get('tbody').findAll('tr')).toHaveLength(50)
    const next=wrapper.findAll('button').find(button => button.text()==='Next 3D rows')
    await next.trigger('click'); await next.trigger('click')
    expect(wrapper.get('tbody').findAll('tr')).toHaveLength(1); expect(wrapper.get('tbody').text()).toContain('NULL')
  })
  it('lists distinct IDs for identical source points after selection', async () => {
    const result=xyzResult(); result.rows.push({...result.rows[0],id:'identical-copy',cell_status:'ambiguous'})
    wrapper=mount(EnergyScene3D,{props:{result,mode:'xyz'}})
    await wrapper.get('[data-source-row="line-1"]').trigger('click')
    const coincident=wrapper.get('[data-testid=energy-3d-coincident]')
    expect(coincident.text()).toContain('2 original rows')
    await coincident.findAll('button')[1].trigger('click')
    expect(wrapper.get('[data-testid=energy-3d-selection]').text()).toContain('Selected identical-copy')
  })
})
