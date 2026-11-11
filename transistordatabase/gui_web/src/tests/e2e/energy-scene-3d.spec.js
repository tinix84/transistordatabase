import { expect, test } from '@playwright/test'
import { xyzText } from '../fixtures/xyzEnergy.js'
import { createHash } from 'node:crypto'

const field = (page, name) => page.getByLabel(`XYZ ${name}`, { exact: true })
async function ready(page) {
  await page.goto('/index.html?view=energy-map')
  await page.getByLabel('Energy map input', { exact: true }).selectOption('xyz')
  await field(page,'source text').fill(xyzText)
  await field(page,'source identifier').fill('Independent 3D source fixture')
  await field(page,'energy unit').selectOption('µJ')
  await field(page,'energy convention').fill('Recorded switching energy for 3D display test')
  await field(page,'common conditions').check()
}
async function parse(page,text) {
  if(text !== undefined) await field(page,'source text').fill(text)
  const waiting=page.waitForResponse(response => response.url().endsWith('/api/analysis/energy-map/xyz') && response.request().method()==='POST')
  await page.getByTestId('xyz-parse').click()
  const response=await waiting; expect(response.status()).toBe(200)
  await expect(page.getByTestId('energy-3d')).toBeVisible()
  return response.json()
}
async function raster(canvas) {
  return canvas.evaluate(node => {
    const ctx=node.getContext('2d'), pixels=ctx.getImageData(0,0,node.width,node.height).data
    let colored=0, digest=0
    for(let index=0;index<pixels.length;index+=4) {
      if(Math.max(pixels[index],pixels[index+1],pixels[index+2])-Math.min(pixels[index],pixels[index+1],pixels[index+2])>45) colored++
      digest=(digest+pixels[index]*(index+1))%2147483647
    }
    return {colored,digest,width:node.width,height:node.height}
  })
}
for(const width of [1440,390]) test(`compiled original-point 3D geometry, picking, rotation and surface at ${width}px`,async({page},info) => {
  const errors=[]; page.on('pageerror',error => errors.push(error.message))
  await page.setViewportSize({width,height:900}); await ready(page)
  const data=await parse(page), scene=page.getByTestId('energy-3d'), canvas=scene.locator('canvas')
  await expect(scene.getByTestId('energy-3d-counts')).toContainText('5 source rows · 5 finite points · 5 rendered points · 0 eligible triangles')
  const initial=await raster(canvas); expect(initial.colored).toBeGreaterThan(100)
  await canvas.scrollIntoViewIfNeeded(); const box=await canvas.boundingBox()
  // Independent orthographic projection of the largest source point, normalized (1,1,1).
  const yaw=-.65,pitch=.48,scale=Math.min(box.width-2,430)*.23
  const x=(box.width-2)/2+(Math.cos(yaw)-Math.sin(yaw))*scale
  const y=215-(Math.cos(pitch)-Math.sin(pitch)*(Math.sin(yaw)+Math.cos(yaw)))*scale
  await canvas.click({position:{x:x+1,y:y+1}})
  await expect(scene.getByTestId('energy-3d-selection')).toContainText('Selected line-1: I 9 A, V 600 V')
  await expect(scene.getByTestId('energy-3d-evidence')).toContainText(data.source.sha256)
  await expect(scene.getByTestId('energy-3d-evidence')).toContainText('Recorded switching energy for 3D display test')
  await scene.locator('[data-source-row="line-5"]').click()
  await expect(scene.getByTestId('energy-3d-selection')).toContainText('E 0 J')
  await canvas.focus(); await canvas.press('ArrowRight'); await canvas.press('+')
  await expect(canvas).toHaveAttribute('data-yaw','-0.5'); await expect(canvas).toHaveAttribute('data-zoom','1.15')
  expect((await raster(canvas)).digest).not.toBe(initial.digest)
  const rotatedX=(box.width-2)/2+(Math.cos(-.5)-Math.sin(-.5))*scale*1.15
  const rotatedY=215-(Math.cos(pitch)-Math.sin(pitch)*(Math.sin(-.5)+Math.cos(-.5)))*scale*1.15
  await canvas.click({position:{x:rotatedX+1,y:rotatedY+1}})
  await expect(scene.getByTestId('energy-3d-selection')).toContainText('Selected line-1: I 9 A, V 600 V')
  await canvas.press('Home'); await expect(canvas).toHaveAttribute('data-yaw','-0.65'); await expect(canvas).toHaveAttribute('data-zoom','1')
  await parse(page,`${xyzText}\n3,600,45`)
  await page.getByLabel('3D presentation',{exact:true}).selectOption('surface')
  await expect(scene.getByTestId('energy-3d-counts')).toContainText('6 rendered points · 4 eligible triangles · 4 drawn triangles')
  expect((await raster(canvas)).colored).toBeGreaterThan(initial.colored)
  await scene.screenshot({path:info.outputPath(`energy-3d-complete-${width}.png`)})
  await parse(page,`${xyzText}\n3,600,45\n3,600,46`)
  await expect(scene.getByTestId('energy-3d-counts')).toContainText('7 source rows · 7 finite points · 7 rendered points · 0 eligible triangles')
  await scene.locator('[data-source-row="line-6"]').click(); await expect(scene.getByTestId('energy-3d-evidence')).toContainText('0.000045')
  await scene.locator('[data-source-row="line-7"]').click(); await expect(scene.getByTestId('energy-3d-evidence')).toContainText('0.000046')
  await scene.screenshot({path:info.outputPath(`energy-3d-conflict-${width}.png`)})
  await field(page,'missing energy tokens').fill('NA'); await parse(page,`${xyzText}\n3,600,NA`)
  await expect(scene.getByTestId('energy-3d-counts')).toContainText('6 source rows · 5 finite points · 5 rendered points · 0 eligible triangles · 0 drawn triangles · 1 unknown energy rows')
  await scene.locator('[data-source-row="line-6"]').click(); await expect(scene.getByTestId('energy-3d-selection')).toContainText('E NULL J')
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width+1)
  await field(page,'source identifier').fill('changed'); await expect(scene).toHaveCount(0)
  expect(errors).toEqual([])
})

test('compiled prepared view preserves original source knots absent displayed grid and excludes mismatched slices',async({page},info) => {
  await page.goto('/index.html?view=energy-map')
  await page.getByLabel('Map currents',{exact:true}).fill('0,9')
  await page.getByLabel('Map voltages',{exact:true}).fill('200,600')
  // The accepted response retains originals independently of the displayed estimate grid.
  const details=page.getByText('Advanced canonical map request',{exact:true})
  await details.click()
  await page.getByLabel('Use map JSON as input',{exact:true}).check()
  const conditions={t_j_c:25,v_g_on_v:null,v_g_off_v:null,r_g_on_ohm:null,r_g_off_ohm:null,commutation_device:null,commutation_source:null,energy_convention:'Original prepared fixture'}
  const slice=(id,v,i,e,kind='on') => ({id,source:`Source ${id}`,source_kind:'manual',energy_kind:kind,voltage_v:v,current_a:i,energy_j:e,conditions})
  const request={energy_kind:'on',conditions,currents_a:[0,9],voltages_v:[200,600],slices:[slice('a',200,[0,3,9],[0,.00003,.00006]),slice('b',600,[0,9],[.000012,.00009]),slice('excluded',800,[0,9],[1,2],'off')]}
  await page.getByLabel('Map request JSON',{exact:true}).fill(JSON.stringify(request))
  const waiting=page.waitForResponse(response => response.url().endsWith('/api/analysis/energy-map') && response.request().method()==='POST')
  await page.getByRole('button',{name:'Build energy map',exact:true}).click(); expect((await waiting).status()).toBe(200)
  const scene=page.getByTestId('energy-3d')
  await expect(scene.getByTestId('energy-3d-counts')).toContainText('5 source rows · 5 finite points · 5 rendered points · 0 eligible triangles')
  await expect(scene.locator('[data-source-row="a:point-2"]')).toContainText('a:point-2')
  await scene.locator('[data-source-row="a:point-2"]').click(); await expect(scene.getByTestId('energy-3d-selection')).toContainText('I 3 A')
  await expect(scene).not.toContainText('Source excluded')
  await scene.screenshot({path:info.outputPath('energy-3d-prepared-originals.png')})
})

test('compiled over-limit source keeps bounded tables, coincident identities, matrix pages and exact source hash',async({page},info) => {
  await ready(page)
  const records=Array.from({length:17*17},(_,index) => `${index%17},${Math.floor(index/17)},0`)
  while(records.length<12001) records.push('0,0,0')
  const text=records.join('\n')
  await field(page,'source input').selectOption('file')
  await field(page,'source file').setInputFiles({name:'bounded-source.csv',mimeType:'text/csv',buffer:Buffer.from(text,'utf8')})
  const data=await parse(page), scene=page.getByTestId('energy-3d')
  expect(data.source.sha256).toBe(createHash('sha256').update(text,'utf8').digest('hex'))
  await expect(scene.getByTestId('energy-3d-counts')).toContainText('12001 source rows · 12001 finite points · 0 rendered points · 0 eligible triangles')
  await expect(scene).toContainText('point render limit is exceeded')
  await expect(scene.locator('tbody tr')).toHaveCount(50)
  await scene.locator('[data-source-row="line-1"]').click()
  const coincident=scene.getByTestId('energy-3d-coincident')
  await expect(coincident).toContainText('11713 original rows')
  await expect(coincident.locator('button')).toHaveCount(50)
  await expect(coincident).toContainText('Use the paged source table')
  const matrix=page.getByTestId('xyz-matrix')
  await expect(matrix.locator('td')).toHaveCount(256)
  await page.getByRole('button',{name:'Next current columns',exact:true}).click()
  await expect(matrix.locator('td')).toHaveCount(16)
  await page.getByRole('button',{name:'Next voltage rows',exact:true}).click()
  await expect(matrix.locator('td')).toHaveCount(1)
  await expect(matrix.locator('td')).toHaveAttribute('data-current','16')
  await expect(matrix.locator('td')).toHaveAttribute('data-voltage','0')
  await page.getByRole('button',{name:'Next 3D rows',exact:true}).click()
  await expect(scene.locator('[data-source-row="line-51"]')).toBeVisible()
  await scene.screenshot({path:info.outputPath('energy-3d-explicit-cap-and-identities.png')})
})
