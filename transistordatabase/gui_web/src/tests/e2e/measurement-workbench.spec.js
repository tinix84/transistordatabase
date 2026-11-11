import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
const repo = '/home/tinix/claude_wsl/transistordatabase', python = '/home/tinix/miniconda3/envs/ntbees2/bin/python'
const field = (page,id) => page.getByTestId('measurement-' + id)
function fixtures() {
  return JSON.parse(execFileSync(python,['-c', `
import sys,json,csv,base64,io,uuid
sys.path.insert(0,'tests')
from test_measurement_import import analytic
from transistordatabase.core.models import Transistor,TransistorMetadata,ElectricalRatings,ThermalProperties
from transistordatabase.core.serialization import core_to_json_document
from spicelib.raw.raw_write import RawWrite,Trace
from pathlib import Path
a=analytic(); out=io.StringIO(); writer=csv.writer(out,quoting=csv.QUOTE_ALL,delimiter=';')
writer.writerow(['time','V(ds)','I(d)','V(gs)'])
writer.writerows(zip(a[0]*1e6,a[1],-a[2],a[3]))
csvdata=out.getvalue().encode('utf-8-sig')
w=RawWrite(numtype='double')
for name,values,kind in zip(('time','V(ds)','I(d)','V(gs)'),a,('time','voltage','current','voltage')):
 w.add_trace(Trace(name,values,whattype=kind,numerical_type='double'))
p=Path('/tmp')/('measurement-browser-'+uuid.uuid4().hex+'.raw');w.save(p)
device=Transistor(TransistorMetadata('Independent fixture','MOSFET','browser reviewer','unknown','unknown'),ElectricalRatings(200.,20.,10.,150.),ThermalProperties(.01,.01))
print(json.dumps({'csv':base64.b64encode(csvdata).decode(),'raw':base64.b64encode(p.read_bytes()).decode(),'native':core_to_json_document(device)}))
`],{cwd:repo,encoding:'utf8'}).trim())
}
async function inspect(page) { const response=page.waitForResponse(r=>r.url().endsWith('/api/measurement/preview')); await field(page,'inspect').click(); expect((await response).status()).toBe(200) }
async function map(page, raw=false) {
  for(const [key,name,unit] of [['time','time',raw?'s':'us'],['voltage','V(ds)','V'],['current','I(d)','A'],['gate','V(gs)','V']]) {await field(page,'map-'+key).selectOption(name); await field(page,'unit-'+key).selectOption(unit)}
  if(!raw) await field(page,'polarity-current').selectOption('-1')
  if(raw) await field(page,'step-index').fill('0')
}
async function declare(page) { for(const [key,value] of Object.entries({bus_voltage_v:100,v_g_on:15,v_g_off:-5,r_g_on:2,r_g_off:7})) { await field(page,'known-'+key).selectOption('true');await field(page,'condition-'+key).fill(String(value))} }
async function preview(page) {const waiting=page.waitForResponse(r=>r.url().endsWith('/api/measurement/preview'));await field(page,'preview').click();const response=await waiting;expect(response.status()).toBe(200);return response.json()}
async function analyze(page) {const waiting=page.waitForResponse(r=>r.url().endsWith('/api/measurement/extract'));await field(page,'analyze').click();const response=await waiting;expect(response.status()).toBe(200);return response.json()}
async function download(page,id) {const waiting=page.waitForEvent('download');await field(page,id).click();return readFile(await (await waiting).path(),'utf8')}
async function upload(page, bytes, name) {await field(page,'file').setInputFiles({name,mimeType:'application/octet-stream',buffer:bytes});await expect(field(page,'inspect')).toBeEnabled()}
const expected=1000*.5e-6*((.99**2/2-.99**3/3)-(.01**2/2-.01**3/3))

test('quoted BOM CSV upload → explicit units/polarity → Python DPT → native core download and reload', async({page},info)=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message)); const f=fixtures(), bytes=Buffer.from(f.csv,'base64')
 await page.goto('/index.html');await page.getByRole('button',{name:'Measurement import',exact:true}).click()
 await field(page,'source-kind').selectOption('simulation');await field(page,'disclosure').fill('Independent analytic two-event simulation; not a calibrated device.')
 await field(page,'delimiter').selectOption(';');await upload(page,bytes,'quoted-analytic.csv');await inspect(page);await map(page);await declare(page)
 await field(page,'native-file').setInputFiles({name:'existing-native.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(f.native))})
 const data=await preview(page);expect(data.source.sha256).toBe(createHash('sha256').update(bytes).digest('hex'));expect(data.waveform.current_a[0]).toBe(10)
 await expect(field(page,'samples')).toContainText('10.000000');await expect(field(page,'statistics')).toContainText('Signed current (A)')
 const result=await analyze(page);expect(result.analysis.e_on).toBeCloseTo(expected,7);expect(result.analysis.e_off).toBeCloseTo(expected,7)
 await expect(field(page,'analysis')).toContainText('0.000083');await expect(field(page,'result')).toContainText('Absolute integration window (s)')
 const archive=await download(page,'native-download');expect(JSON.parse(archive)).toEqual(result.core_json_document)
 const checked=execFileSync(python,['-c',`import sys,json;from transistordatabase.core.serialization import core_from_json_document,core_to_json_document;c=core_from_json_document(json.load(sys.stdin));assert c.switch.e_on_data[0].r_g==2;assert c.switch.e_off_data[0].r_g==7;assert c.switch.raw_measurement_data[0].evidence.sha256=='${data.source.sha256}';print(len(c.switch.raw_measurement_data[0].gate_v))`],{cwd:repo,input:archive,encoding:'utf8'});expect(checked.trim()).toBe('1601')
 const review=JSON.parse(await download(page,'review-download'));expect(review.raw_record.gate_v).toHaveLength(1601)
 await field(page,'range-start').fill('0.000001');await field(page,'range-end').fill('0.000006');await page.screenshot({path:info.outputPath('measurement-csv-review.png'),fullPage:true});await field(page,'hash').scrollIntoViewIfNeeded();await page.screenshot({path:info.outputPath('measurement-csv-source-viewport.png')});await page.locator('.measurement-workbench .curve-plotter').nth(1).scrollIntoViewIfNeeded();await page.screenshot({path:info.outputPath('measurement-csv-current-viewport.png')});await field(page,'analysis').scrollIntoViewIfNeeded();await page.screenshot({path:info.outputPath('measurement-csv-analysis-viewport.png')})
 await field(page,'condition-v_g_on').fill('16');await expect(field(page,'result')).toHaveCount(0);await expect(field(page,'analyze')).toBeDisabled();expect(errors).toEqual([])
})

test('real EPC log retains 30 source integral records, original windows, and selected derived step',async({page},info)=>{
 await page.goto('/index.html?view=measurements');await field(page,'format').selectOption('log');await field(page,'source-kind').selectOption('simulation');await field(page,'disclosure').fill('Historical generic LTspice simulation, no calibrated measured device accuracy.')
 const bytes=await readFile('/home/tinix/claude_wsl/archive_ntbees2/sc/tests/DoublePulseTester_EPC8009_sync.log');await upload(page,bytes,'DoublePulseTester_EPC8009_sync.log');await inspect(page)
 for(const [key,value] of Object.entries({energy_off:'eoff',current:'idc',bus:'vdc',energy_unit:'J',current_unit:'A',bus_unit:'V'})) await field(page,'log-'+(key.endsWith('_unit') ? key.replace('_unit','-unit') : key)).selectOption(value)
 await field(page,'step-index').fill('2');const data=await preview(page);expect(data.log_records).toHaveLength(30);expect(data.log_records.slice(0,3).map(r=>r.energy_j)).toEqual([1.0141e-7,9.47901e-7,2.69621e-6]);expect(data.log_records[0].window_s).toEqual([39e-6,41e-6]);await expect(field(page,'log-records')).toContainText('1.0141000e-7')
 const result=await analyze(page);expect(result.analysis).toBeNull();expect(result.raw_record).toBeNull();expect(result.points.turn_off[0].i_x).toBe(5);expect(result.points.turn_off[0].evidence.method).toBe('supplied-log-integral')
 await page.screenshot({path:info.outputPath('measurement-log-review.png'),fullPage:true})
})

test('binary RAW trace selection reproduces CSV SI data and mobile review remains contained',async({page},info)=>{
 const f=fixtures();await page.setViewportSize({width:390,height:844});await page.goto('/index.html?view=measurements');await field(page,'format').selectOption('raw');await field(page,'source-kind').selectOption('simulation');await field(page,'disclosure').fill('Independent binary RAW fixture, no device calibration.');await upload(page,Buffer.from(f.raw,'base64'),'analytic.raw');await inspect(page);await map(page,true);await declare(page)
 const data=await preview(page);expect(data.waveform.current_a[0]).toBe(10);expect(data.waveform.gate_v[0]).toBe(15);const result=await analyze(page);expect(result.analysis.e_on).toBeCloseTo(expected,7)
 await expect(field(page,'samples')).toContainText('10.000000');expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(391)
 await page.screenshot({path:info.outputPath('measurement-mobile-raw.png'),fullPage:true});await field(page,'hash').scrollIntoViewIfNeeded();await page.screenshot({path:info.outputPath('measurement-mobile-source-viewport.png')});await page.locator('.measurement-workbench .curve-plotter').nth(1).scrollIntoViewIfNeeded();await page.screenshot({path:info.outputPath('measurement-mobile-current-viewport.png')});await field(page,'analysis').scrollIntoViewIfNeeded();await page.screenshot({path:info.outputPath('measurement-mobile-analysis-viewport.png')})
})

test('More Tools popup paints and hit-tests first/last tools on desktop and mobile',async({page},info)=>{
 for(const [width,height] of [[1440,844],[390,480]]) {
  await page.setViewportSize({width,height});await page.goto('/index.html');await page.getByRole('button',{name:'More Tools'}).click();const menu=page.locator('#more-tools-popup');await expect(menu).toBeVisible()
  for(const name of ['Waveform workbench','Batch Export']) {
   const button=menu.getByRole('button',{name,exact:true});await button.scrollIntoViewIfNeeded();expect(await button.evaluate(e=>{const r=e.getBoundingClientRect();return e.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2))})).toBe(true)
  }
  await page.screenshot({path:info.outputPath(`more-tools-${width}.png`)})
  await menu.getByRole('button',{name:'Image digitizer',exact:true}).click();await expect(page.getByRole('heading',{name:'Image digitizer',exact:true})).toBeVisible()
  await page.getByRole('button',{name:'Measurement import',exact:true}).click();await expect(field(page,'workbench')).toBeVisible()
  await page.getByRole('button',{name:'More Tools'}).click();await page.keyboard.press('Escape');await expect(menu).toHaveCount(0)
 }
})



test('missing gate extraction and held waveform responses never create stale review exports',async({page})=>{
 const f=fixtures();await page.goto('/index.html?view=measurements');await field(page,'delimiter').selectOption(';');await upload(page,Buffer.from(f.csv,'base64'),'analytic.csv');await inspect(page)
 for(const [key,name,unit] of [['time','time','us'],['voltage','V(ds)','V'],['current','I(d)','A']]) {await field(page,'map-'+key).selectOption(name);await field(page,'unit-'+key).selectOption(unit)}
 await field(page,'polarity-current').selectOption('-1');await preview(page);await field(page,'analyze').click();await expect(page.getByRole('alert')).toContainText('Map a gate waveform')
 await field(page,'map-gate').selectOption('V(gs)');await field(page,'unit-gate').selectOption('V');await declare(page);await preview(page)
 let release, fetched;const paused=new Promise(r=>{release=r}), ready=new Promise(r=>{fetched=r})
 await page.route('**/api/measurement/extract',async route=>{const response=await route.fetch();fetched();await paused;await route.fulfill({response})})
 await field(page,'analyze').click();await ready;await field(page,'condition-v_g_on').fill('16');release();await expect(field(page,'analyze')).toBeDisabled();await expect(field(page,'result')).toHaveCount(0)
 await upload(page,Buffer.from('time;V(ds);I(d);V(gs)\n0;0;1;15\n0;100;0;-5'),'invalid-time.csv');await inspect(page);await map(page);await field(page,'preview').click();await expect(page.getByRole('alert')).toContainText('increase strictly');await expect(field(page,'result')).toHaveCount(0)
})


