import { test, expect } from '@playwright/test'
import { readFile } from 'node:fs/promises'
const constant='Verifier_Grid_Constant', zero='Verifier_Grid_Zero', missing='Verifier_Grid_Missing', knownZeroRg='Verifier_Grid_KnownZeroRg'
const lineEndpoint='/api/topology/switching-frequency-grid', paretoEndpoint='/api/topology/switching-frequency-pareto'
const close=(actual,expected,tolerance=1e-6)=>expect(Math.abs(actual-expected)).toBeLessThanOrEqual(tolerance)
async function evidence(info,name,data){await info.attach(name,{body:JSON.stringify(data,null,2),contentType:'application/json'})}
async function download(page,button,info,name){const ready=page.waitForEvent('download');await button.click();const item=await ready;const path=info.outputPath(name);await item.saveAs(path);return readFile(path,'utf8')}
async function responseFor(page,endpoint,action){const ready=page.waitForResponse(r=>r.url().endsWith(endpoint)&&r.request().method()==='POST');await action();const response=await ready;return {status:response.status(),body:await response.json(),request:response.request().postDataJSON()}}
function oracle(point,frequency,energy=.003){const heat=frequency*energy,current=(1000+heat)/100,duty=1-5/current;close(point.losses.total_w,heat);close(point.calculated.i_in_avg_a,current);close(point.calculated.duty_cycle,duty);close(point.calculated.inductor_volt_second_residual_v,100-200*(1-duty));expect(Math.abs(point.performance.power_balance_residual_w)).toBeLessThan(1e-6);expect(point.waveforms ?? null).toBeNull()}
function calculator(page){return page.locator('.topology-calculator')}
const numericNames={"Input voltage":"Input voltage V","Output voltage magnitude":"Output voltage magnitude V","Output power":"Output power W","Switching frequency":"Switching frequency kHz","Inductance":"Inductance µH","Output capacitance":"Output capacitance µF","Junction temperature":"Junction temperature °C","Case / sink reference temperature":"Case / sink reference temperature °C","Switch gate voltage":"Switch gate voltage V","Diode channel gate voltage (optional; required for gate-conditioned curves)":"Diode channel gate voltage (optional; required for gate-conditioned curves) V","Turn-off gate voltage (required)":"Turn-off gate voltage (required) V","Turn-on gate resistance (required)":"Turn-on gate resistance (required) Ω","Turn-off gate resistance (required)":"Turn-off gate resistance (required) Ω","X minimum":"X minimum (kHz)","X maximum":"X maximum (kHz)","X points":"X points"}
const strictSourceLabels=new Set(['Junction temperature','Switch gate voltage','Diode channel gate voltage (optional; required for gate-conditioned curves)','Turn-off gate voltage (required)','Turn-on gate resistance (required)','Turn-off gate resistance (required)'])
function numericControl(root,label){return root.getByRole(strictSourceLabels.has(label)?'textbox':'spinbutton',{name:numericNames[label] || label,exact:true})}
const catalogNumbers=new Set(['One-unit price (USD)','Length (mm)','Width (mm)','Height (mm)'])
async function lineControls(page,{source=constant,minimum='1',maximum='19',points='3',resistance='10'}={}){
 const root=calculator(page), form=root.locator('.parameters-panel')
 await root.getByRole('button',{name:'Boost',exact:false}).first().click()
 await form.getByRole('combobox',{name:'Calculation mode',exact:true}).selectOption('switching-loss-balanced')
 for(const [label,value] of Object.entries({'Input voltage':'100','Output voltage magnitude':'200','Output power':'1000','Switching frequency':'20','Inductance':'1000','Output capacitance':'1000','Junction temperature':'25','Case / sink reference temperature':'25','Switch gate voltage':'15'}))await numericControl(form,label).fill(value)
 await form.getByRole('combobox',{name:'Switch transistor',exact:true}).selectOption(source)
 await form.getByRole('combobox',{name:'Diode transistor',exact:true}).selectOption(source)
 await numericControl(form,'Diode channel gate voltage (optional; required for gate-conditioned curves)').fill('')
 await numericControl(form,'Turn-off gate voltage (required)').fill('-5')
 await numericControl(form,'Turn-on gate resistance (required)').fill(resistance)
 await numericControl(form,'Turn-off gate resistance (required)').fill(resistance)
 await form.locator('.sweep-toggle input').check()
 await numericControl(form,'X minimum').fill(minimum)
 await numericControl(form,'X maximum').fill(maximum)
 await numericControl(form,'X points').fill(points)
 await form.getByRole('combobox',{name:'Plotted metric',exact:true}).selectOption('p_loss_total_w')
 return {root,form}
}
async function screenControls(page,{second=constant,frequencies='10000,20000'}={}){
 const panel=page.locator('.pareto-panel')
 await panel.getByTestId('pareto-switching-mode').check()
 await panel.getByTestId('pareto-frequency-grid').fill(frequencies)
 for(const [label,value] of Object.entries({'Input voltage (V)':'100','Output voltage (V)':'200','Output power (W)':'1000','Switching frequency (kHz)':'20','Inductance (µH)':'1000','Output capacitance (µF)':'1000','Junction temperature (°C)':'25','Fixed case/sink temperature (°C)':'25'}))await panel.getByRole(label==='Junction temperature (°C)'?'textbox':'spinbutton',{name:label,exact:true}).fill(value)
 const cards=panel.locator('.candidate-card')
 for(let count=await cards.count();count>2;count--){await cards.last().getByRole('button',{name:/^Remove /}).click();await expect(cards).toHaveCount(count-1)}
 await expect(cards).toHaveCount(2)
 for(let index=0;index<2;index++){const card=cards.nth(index),name=index?second:constant;await card.getByRole('textbox',{name:'Candidate label',exact:true}).fill(index?'Synthetic B':'Synthetic A');await card.getByRole('spinbutton',{name:'Parallel switches',exact:true}).fill('1');await card.getByRole('spinbutton',{name:'Parallel rectifiers',exact:true}).fill('1');await card.getByRole('combobox',{name:'Switch TDB device',exact:true}).selectOption(name);await card.getByRole('combobox',{name:'Rectifier TDB device',exact:true}).selectOption(name);await card.getByRole('textbox',{name:'Switch gate voltage (V)',exact:true}).fill('15');await card.getByRole('textbox',{name:'Rectifier curve gate voltage (V, optional)',exact:true}).fill('');await card.getByRole('textbox',{name:'Turn-off gate voltage (V)',exact:true}).fill('-5');await card.getByRole('textbox',{name:'Turn-on gate resistance (Ω)',exact:true}).fill('10');await card.getByRole('textbox',{name:'Turn-off gate resistance (Ω)',exact:true}).fill('10')}
 for(const card of await panel.locator('.catalog-card').all()){for(const [label,value] of Object.entries({'Orderable part':'Independent analytic fixture','One-unit price (USD)':'1','Price accessed on':'2026-10-01','Price source URL':'https://fixture.invalid/price','Length (mm)':'2','Width (mm)':'3','Height (mm)':'1','Dimensions source URL':'https://fixture.invalid/body'}))await card.getByRole(catalogNumbers.has(label)?'spinbutton':'textbox',{name:label,exact:true}).fill(value)}
 return panel
}
test.beforeEach(async({page},info)=>{
 info.capture={pageerrors:[],failed_requests:[],http5xx:[],console:[],calculation_posts:[]}
 page.on('pageerror',e=>info.capture.pageerrors.push(e.message));page.on('requestfailed',r=>info.capture.failed_requests.push({url:r.url(),failure:r.failure()}));page.on('response',r=>{if(r.status()>=500)info.capture.http5xx.push({url:r.url(),status:r.status()})});page.on('console',m=>info.capture.console.push({type:m.type(),text:m.text()}))
 page.on('request',item=>{if(item.method()==='POST'&&[lineEndpoint,paretoEndpoint].some(endpoint=>item.url().endsWith(endpoint)))info.capture.calculation_posts.push({url:item.url(),body:item.postDataJSON()})})
 await page.goto('/index.html');await page.getByRole('navigation').getByRole('button',{name:'Topology',exact:true}).click();await expect(page.getByRole('heading',{name:'Topology Calculator',exact:true})).toBeVisible()
})
test.afterEach(async({page},info)=>{await evidence(info,'browser-console-network',info.capture);expect(info.capture.pageerrors).toEqual([]);expect(info.capture.failed_requests).toEqual([]);expect(info.capture.http5xx).toEqual([]);if(info.project.name==='mobile-390')expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(391)})
test('Boost frequency acceptance mixed line real equations export and stale clear',async({page},info)=>{
 const {root,form}=await lineControls(page)
 const response=await responseFor(page,lineEndpoint,()=>form.getByRole('button',{name:'Calculate',exact:true}).click())
 expect(response.status).toBe(200);const result=response.body
 expect(result.sweep.rows.map(r=>r.frequency_hz)).toEqual([1000,10000,19000]);expect(result.sweep.rows.map(r=>r.status)).toEqual(['unsupported','qualified','qualified']);expect(result.sweep.rows[0].value).toBeNull()
 for(const row of result.sweep.rows.slice(1))oracle(row.operating_point,row.frequency_hz)
 expect(result.summary_frequency_hz).toBe(10000);expect(result.sweep.scalar_call_count).toBe(3);expect(result.source_snapshots[constant]).toBeNull();expect(result.sweep.voltage_law_enforced).toBe(false)
 await expect(root.getByTestId('switching-frequency-rows')).toContainText('unsupported');await expect(root.getByTestId('switching-frequency-rows')).toContainText('Voltage closure is not enforced or proved')
 const exported=JSON.parse(await download(page,root.locator('.header-actions').getByRole('button',{name:'Download JSON',exact:true}),info,'mixed-frequency-line.json'));expect(exported.result.sweep).toEqual(result.sweep)
 await page.screenshot({path:info.outputPath('mixed-frequency-line.png'),fullPage:true});await evidence(info,'actual-line-request-response',response)
 await numericControl(form,'Output power').fill('1001');await expect(root.getByTestId('switching-frequency-rows')).toHaveCount(0)
})
test('Boost frequency acceptance complete populations ties CSV JSON and residual eligibility',async({page},info)=>{
 const panel=await screenControls(page)
 let response=await responseFor(page,paretoEndpoint,()=>panel.getByRole('button',{name:'Calculate Pareto screen',exact:true}).click())
 expect(response.status).toBe(200);expect(response.body.rows).toHaveLength(2);expect(response.body.scalar_call_count).toBe(4)
 for(const row of response.body.rows){expect(row.population.candidates).toHaveLength(2);expect(row.population.eligible_candidate_count).toBe(2);expect(row.population.pareto_candidate_ids).toHaveLength(2);for(const item of row.population.candidates)oracle(item.operating_point,row.frequency_hz)}
 await panel.getByTestId('pareto-result-frequency').selectOption('1');await expect(panel.locator('.results')).toContainText('60.000')
 const csv=await download(page,panel.getByRole('button',{name:'Download scalar CSV',exact:true}),info,'complete-populations.csv');expect(csv.trim().split(/\r?\n/)).toHaveLength(5);expect(csv).toContain('voltage_law_residual_v');expect(csv).not.toContain('graph_i_e')
 const exported=JSON.parse(await download(page,panel.getByRole('button',{name:'Download JSON',exact:true}),info,'complete-populations.json'));expect(exported.result.rows).toEqual(response.body.rows)
 await panel.getByTestId('pareto-voltage-tolerance').fill('0');await expect(panel.locator('.results')).toHaveCount(0)
 response=await responseFor(page,paretoEndpoint,()=>panel.getByRole('button',{name:'Calculate Pareto screen',exact:true}).click())
 for(const row of response.body.rows){expect(row.population.eligible_candidate_count).toBe(0);for(const item of row.population.candidates){expect(item.operating_point).not.toBeNull();expect(item.ineligible_reasons.join(' ')).toContain('voltage residual')}}
 await evidence(info,'actual-tolerance-populations',response);await page.screenshot({path:info.outputPath('residual-qualified-populations.png'),fullPage:true})
})
test('Boost frequency acceptance known zero and missing energy stay distinct',async({page},info)=>{
 const {form}=await lineControls(page,{source:zero,minimum:'10',maximum:'20',points:'2'})
 let response=await responseFor(page,lineEndpoint,()=>form.getByRole('button',{name:'Calculate',exact:true}).click())
 expect(response.status).toBe(200);expect(response.body.sweep.values).toEqual([0,0]);for(const row of response.body.sweep.rows)oracle(row.operating_point,row.frequency_hz,0)
 await form.getByRole('combobox',{name:'Switch transistor',exact:true}).selectOption(missing);await form.getByRole('combobox',{name:'Diode transistor',exact:true}).selectOption(missing)
 response=await responseFor(page,lineEndpoint,()=>form.getByRole('button',{name:'Calculate',exact:true}).click())
 expect(response.status).toBe(200);expect(response.body.sweep.values).toEqual([null,null]);expect(response.body.summary_frequency_hz).toBeNull();expect(response.body.sweep.rows.every(r=>r.status==='unsupported'&&r.diagnostic)).toBe(true)
 await evidence(info,'missing-source-actual-response',response)
})
test('Boost frequency acceptance bounded route precision and old opt-in hints',async({page},info)=>{
 const {form}=await lineControls(page,{source:knownZeroRg,minimum:'10',maximum:'20',points:'2',resistance:'0'})
 const successful=await responseFor(page,lineEndpoint,()=>form.getByRole('button',{name:'Calculate',exact:true}).click())
 expect(successful.status).toBe(200);const request=successful.request
 expect(request.r_g_on).toBe(0);expect(request.r_g_off).toBe(0)
 for(const row of successful.body.sweep.rows){expect(row.status).toBe('qualified');oracle(row.operating_point,row.frequency_hz);expect(row.operating_point.model.curve_conditions.e_on.r_g).toBe(0);expect(row.operating_point.model.curve_conditions.e_off.r_g).toBe(0)}
 const input=numericControl(form,'Turn-on gate resistance (required)'),posts=info.capture.calculation_posts.length
 await expect(input).toHaveAttribute('type','text')
 await input.fill('1e-400');await form.getByRole('button',{name:'Calculate',exact:true}).click();await expect(form.getByRole('alert')).toBeVisible();expect(info.capture.calculation_posts).toHaveLength(posts)
 await input.fill('');await form.getByRole('button',{name:'Calculate',exact:true}).click();await expect(form.getByRole('alert')).toBeVisible();expect(info.capture.calculation_posts).toHaveLength(posts)
 await input.fill('0');const restored=await responseFor(page,lineEndpoint,()=>form.getByRole('button',{name:'Calculate',exact:true}).click());expect(restored.status).toBe(200);expect(restored.request.r_g_on).toBe(0);expect(restored.body.sweep.rows.every(row=>row.status==='qualified')).toBe(true)
 const invalid={...request,sweep:{...request.sweep,x_points:33}}
 let response=await page.request.post(lineEndpoint,{data:invalid});expect(response.status()).toBe(422)
 response=await page.request.post('/api/topology/calculate',{data:request});expect(response.status()).toBe(422);expect((await response.json()).detail).toContain(lineEndpoint)
 const raw=JSON.stringify(request).replace('"x_min":10000','"x_min":1e-400')
 expect(raw).toContain('"x_min":1e-400');response=await page.request.post(lineEndpoint,{data:raw,headers:{'Content-Type':'application/json'}});expect(response.status()).toBe(422)
 response=await page.request.post(lineEndpoint,{data:{...request,t_j:'25.0000000000000000001'}});expect(response.status()).toBe(422)
 response=await page.request.post(lineEndpoint,{data:{...request,r_g_on:false}});expect(response.status()).toBe(422)
 response=await page.request.post('/api/topology/calculate',{data:{topology:'boost',mode:'estimate',v_in:'100',v_out:'200',p_out:'1000',f_sw:20000,l_value:.001,c_out:.001}});expect(response.status()).toBe(200);expect((await response.json()).model.mode).toBe('estimate')
 const panel=await screenControls(page);await panel.getByTestId('pareto-frequency-grid').fill('10000,')
 await panel.getByRole('button',{name:'Calculate Pareto screen',exact:true}).click();await expect(panel.getByRole('alert')).toContainText('positive finite')
 await evidence(info,'actual-known-zero-and-precision-boundaries',{known_zero:successful,restored_zero:restored,underflow_and_blank_ui_posts:0,quoted_number_status:422,bool_status:422,legacy_quoted_control_status:200})
 await page.screenshot({path:info.outputPath('known-zero-raw-token-rejection.png'),fullPage:true})
})
test('Boost frequency acceptance legacy estimate keeps omitted policy shape',async({page},info)=>{
 const form=calculator(page).locator('.parameters-panel')
 await calculator(page).getByRole('button',{name:'Boost',exact:false}).first().click()
 await form.getByRole('combobox',{name:'Calculation mode',exact:true}).selectOption('estimate')
 for(const [label,value] of Object.entries({'Input voltage':'100','Output voltage magnitude':'200','Output power':'1000','Switching frequency':'20','Inductance':'1000','Output capacitance':'1000'}))await numericControl(form,label).fill(value)
 const response=await responseFor(page,'/api/topology/calculate',()=>form.getByRole('button',{name:'Calculate',exact:true}).click())
 expect(response.status).toBe(200);expect(response.request.sweep).toBeUndefined();expect(response.body.source_snapshots).toBeUndefined();expect(response.body.sweep).toBeUndefined();expect(response.body.model.mode).toBe('estimate');await evidence(info,'legacy-estimate-real-response',response)
})
