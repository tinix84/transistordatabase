import { readFile } from 'node:fs/promises'
import { test, expect } from '@playwright/test'

function paretoResponse() {
  const rows = [
    ['cree-one', 'CREE ×1 + CREE rectifier ×1', 18.849, 0.004662, 853.2, 19.28],
    ['infineon-one', 'Infineon ×1 + CREE rectifier ×1', 17.623, 0.004179, 860.4, 20.42],
    ['infineon-two', 'Infineon ×2 + CREE rectifier ×1', 16.650, 0.003705, 1294.1, 31.2]
  ]
  return {
    topology: 'boost',
    boundary: {
      v_in: 100, v_out: 200, p_out: 1000, f_sw: 20000,
      l_value: 0.001, c_out: 0.001, t_j: 25, t_case: 60,
      mode: 'loss-balanced', conduction_regime: 'CCM'
    },
    mode: 'loss-balanced',
    conduction_regime: 'CCM',
    loss_scope: 'channel_conduction_only',
    objective_directions: {
      channel_conduction_loss_w: 'minimize',
      package_body_volume_mm3: 'minimize',
      component_cost_usd: 'minimize'
    },
    dominance_rule: 'All objectives are minimized. Exact objective ties remain non-dominated.',
    cost_basis: 'one-unit USD list price per physical package; counts are included',
    eligible_candidate_count: 3,
    pareto_candidate_ids: rows.map(([id]) => id),
    model_limitations: ['Switching, reverse-recovery, gate-drive, passive, and magnetic losses are excluded.'],
    candidates: rows.map(([candidate_id, label, loss, residual, volume, cost]) => ({
      candidate_id, label, eligible: true, pareto: true, ineligible_reasons: [],
      objectives: {
        channel_conduction_loss_w: loss,
        package_body_volume_mm3: volume,
        component_cost_usd: cost
      },
      operating_point: {
        model: { assumptions: ['Boost CCM, channel-conduction-only feedback.', 'Capacitor ESR is ignored.'] },
        calculated: { inductor_volt_second_residual_v: residual },
        losses: { scope: 'Included switch and rectifier channel conduction only.' },
        temperatures: {
          switch_c: null, diode_c: null,
          switch_unavailable_reason: 'Required TDB thermal resistances are unavailable.',
          diode_unavailable_reason: 'Required TDB thermal resistances are unavailable.'
        }
      },
      ratings: { pass: true },
      catalog: {
        switch: {
          orderable_part: candidate_id === 'cree-one' ? 'C3M0060065J' : 'IPBE65R050CFD7AATMA1',
          unit_cost_usd: candidate_id === 'cree-one' ? 9.64 : 10.78,
          cost_accessed_on: '2026-10-14',
          price_source_url: 'https://example.com/price',
          dimensions_source_url: 'https://example.com/datasheet'
        },
        diode: {
          orderable_part: 'C3M0060065J', unit_cost_usd: 9.64,
          cost_accessed_on: '2026-10-14',
          price_source_url: 'https://example.com/price',
          dimensions_source_url: 'https://example.com/datasheet'
        }
      }
    }))
  }
}

test('screens the notebook Boost boundary in the browser and exports provenance', async ({ page }) => {
  await page.route('**/api/transistors', route => route.fulfill({ json: [
    { metadata: { name: 'CREE_C3M0060065J' } },
    { metadata: { name: 'Infineon_IPBE65R050CFD7A' } }
  ] }))

  let capturedRequest
  await page.route('**/api/topology/pareto', async route => {
    capturedRequest = route.request().postDataJSON()
    await route.fulfill({ json: paretoResponse() })
  })

  await page.goto('/')
  await page.getByRole('button', { name: 'Topology', exact: true }).click()
  const panel = page.locator('details.pareto-panel')
  await expect(panel).toBeVisible()
  await panel.getByRole('button', { name: 'Calculate Pareto screen' }).click()

  await expect(panel.getByRole('heading', { name: 'Pareto results' })).toBeVisible()
  await expect(panel.locator('tbody')).toContainText('18.849')
  await expect(panel.locator('tbody')).toContainText('0.004662')
  await expect(panel.locator('tbody')).toContainText('Pareto')
  await expect(panel).toContainText('2026-10-14')
  await expect(panel).toContainText('Temperature remains unavailable')
  await expect(panel).toContainText('Capacitor ESR is ignored.')
  await expect(panel.locator('svg.pareto-plot')).toHaveCount(3)
  await expect(panel.locator('circle.frontier-point')).toHaveCount(9)

  expect(capturedRequest.boundary).toMatchObject({
    v_in: 100, v_out: 200, p_out: 1000, f_sw: 20000,
    l_value: 0.001, c_out: 0.001, t_j: 25, t_case: 60
  })
  expect(capturedRequest.candidates).toHaveLength(3)
  expect(capturedRequest.candidates.map(candidate => candidate.parallel_switches)).toEqual([1, 1, 2])
  expect(capturedRequest.candidates.map(candidate => candidate.v_g)).toEqual([15, 10, 10])
  expect(capturedRequest.catalog.CREE_C3M0060065J).toMatchObject({
    unit_cost_usd: 9.64,
    cost_accessed_on: '2026-10-14',
    length_mm: 9.125,
    price_source_url: 'https://www.digikey.com/en/products/detail/wolfspeed-inc/C3M0060065J/12525328'
  })

  const downloadPromise = page.waitForEvent('download')
  await panel.getByRole('button', { name: 'Download JSON' }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toMatch(/^boost-semiconductor-pareto-\d{4}-\d{2}-\d{2}\.json$/)
  const exported = JSON.parse(await readFile(await download.path(), 'utf8'))
  expect(exported).toMatchObject({
    workflow: 'fixed-boundary-boost-semiconductor-pareto',
    request: { boundary: { v_in: 100, v_out: 200 } },
    result: { pareto_candidate_ids: ['cree-one', 'infineon-one', 'infineon-two'] }
  })
  expect(exported.request.candidates).toHaveLength(3)
  expect(exported.request.candidates[0].candidate_id).toBe('cree-one')
  expect(exported.request.catalog.CREE_C3M0060065J.price_source_url).toContain('digikey.com')
})

test('sends the browser Pareto request to the running FastAPI endpoint', async ({ page }) => {
  await page.route('**/api/transistors', route => route.fulfill({ json: [
    { metadata: { name: '__pareto-browser-missing-switch__' } },
    { metadata: { name: '__pareto-browser-missing-diode__' } }
  ] }))

  await page.goto('/')
  await page.getByRole('button', { name: 'Topology', exact: true }).click()
  const panel = page.locator('details.pareto-panel')
  const firstCandidate = panel.locator('article.candidate-card').first()
  await firstCandidate.getByLabel('Switch TDB device').selectOption('__pareto-browser-missing-switch__')
  await firstCandidate.getByLabel('Rectifier TDB device').selectOption('__pareto-browser-missing-diode__')

  const responsePromise = page.waitForResponse(response => (
    response.url().endsWith('/api/topology/pareto') && response.request().method() === 'POST'
  ))
  await panel.getByRole('button', { name: 'Calculate Pareto screen' }).click()
  const response = await responsePromise
  expect(response.status()).toBe(200)

  const result = await response.json()
  expect(result.eligible_candidate_count).toBe(0)
  expect(result.candidates[0].eligible).toBe(false)
  expect(result.candidates[0].ineligible_reasons.join(' ')).toContain(
    '__pareto-browser-missing-switch__'
  )
  await expect(panel.getByRole('heading', { name: 'Pareto results' })).toBeVisible()
  await expect(panel.locator('.candidate-result').first()).toContainText(
    '__pareto-browser-missing-switch__'
  )
})
