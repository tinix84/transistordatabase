import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { test, expect } from '@playwright/test'

const root = fileURLToPath(new URL('../../../../../', import.meta.url))
const base = process.env.NATIVE_EXPORT_BASE_URL || 'http://localhost:8003'
const python = process.env.NATIVE_EXPORT_PYTHON || '/home/tinix/miniconda3/envs/ntbees2/bin/python'
const selection = { component: 'switch', t_j: 37, v_supply: 430, v_g_on: 11, v_g_off: -3, r_g_on: 4, r_g_off: 7 }
const validator = `
import json, sys, zipfile
from io import BytesIO
from pathlib import Path
from scipy.io import loadmat
from transistordatabase.core.serialization import core_from_json_document
from transistordatabase.native_export import render_gecko
data=Path(sys.argv[1]).read_bytes()
kind=sys.argv[2]
if kind == 'matlab':
    mat=loadmat(BytesIO(data), simplify_cells=True)
    device=core_from_json_document(json.loads(str(mat['core_json'])))
    assert device.switch.metadata['source_id'] == 2**60+1
    assert device.thermal_properties.r_th_cs is None
elif kind == 'json':
    device=core_from_json_document(json.loads(data))
    assert device.switch.metadata['source_id'] == 2**60+1
    assert device.thermal_properties.r_th_cs is None
else:
    with zipfile.ZipFile(BytesIO(data)) as z:
        manifests=[n for n in z.namelist() if n.endswith('.manifest.json')]
        assert len(manifests) == (2 if kind == 'batch' else 1)
        assert len(z.namelist()) == 2*len(manifests)
        for name in manifests:
            manifest=json.loads(z.read(name))
            device=core_from_json_document(manifest['core_document'])
            assert device.switch.metadata['source_id'] == 2**60+1
            if device.metadata.name.endswith('雪'): assert device.thermal_properties.r_th_cs is None
            assert manifest['switching_current_coverage_a'] == [3,17]
            files=render_gecko(device, manifest['selection'])
            prefix=name.rsplit('/',1)[0]+'/' if '/' in name else ''
            assert z.read(prefix+manifest['file']) == files[manifest['file']]
print('Validated '+kind+' download')
`
async function download(page, kind) {
  const pending = page.waitForEvent('download')
  await page.getByTestId('download-export').click()
  const file = await pending
  const path = await file.path()
  expect(execFileSync(python, ['-c', validator, path, kind], { cwd: root }).toString()).toContain(`Validated ${kind}`)
  return { file, bytes: readFileSync(path) }
}
for (const mobile of [false, true]) {
  test.describe(mobile ? 'mobile export downloads' : 'desktop export downloads', () => {
    test.use(mobile ? { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } : { viewport: { width: 1440, height: 900 } })
    test('downloads actual complete JSON, MAT5, Gecko and batch bytes; rejects unsupported conditions', async ({ page }) => {
      const errors = []
      page.on('pageerror', error => errors.push(error.message))
      await page.goto(base + '/index.html?view=export')
      await page.getByRole('button', { name: 'Export', exact: true }).first().click()
      await expect(page.getByRole('heading', { name: 'Exporting Tools', exact: true })).toBeVisible()
      await page.getByRole('checkbox', { name: 'Browser export 雪', exact: false }).check()
      const json = await download(page, 'json')
      expect(json.bytes.toString()).toContain('1152921504606846977')
      await page.getByTestId('export-format').selectOption('matlab')
      expect((await download(page, 'matlab')).file.suggestedFilename()).toMatch(/\.mat$/)
      await page.getByTestId('export-format').selectOption('gecko')
      await expect(page.getByTestId('download-export')).toBeDisabled()
      for (const [key, value] of Object.entries(selection)) {
        if (key !== 'component') await page.getByTestId('gecko-' + key).fill(String(value))
      }
      expect((await download(page, 'gecko')).file.suggestedFilename()).toMatch(/\.zip$/)
      await page.getByRole('checkbox', { name: 'Browser export second', exact: false }).check()
      await download(page, 'batch')
      await page.getByTestId('gecko-t_j').fill('38')
      const failed = page.waitForResponse(response => response.url().endsWith('/api/transistors/batch_export'))
      await page.getByTestId('download-export').click()
      expect((await failed).status()).toBe(422)
      await expect(page.getByRole('alert')).toContainText(/recorded|exact|channel/i)
      expect(await page.locator('body').evaluate(element => element.scrollWidth <= window.innerWidth + 1)).toBe(true)
      await page.screenshot({ path: test.info().outputPath('export-downloads.png'), fullPage: true })
      await page.getByTestId('export-format').scrollIntoViewIfNeeded()
      await page.screenshot({ path: test.info().outputPath('export-selector-viewport.png') })
      await page.getByRole('button', { name: 'Open PLECS data workbench', exact: true }).click()
      await expect(page.getByTestId('plecs-xml')).toBeVisible()
      await page.goto(base + '/index.html?view=export')
      await page.getByRole('button', { name: 'Export', exact: true }).first().click()
      await page.getByRole('button', { name: 'Open DPT workbench', exact: true }).click()
      await expect(page.getByLabel('DPT transistor label')).toBeVisible()
      expect(errors).toEqual([])
    })

    test('compares recorded ratings while unknown resistance and unavailable power remain explicit', async ({ page }) => {
      const errors = []
      page.on('pageerror', error => errors.push(error.message))
      await page.goto(base + '/index.html')
      await expect(page.locator('.results-table')).toContainText('Unknown K/W')
      await page.locator('.results-table').getByRole('cell', { name: 'Unknown K/W', exact: true }).scrollIntoViewIfNeeded()
      await page.screenshot({ path: test.info().outputPath('unknown-thermal-search-viewport.png') })
      await page.getByRole('button', { name: 'Compare', exact: true }).first().click()
      await page.locator('.transistor-card').filter({ hasText: 'Browser export 雪' }).click()
      await page.locator('.transistor-card').filter({ hasText: 'Browser export second' }).click()
      const rows = page.locator('.comparison-table tbody tr')
      await expect(rows.filter({ hasText: 'Power Rating (W)' })).toContainText('Unavailable · no recorded power rating')
      await expect(rows.filter({ hasText: 'Thermal Resistance (K/W)' })).toContainText('Unknown')
      await expect(page.locator('.comparison-summary')).toContainText('(1/2 known)')
      await expect(page.locator('#voltageCurrentChart')).toBeVisible()
      await expect(page.locator('#powerThermalChart')).toHaveCount(0)
      await expect(page.locator('#radarChart')).toHaveCount(0)
      await expect(page.locator('.comparison-summary')).not.toContainText('Best Power Density')
      await page.screenshot({ path: test.info().outputPath('comparison-rating-scope.png'), fullPage: true })
      await rows.filter({ hasText: 'Power Rating (W)' }).scrollIntoViewIfNeeded()
      await page.screenshot({ path: test.info().outputPath('comparison-power-viewport.png') })
      expect(errors).toEqual([])
    })
  })
}
