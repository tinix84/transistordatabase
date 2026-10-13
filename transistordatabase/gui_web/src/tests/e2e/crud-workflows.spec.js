/**
 * Browser-level CRUD checks run against an in-memory API fixture.
 * No request from these tests reaches the repository's JSON database.
 */
import { test, expect } from '@playwright/test'

const DEVICE_NAME = 'E2E Lifecycle Device'

function fixtureDevice(name = DEVICE_NAME) {
  return {
    metadata: {
      name,
      type: 'MOSFET',
      manufacturer: 'Fixture Semiconductor',
      housing_type: 'TO263',
      author: 'E2E Fixture',
      comment: 'Isolated browser fixture'
    },
    electrical: {
      v_abs_max: 650,
      i_abs_max: 120,
      i_cont: 60,
      t_j_max: 175
    },
    thermal: {
      r_th_cs: 0.24,
      r_th_switch_cs: 0.12,
      r_th_diode_cs: 0.18,
      housing_area: 0.0012,
      cooling_area: 0.0008
    },
    switch: { channel_count: 1, e_on_count: 1, e_off_count: 1 },
    diode: { channel_count: 1, e_rr_count: 0 },
    capacitances: { c_oss_count: 1, c_iss_count: 0, c_rss_count: 0 }
  }
}

async function installIsolatedApi(page, options = {}) {
  const records = [fixtureDevice(options.deviceName || DEVICE_NAME)]
  for (const name of options.additionalNames || []) records.push(fixtureDevice(name))
  const requests = []
  const unmatchedRequests = []
  apiMocks.set(page, { unmatchedRequests })

  await page.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    const method = request.method()
    const path = url.pathname
    const requestRecord = { method, path, name: url.searchParams.get('name') }
    requests.push(requestRecord)

    if (method === 'GET' && path === '/api/transistors') {
      await route.fulfill({ status: 200, json: records })
      return
    }

    if (method === 'GET' && path.startsWith('/api/transistors/') && path.endsWith('/curves')) {
      await route.fulfill({
        status: 200,
        json: { switch: {}, diode: {}, capacitance: {} }
      })
      return
    }

    if (method === 'POST' && path === '/api/transistors') {
      const payload = request.postDataJSON()
      requestRecord.payload = payload
      if (records.some(record => record.metadata.name === payload.metadata.name)) {
        requestRecord.status = 409
        await route.fulfill({
          status: 409,
          json: { detail: `Transistor '${payload.metadata.name}' already exists.` }
        })
        return
      }
      records.push(payload)
      requestRecord.status = 200
      await route.fulfill({
        status: 200,
        json: { id: payload.metadata.name, message: 'Transistor created successfully' }
      })
      return
    }

    if (method === 'PUT' && path === '/api/transistor-record') {
      const id = requestRecord.name
      if (options.failUpdate) {
        await route.fulfill({
          status: 500,
          json: { detail: 'Fixture update failed.' }
        })
        return
      }
      const payload = request.postDataJSON()
      requestRecord.payload = payload
      const index = records.findIndex(record => record.metadata.name === id)
      if (index < 0 || payload.metadata.name !== id) {
        await route.fulfill({ status: 404, json: { detail: `No exact fixture named ${id}` } })
        return
      }
      records[index] = payload
      await route.fulfill({
        status: 200,
        json: { message: 'Transistor updated successfully' }
      })
      return
    }

    if (method === 'DELETE' && path === '/api/transistor-record') {
      const id = requestRecord.name
      if (options.failDelete) {
        await route.fulfill({
          status: 500,
          json: { detail: 'Fixture delete failed.' }
        })
        return
      }
      const index = records.findIndex(record => record.metadata.name === id)
      if (index < 0) {
        await route.fulfill({ status: 404, json: { detail: `No exact fixture named ${id}` } })
        return
      }
      records.splice(index, 1)
      await route.fulfill({
        status: 200,
        json: { message: 'Transistor deleted successfully' }
      })
      return
    }

    unmatchedRequests.push(`${method} ${url.pathname}${url.search}`)
    await route.fulfill({
      status: 501,
      json: { detail: `Unmocked API request: ${method} ${url.pathname}${url.search}` }
    })
  })

  return { records, requests }
}

async function fillCreateForm(page, name) {
  await page.locator('#name').fill(name)
  await page.locator('#type').selectOption('MOSFET')
  await page.locator('#manufacturer').fill('Fixture Semiconductor')
  await page.locator('#housing').selectOption('TO247')
  await page.locator('#v_abs_max').fill('900')
  await page.locator('#i_abs_max').fill('80')
  await page.locator('#i_cont').fill('50')
  await page.locator('#t_j_max').fill('175')
  await page.locator('#r_th_cs').fill('0.25')
  await page.locator('#r_th_switch_cs').fill('0.12')
  await page.locator('#r_th_diode_cs').fill('0.18')
  await page.locator('#housing_area').fill('1')
  await page.locator('#cooling_area').fill('0.5')
}

const apiMocks = new WeakMap()

test.describe('Database lifecycle from search results', () => {
  test.afterEach(async ({ page }) => {
    expect(apiMocks.get(page)?.unmatchedRequests || []).toEqual([])
  })

  test('creates an isolated record and shows it in search results', async ({ page }) => {
    await installIsolatedApi(page)
    await page.goto('/')

    await page.getByRole('button', { name: 'Create/Edit' }).click()
    await fillCreateForm(page, 'E2E_Created_Device')
    await page.getByRole('button', { name: 'Create', exact: true }).click()

    await expect(page.getByRole('status')).toHaveText('Transistor saved successfully.')
    await expect(page.getByTestId('transistor-row').filter({ hasText: 'E2E_Created_Device' })).toBeVisible()
  })

  test('rejects a duplicate create and leaves the populated fixture unchanged', async ({ page }) => {
    const api = await installIsolatedApi(page)
    const originalRecords = structuredClone(api.records)
    await page.goto('/')

    await page.getByRole('button', { name: 'Create/Edit' }).click()
    await fillCreateForm(page, DEVICE_NAME)
    await page.getByRole('button', { name: 'Create', exact: true }).click()

    await expect(page.getByRole('alert'))
      .toHaveText(`Transistor '${DEVICE_NAME}' already exists.`)
    expect(api.records).toEqual(originalRecords)
    const creation = api.requests.find(request => request.method === 'POST')
    expect(creation.status).toBe(409)
    expect(creation.payload.metadata.name).toBe(DEVICE_NAME)
    expect(creation.payload.thermal).toMatchObject({
      r_th_switch_cs: 0.12,
      r_th_diode_cs: 0.18,
      housing_area: 0.0001,
      cooling_area: 0.00005
    })
  })

  test('edits a table result through the update endpoint', async ({ page }) => {
    const api = await installIsolatedApi(page)
    await page.goto('/')

    const row = page.getByTestId('transistor-row').filter({ hasText: DEVICE_NAME })
    await row.getByRole('button', { name: 'Edit' }).click()

    await expect(page.getByRole('heading', { name: 'Edit Transistor' })).toBeVisible()
    await expect(page.locator('#name')).toHaveAttribute('readonly', '')
    await expect(page.locator('#manufacturer')).toHaveValue('Fixture Semiconductor')
    await expect(page.locator('#housing_area')).toHaveValue('12')
    await expect(page.locator('#cooling_area')).toHaveValue('8')
    await page.locator('#manufacturer').fill('Updated Fixture Semiconductor')
    await page.getByRole('button', { name: 'Update', exact: true }).click()

    await expect(page.getByRole('status')).toHaveText('Transistor saved successfully.')
    await expect(page.getByTestId('transistor-row').filter({ hasText: 'Updated Fixture Semiconductor' })).toBeVisible()
    const update = api.requests.find(request => request.method === 'PUT')
    expect(update.path).toBe('/api/transistor-record')
    expect(update.name).toBe(DEVICE_NAME)
    expect(update.payload.metadata.manufacturer).toBe('Updated Fixture Semiconductor')
    expect(update.payload.thermal.housing_area).toBe(0.0012)
    expect(update.payload.thermal.cooling_area).toBe(0.0008)
  })

  test('shows update errors inline and keeps the edit form open', async ({ page }) => {
    const api = await installIsolatedApi(page, { failUpdate: true })
    await page.goto('/')

    const row = page.getByTestId('transistor-row').filter({ hasText: DEVICE_NAME })
    await row.getByRole('button', { name: 'Edit' }).click()
    await page.locator('#manufacturer').fill('Updated Fixture Semiconductor')
    await page.getByRole('button', { name: 'Update', exact: true }).click()

    await expect(page.getByRole('alert')).toHaveText('Fixture update failed.')
    await expect(page.getByRole('heading', { name: 'Edit Transistor' })).toBeVisible()
    expect(api.requests.some(request => request.method === 'PUT')).toBe(true)
  })

  test('confirms a table delete, calls the endpoint, and refreshes results', async ({ page }) => {
    const api = await installIsolatedApi(page)
    await page.goto('/')

    const row = page.getByTestId('transistor-row').filter({ hasText: DEVICE_NAME })
    let confirmation = ''
    page.once('dialog', async dialog => {
      confirmation = dialog.message()
      await dialog.accept()
    })
    await row.getByRole('button', { name: 'Delete' }).click()

    expect(confirmation).toContain(DEVICE_NAME)
    await expect(page.getByTestId('transistor-row')).toHaveCount(0)
    await expect(page.getByRole('status')).toHaveText(DEVICE_NAME + ' was deleted.')
    expect(api.requests.filter(request => request.method === 'DELETE')).toHaveLength(1)
  })

  test('canceling a delete leaves the isolated record untouched', async ({ page }) => {
    const api = await installIsolatedApi(page)
    await page.goto('/')

    const row = page.getByTestId('transistor-row').filter({ hasText: DEVICE_NAME })
    page.once('dialog', dialog => dialog.dismiss())
    await row.getByRole('button', { name: 'Delete' }).click()

    await expect(row).toBeVisible()
    expect(api.requests.some(request => request.method === 'DELETE')).toBe(false)
  })

  test('shows delete errors to the user', async ({ page }) => {
    const api = await installIsolatedApi(page, { failDelete: true })
    await page.goto('/')

    const row = page.getByTestId('transistor-row').filter({ hasText: DEVICE_NAME })
    page.once('dialog', dialog => dialog.accept())
    await row.getByRole('button', { name: 'Delete' }).click()

    await expect(page.getByRole('alert')).toHaveText('Fixture delete failed.')
    await expect(row).toBeVisible()
    expect(api.requests.some(request => request.method === 'DELETE')).toBe(true)
  })

  test('card results expose edit and confirmed delete actions', async ({ page }) => {
    const api = await installIsolatedApi(page)
    await page.goto('/')

    await page.getByRole('button', { name: /Cards/ }).click()
    const card = page.getByTestId('transistor-card').filter({ hasText: DEVICE_NAME })
    await expect(card.getByRole('button', { name: 'Edit' })).toBeVisible()
    await card.getByRole('button', { name: 'Edit' }).click()
    await expect(page.getByRole('heading', { name: 'Edit Transistor' })).toBeVisible()
    await page.getByRole('button', { name: 'Cancel', exact: true }).first().click()

    await page.getByRole('button', { name: 'Search Database' }).click()
    await page.getByRole('button', { name: /Cards/ }).click()
    const visibleCard = page.getByTestId('transistor-card').filter({ hasText: DEVICE_NAME })
    page.once('dialog', dialog => dialog.accept())
    await visibleCard.getByRole('button', { name: 'Delete' }).click()

    await expect(page.getByTestId('transistor-card')).toHaveCount(0)
    expect(api.requests.filter(request => request.method === 'DELETE')).toHaveLength(1)
  })

  test('preserves exact spaced and URL-reserved identity for update and delete', async ({ page }) => {
    const reservedName = 'E2E #Lifecycle?Device'
    const similarName = 'E2E '
    const api = await installIsolatedApi(page, {
      deviceName: reservedName,
      additionalNames: [similarName]
    })
    await page.goto('/')

    const row = page.getByTestId('transistor-row').filter({ hasText: reservedName })
    await row.getByRole('button', { name: 'Edit' }).click()
    await expect(page.locator('#housing')).toHaveValue('TO263')
    await page.locator('#manufacturer').fill('Reserved Name Updated')
    await page.getByRole('button', { name: 'Update', exact: true }).click()
    await expect(page.getByRole('status')).toHaveText('Transistor saved successfully.')

    const update = api.requests.find(request => request.method === 'PUT')
    expect(update.path).toBe('/api/transistor-record')
    expect(update.name).toBe(reservedName)
    expect(api.records.find(record => record.metadata.name === reservedName).metadata.manufacturer)
      .toBe('Reserved Name Updated')
    expect(api.records.find(record => record.metadata.name === similarName).metadata.manufacturer)
      .toBe('Fixture Semiconductor')

    page.once('dialog', dialog => dialog.accept())
    await page.getByTestId('transistor-row')
      .filter({ hasText: reservedName })
      .getByRole('button', { name: 'Delete' })
      .click()
    await expect(page.getByRole('status')).toHaveText(reservedName + ' was deleted.')
    await expect(page.getByTestId('transistor-row').filter({ hasText: reservedName })).toHaveCount(0)

    const deletion = api.requests.find(request => request.method === 'DELETE')
    expect(deletion.name).toBe(reservedName)
    expect(api.records.map(record => record.metadata.name)).toEqual([similarName])
  })
})
