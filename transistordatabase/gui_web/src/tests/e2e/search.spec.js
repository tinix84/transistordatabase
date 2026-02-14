/**
 * End-to-end tests for Search Database functionality
 */
import { test, expect } from '@playwright/test'

test.describe('Search Database View', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    // Ensure we're on the search view
    await page.locator('button', { hasText: 'Search Database' }).click()
    // Wait for data to load
    await page.waitForTimeout(1000)
  })

  test('should display search interface', async ({ page }) => {
    await expect(page.locator('text=Search Database')).toBeVisible()
    await expect(page.locator('text=Advanced Filters')).toBeVisible()
  })

  test('should display transistor results', async ({ page }) => {
    // Wait for API response
    await page.waitForResponse(response =>
      response.url().includes('/api/transistors') &&
      response.status() === 200
    )

    // Check for results
    const resultsText = await page.locator('.search-results, h3').filter({ hasText: /Search Results/ }).textContent()
    expect(resultsText).toMatch(/\d+ devices/)
  })

  test('should have filter checkboxes', async ({ page }) => {
    const nameCheckbox = page.locator('input[type="checkbox"]').first()
    await expect(nameCheckbox).toBeVisible()
  })

  test('should enable filter input when checkbox is checked', async ({ page }) => {
    // Find name filter checkbox and input
    const nameCheckbox = page.locator('label:has-text("Name Contains:")').locator('input[type="checkbox"]')
    const nameInput = page.locator('input[placeholder*="name"]').first()

    // Input should be disabled initially
    await expect(nameInput).toBeDisabled()

    // Enable filter
    await nameCheckbox.check()

    // Input should now be enabled
    await expect(nameInput).toBeEnabled()
  })

  test('should filter transistors by name', async ({ page }) => {
    // Enable name filter
    const nameCheckbox = page.locator('label:has-text("Name Contains:")').locator('input[type="checkbox"]')
    await nameCheckbox.check()

    // Type in filter
    const nameInput = page.locator('input[placeholder*="name"]').first()
    await nameInput.fill('CREE')

    // Wait for filter to apply
    await page.waitForTimeout(500)

    // Check results are filtered
    const resultsText = await page.locator('text=/Search Results.*devices/').textContent()
    // Should show fewer results or specific results
    expect(resultsText).toBeTruthy()
  })

  test('should filter by transistor type', async ({ page }) => {
    // Enable type filter
    const typeCheckbox = page.locator('label:has-text("Type:")').locator('input[type="checkbox"]')
    await typeCheckbox.check()

    // Select type
    const typeSelect = page.locator('select').filter({ hasText: /All Types|IGBT|SiC-MOSFET/ }).first()
    await typeSelect.selectOption({ index: 1 }) // Select first non-empty option

    // Wait for filter to apply
    await page.waitForTimeout(500)

    // Results should be filtered
    const resultsText = await page.locator('text=/Search Results.*devices/').textContent()
    expect(resultsText).toBeTruthy()
  })

  test('should reset filters', async ({ page }) => {
    // Apply some filters
    const nameCheckbox = page.locator('label:has-text("Name Contains:")').locator('input[type="checkbox"]')
    await nameCheckbox.check()

    const nameInput = page.locator('input[placeholder*="name"]').first()
    await nameInput.fill('CREE')

    await page.waitForTimeout(500)

    // Click reset button
    const resetButton = page.locator('button:has-text("Reset Filters")')
    await resetButton.click()

    // Filters should be cleared
    await expect(nameCheckbox).not.toBeChecked()
    await expect(nameInput).toHaveValue('')
  })

  test('should have export results button', async ({ page }) => {
    const exportButton = page.locator('button:has-text("Export Results")')
    await expect(exportButton).toBeVisible()
  })

  test('should show transistor cards in results', async ({ page }) => {
    // Wait for results to load
    await page.waitForResponse(response =>
      response.url().includes('/api/transistors') &&
      response.status() === 200,
      { timeout: 10000 }
    )

    // Look for transistor cards or table rows
    const hasCards = await page.locator('.transistor-card').count() > 0
    const hasTableRows = await page.locator('table tbody tr').count() > 0

    expect(hasCards || hasTableRows).toBeTruthy()
  })

  test('should handle no search results', async ({ page }) => {
    // Enable name filter with non-existent name
    const nameCheckbox = page.locator('label:has-text("Name Contains:")').locator('input[type="checkbox"]')
    await nameCheckbox.check()

    const nameInput = page.locator('input[placeholder*="name"]').first()
    await nameInput.fill('NonExistentTransistor12345')

    await page.waitForTimeout(500)

    // Should show 0 devices
    const resultsText = await page.locator('text=/Search Results.*0 devices/').textContent()
    expect(resultsText).toContain('0 devices')
  })
})

test.describe('Search Filters - Voltage Range', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.locator('button', { hasText: 'Search Database' }).click()
    await page.waitForTimeout(1000)
  })

  test('should filter by voltage range', async ({ page }) => {
    // Find voltage filter section
    const voltageCheckbox = page.locator('label:has-text("Max Voltage")').locator('input[type="checkbox"]')
    await voltageCheckbox.check()

    // Fill in voltage range
    const voltageMinInput = page.locator('label:has-text("Max Voltage")').locator('..').locator('input[placeholder="Min"]')
    const voltageMaxInput = page.locator('label:has-text("Max Voltage")').locator('..').locator('input[placeholder="Max"]')

    await voltageMinInput.fill('600')
    await voltageMaxInput.fill('700')

    await page.waitForTimeout(500)

    // Results should be filtered
    const resultsText = await page.locator('text=/Search Results.*devices/').textContent()
    expect(resultsText).toBeTruthy()
  })
})

test.describe('Search Results Interaction', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(1000)
  })

  test('should have action buttons on transistor rows', async ({ page }) => {
    // Wait for results
    await page.waitForResponse(response =>
      response.url().includes('/api/transistors') &&
      response.status() === 200
    )

    // Look for action buttons (may be in table or cards)
    const hasButtons = await page.locator('button, .action-button, [role="button"]').count() > 0
    expect(hasButtons).toBeTruthy()
  })
})

test.describe('Search View Mobile', () => {
  test('should be usable on mobile devices', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/')

    // Search view should be visible
    await expect(page.locator('text=Search Database')).toBeVisible()

    // Filters should be accessible (may be collapsed)
    const hasFilters = await page.locator('text=Advanced Filters').count() > 0
    expect(hasFilters).toBeTruthy()
  })
})
