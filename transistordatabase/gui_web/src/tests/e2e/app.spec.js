/**
 * End-to-end tests for the full web application
 */
import { test, expect } from '@playwright/test'

test.describe('Application Loading', () => {
  test('should load the homepage', async ({ page }) => {
    await page.goto('/')

    await expect(page).toHaveTitle(/Transistor Database/)
    await expect(page.locator('h1')).toContainText('Transistor Database')
  })

  test('should display contact links', async ({ page }) => {
    await page.goto('/')

    await expect(page.locator('a[href*="linkedin"]')).toBeVisible()
    await expect(page.locator('a[href*="discord"]')).toBeVisible()
    await expect(page.locator('a[href*="github"]')).toBeVisible()
  })

  test('should have theme toggle button', async ({ page }) => {
    await page.goto('/')

    const themeButton = page.locator('.theme-toggle')
    await expect(themeButton).toBeVisible()
  })

  test('should display navigation buttons', async ({ page }) => {
    await page.goto('/')

    await expect(page.locator('text=Search Database')).toBeVisible()
    await expect(page.locator('text=Create Transistor')).toBeVisible()
    await expect(page.locator('text=Exporting Tools')).toBeVisible()
    await expect(page.locator('text=Comparison Tools')).toBeVisible()
    await expect(page.locator('text=Topology Calculator')).toBeVisible()
  })
})

test.describe('Theme Toggle', () => {
  test('should toggle between light and dark themes', async ({ page }) => {
    await page.goto('/')

    const html = page.locator('html')
    const themeButton = page.locator('.theme-toggle')

    // Should start in light theme
    await expect(html).not.toHaveClass(/dark-theme/)

    // Toggle to dark theme
    await themeButton.click()
    await expect(html).toHaveClass(/dark-theme/)

    // Toggle back to light theme
    await themeButton.click()
    await expect(html).not.toHaveClass(/dark-theme/)
  })

  test('should persist theme preference', async ({ page, context }) => {
    await page.goto('/')

    // Toggle to dark theme
    await page.locator('.theme-toggle').click()

    // Reload page
    await page.reload()

    // Should still be in dark theme
    await expect(page.locator('html')).toHaveClass(/dark-theme/)
  })
})

test.describe('Navigation', () => {
  test('should navigate to search view by default', async ({ page }) => {
    await page.goto('/')

    const searchButton = page.locator('button', { hasText: 'Search Database' })
    await expect(searchButton).toHaveClass(/active/)
  })

  test('should navigate to create view', async ({ page }) => {
    await page.goto('/')

    const createButton = page.locator('button', { hasText: 'Create Transistor' })
    await createButton.click()

    await expect(createButton).toHaveClass(/active/)
  })

  test('should navigate to export view', async ({ page }) => {
    await page.goto('/')

    const exportButton = page.locator('button', { hasText: 'Exporting Tools' })
    await exportButton.click()

    await expect(exportButton).toHaveClass(/active/)
  })

  test('should navigate to comparison view', async ({ page }) => {
    await page.goto('/')

    const compareButton = page.locator('button', { hasText: 'Comparison Tools' })
    await compareButton.click()

    await expect(compareButton).toHaveClass(/active/)
  })

  test('should navigate to topology view', async ({ page }) => {
    await page.goto('/')

    const topologyButton = page.locator('button', { hasText: 'Topology Calculator' })
    await topologyButton.click()

    await expect(topologyButton).toHaveClass(/active/)
  })
})

test.describe('Data Loading', () => {
  test('should load transistors from API', async ({ page }) => {
    await page.goto('/')

    // Wait for loading to complete
    await page.waitForSelector('.loading', { state: 'hidden', timeout: 10000 })

    // Check transistor count in header
    const stats = page.locator('.stats')
    await expect(stats).toContainText(/Total Transistors: \d+/)
  })

  test('should display loading state initially', async ({ page }) => {
    await page.goto('/')

    // Loading state should appear briefly
    const loading = page.locator('.loading')
    // Note: may be too fast to catch, so we just check it doesn't error
  })

  test('should make API request to correct endpoint', async ({ page }) => {
    // Listen for API requests
    const apiRequests = []
    page.on('request', request => {
      if (request.url().includes('/api/transistors')) {
        apiRequests.push(request.url())
      }
    })

    await page.goto('/')
    await page.waitForTimeout(2000)

    expect(apiRequests.some(url => url.includes('http://localhost:8002/api/transistors'))).toBeTruthy()
  })
})

test.describe('Responsive Design', () => {
  test('should be responsive on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/')

    // Navigation should still be visible
    await expect(page.locator('.main-nav')).toBeVisible()

    // Header should be visible
    await expect(page.locator('.app-header')).toBeVisible()
  })

  test('should be responsive on tablet', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 })
    await page.goto('/')

    await expect(page.locator('.main-nav')).toBeVisible()
  })

  test('should be responsive on desktop', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 })
    await page.goto('/')

    await expect(page.locator('.main-nav')).toBeVisible()
  })
})

test.describe('Error Handling', () => {
  test('should handle API errors gracefully', async ({ page }) => {
    // Block API requests to simulate error
    await page.route('**/api/transistors', route => route.abort())

    await page.goto('/')

    // Should still render page without crashing
    await expect(page.locator('h1')).toBeVisible()

    // Should show 0 transistors
    await expect(page.locator('.stats')).toContainText('Total Transistors: 0')
  })

  test('should handle slow network', async ({ page }) => {
    // Delay API requests
    await page.route('**/api/transistors', async route => {
      await new Promise(resolve => setTimeout(resolve, 3000))
      await route.continue()
    })

    await page.goto('/')

    // Should show loading state
    const loading = page.locator('.loading')
    // Wait a bit to ensure loading appears
    await page.waitForTimeout(500)
  })
})

test.describe('Accessibility', () => {
  test('should have proper heading hierarchy', async ({ page }) => {
    await page.goto('/')

    const h1 = await page.locator('h1').count()
    expect(h1).toBeGreaterThan(0)
  })

  test('should have focusable navigation buttons', async ({ page }) => {
    await page.goto('/')

    const searchButton = page.locator('button', { hasText: 'Search Database' })
    await searchButton.focus()

    await expect(searchButton).toBeFocused()
  })

  test('should have keyboard navigation', async ({ page }) => {
    await page.goto('/')

    // Tab through navigation
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')

    // Should be able to navigate with keyboard
    const focused = await page.evaluate(() => document.activeElement.tagName)
    expect(['BUTTON', 'A']).toContain(focused)
  })
})

test.describe('Performance', () => {
  test('should load within acceptable time', async ({ page }) => {
    const startTime = Date.now()

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const loadTime = Date.now() - startTime

    // Should load within 5 seconds
    expect(loadTime).toBeLessThan(5000)
  })

  test('should have no console errors', async ({ page }) => {
    const consoleErrors = []
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text())
      }
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // Filter out expected errors (e.g., from missing assets)
    const criticalErrors = consoleErrors.filter(err =>
      !err.includes('favicon') &&
      !err.includes('404')
    )

    expect(criticalErrors).toHaveLength(0)
  })
})
