import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

const { downloadDatabaseBackup, validateDatabaseBackup, restoreDatabaseBackup } = vi.hoisted(() => ({
  downloadDatabaseBackup: vi.fn(),
  validateDatabaseBackup: vi.fn(),
  restoreDatabaseBackup: vi.fn()
}))

vi.mock('../../services/api.js', () => ({
  transistorApi: { downloadDatabaseBackup, validateDatabaseBackup, restoreDatabaseBackup }
}))

import DatabaseBackupManager from '../../components/DatabaseBackupManager.vue'

describe('DatabaseBackupManager', () => {
  let wrapper
  let selectedFile

  beforeEach(() => {
    downloadDatabaseBackup.mockReset()
    validateDatabaseBackup.mockReset()
    restoreDatabaseBackup.mockReset()
    validateDatabaseBackup.mockResolvedValue({
      valid: true,
      format: 'transistordatabase.database-backup',
      version: 1,
      record_count: 2,
      restore_count: 1,
      restore_names: ['New Device'],
      conflicts: ['Existing Device']
    })
    restoreDatabaseBackup.mockResolvedValue({
      restored: ['New Device'],
      conflicts: ['Existing Device'],
      errors: []
    })
    selectedFile = new File(['{}'], 'backup.json', { type: 'application/json' })
    vi.stubGlobal('confirm', vi.fn(() => true))
    wrapper = mount(DatabaseBackupManager)
  })

  afterEach(() => {
    wrapper.unmount()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  async function selectBackup() {
    const input = wrapper.get('input[type="file"]')
    Object.defineProperty(input.element, 'files', { configurable: true, value: [selectedFile] })
    await input.trigger('change')
    await wrapper.get('button.btn-secondary').trigger('click')
    await flushPromises()
  }

  it('shows validation format and conflicts, then requires confirmation before restore', async () => {
    await selectBackup()

    expect(validateDatabaseBackup).toHaveBeenCalledWith(selectedFile)
    expect(wrapper.text()).toContain('Format: transistordatabase.database-backup, version 1.')
    expect(wrapper.text()).toContain('Existing-name conflicts will be skipped: Existing Device')
    expect(wrapper.text()).toContain('New records: New Device')

    window.confirm.mockReturnValueOnce(false)
    await wrapper.get('.backup-validation button').trigger('click')
    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('Existing records will remain unchanged'))
    expect(restoreDatabaseBackup).not.toHaveBeenCalled()

    await wrapper.get('.backup-validation button').trigger('click')
    await flushPromises()
    expect(restoreDatabaseBackup).toHaveBeenCalledWith(selectedFile)
    expect(wrapper.text()).toContain('1 record added.')
    expect(wrapper.text()).toContain('Skipped existing names: Existing Device')
    expect(wrapper.emitted('restored')).toHaveLength(1)
  })

  it('shows validation errors and does not offer restore for an invalid file', async () => {
    validateDatabaseBackup.mockRejectedValueOnce({
      response: {
        data: {
          detail: {
            message: 'Backup validation failed; no records can be restored.',
            errors: [{ index: 0, name: 'Unsafe/Device', error: 'Path separators are not allowed.' }]
          }
        }
      }
    })

    await selectBackup()

    expect(wrapper.get('[role="alert"]').text()).toContain('Unsafe/Device')
    expect(wrapper.find('.backup-validation button').exists()).toBe(false)
    expect(restoreDatabaseBackup).not.toHaveBeenCalled()
  })

  it('downloads the server-generated database backup', async () => {
    const blob = new Blob(['backup'])
    const createObjectURL = vi.fn(() => 'blob:database-backup')
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL: vi.fn() })
    downloadDatabaseBackup.mockResolvedValue(blob)
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})

    await wrapper.get('.backup-action .btn-primary').trigger('click')
    await flushPromises()

    expect(downloadDatabaseBackup).toHaveBeenCalledOnce()
    expect(createObjectURL).toHaveBeenCalledWith(blob)
    expect(wrapper.text()).toContain('Database backup downloaded.')
  })

  it('shows corrupt database record details when backup download is rejected', async () => {
    downloadDatabaseBackup.mockRejectedValueOnce({
      response: {
        data: new Blob([JSON.stringify({
          detail: {
            message: 'Database backup was not created because records are invalid.',
            records: [{ file: 'corrupt-device.json', error: 'Invalid JSON.' }]
          }
        })], { type: 'application/json' })
      }
    })

    await wrapper.get('.backup-action .btn-primary').trigger('click')
    await flushPromises()

    expect(wrapper.get('[role="alert"]').text()).toContain('corrupt-device.json: Invalid JSON.')
  })
})
