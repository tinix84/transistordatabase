<template>
  <div class="curve-list">
    <div class="list-header">
      <h4>{{ title }}</h4>
      <button @click="$emit('add')" class="btn btn-primary">
        ➕ Add {{ curveType }}
      </button>
    </div>

    <div v-if="curves.length === 0" class="empty-state">
      <p>📊 No {{ curveType }} curves yet</p>
      <p class="help-text">Click "Add {{ curveType }}" to create a new curve</p>
    </div>

    <div v-else class="curves-grid">
      <div
        v-for="(curve, index) in curves"
        :key="index"
        class="curve-card"
        @click="selectCurve(index)"
        :class="{ selected: selectedIndex === index }"
      >
        <div class="card-header">
          <div class="curve-id">
            <strong>Curve #{{ index + 1 }}</strong>
          </div>
          <div class="card-actions">
            <button
              @click.stop="$emit('view', index)"
              class="btn-icon"
              title="View Curve"
            >
              👁️
            </button>
            <button
              @click.stop="$emit('edit', index)"
              class="btn-icon"
              title="Edit"
            >
              ✏️
            </button>
            <button
              @click.stop="confirmDelete(index)"
              class="btn-icon btn-danger"
              title="Delete"
            >
              🗑️
            </button>
          </div>
        </div>

        <div class="card-content">
          <div class="curve-info">
            <div v-for="(value, key) in getCurveInfo(curve)" :key="key" class="info-row">
              <span class="info-label">{{ formatLabel(key) }}:</span>
              <span class="info-value">{{ formatValue(key, value) }}</span>
            </div>
          </div>

          <div v-if="showPreview && hasGraphData(curve)" class="curve-preview">
            <div class="preview-mini-chart">
              📈 {{ getDataPointCount(curve) }} points
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Delete Confirmation Modal -->
    <div v-if="deleteConfirmIndex !== null" class="modal-overlay" @click="cancelDelete">
      <div class="modal-content" @click.stop>
        <h3>⚠️ Confirm Delete</h3>
        <p>
          Are you sure you want to delete <strong>Curve #{{ deleteConfirmIndex + 1 }}</strong>?
        </p>
        <p class="warning-text">This action cannot be undone.</p>
        <div class="modal-actions">
          <button @click="cancelDelete" class="btn btn-secondary">
            Cancel
          </button>
          <button @click="executeDelete" class="btn btn-danger">
            Delete
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CurveList',
  props: {
    curves: {
      type: Array,
      required: true
    },
    title: {
      type: String,
      required: true
    },
    curveType: {
      type: String,
      required: true
    },
    showPreview: {
      type: Boolean,
      default: true
    }
  },
  emits: ['add', 'edit', 'view', 'delete', 'select'],
  data() {
    return {
      selectedIndex: null,
      deleteConfirmIndex: null
    }
  },
  methods: {
    selectCurve(index) {
      this.selectedIndex = this.selectedIndex === index ? null : index
      this.$emit('select', this.selectedIndex !== null ? index : null)
    },

    confirmDelete(index) {
      this.deleteConfirmIndex = index
    },

    cancelDelete() {
      this.deleteConfirmIndex = null
    },

    executeDelete() {
      if (this.deleteConfirmIndex !== null) {
        this.$emit('delete', this.deleteConfirmIndex)
        this.deleteConfirmIndex = null
        if (this.selectedIndex === this.deleteConfirmIndex) {
          this.selectedIndex = null
        }
      }
    },

    getCurveInfo(curve) {
      const info = {}

      // Extract relevant fields (exclude graph data)
      for (const [key, value] of Object.entries(curve)) {
        if (!key.startsWith('graph_') && value !== null && value !== undefined) {
          info[key] = value
        }
      }

      return info
    },

    formatLabel(key) {
      const labelMap = {
        't_j': 'T_j',
        'v_g': 'V_g',
        'v_supply': 'V_supply',
        'i_channel': 'I_channel',
        'r_g': 'R_g',
        'v_g_off': 'V_g,off',
        't_c': 'T_c',
        'time_pulse': 'Pulse Time',
        'dataset_type': 'Type',
        'i_g': 'I_g'
      }
      return labelMap[key] || key
    },

    formatValue(key, value) {
      if (typeof value === 'number') {
        if (key === 't_j' || key === 't_c') {
          return `${value}°C`
        } else if (key.startsWith('v_')) {
          return `${value}V`
        } else if (key.startsWith('i_')) {
          return `${value}A`
        } else if (key.startsWith('r_')) {
          return `${value}Ω`
        } else if (key === 'time_pulse') {
          return `${value}s`
        }
        return value.toFixed(2)
      }
      return value
    },

    hasGraphData(curve) {
      return Object.keys(curve).some(key => key.startsWith('graph_'))
    },

    getDataPointCount(curve) {
      for (const [key, value] of Object.entries(curve)) {
        if (key.startsWith('graph_') && value && Array.isArray(value)) {
          return Array.isArray(value[0]) ? value[0].length : value.length
        }
      }
      return 0
    }
  }
}
</script>

<style scoped>
.curve-list {
  padding: 16px;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.list-header h4 {
  margin: 0;
  font-size: 18px;
  color: #333;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-primary {
  background: #3b82f6;
  color: white;
}

.btn-primary:hover {
  background: #2563eb;
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}

.btn-danger {
  background: #ef4444 !important;
  color: white !important;
}

.btn-danger:hover {
  background: #dc2626 !important;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #9ca3af;
}

.empty-state p {
  margin: 8px 0;
  font-size: 16px;
}

.help-text {
  font-size: 14px;
  color: #9ca3af;
}

.curves-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}

.curve-card {
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  padding: 16px;
  background: white;
  cursor: pointer;
  transition: all 0.2s;
}

.curve-card:hover {
  border-color: #3b82f6;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.curve-card.selected {
  border-color: #3b82f6;
  background: #eff6ff;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid #e5e7eb;
}

.curve-id strong {
  color: #1f2937;
  font-size: 14px;
}

.card-actions {
  display: flex;
  gap: 6px;
}

.btn-icon {
  background: none;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  padding: 4px 8px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.btn-icon:hover {
  background: #f3f4f6;
  border-color: #9ca3af;
}

.card-content {
  font-size: 13px;
}

.curve-info {
  display: grid;
  gap: 6px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.info-label {
  color: #6b7280;
  font-weight: 500;
}

.info-value {
  color: #1f2937;
  font-family: 'Courier New', monospace;
}

.curve-preview {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #e5e7eb;
}

.preview-mini-chart {
  text-align: center;
  font-size: 12px;
  color: #6b7280;
}

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 12px;
  padding: 24px;
  max-width: 400px;
  width: 90%;
  box-shadow: 0 20px 25px rgba(0, 0, 0, 0.3);
}

.modal-content h3 {
  margin-top: 0;
  color: #1f2937;
}

.warning-text {
  color: #ef4444;
  font-size: 14px;
  margin-top: 8px;
}

.modal-actions {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  margin-top: 24px;
}
</style>
