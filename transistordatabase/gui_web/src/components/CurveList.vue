<template>
  <div class="curve-list">
    <div class="list-header">
      <h4>{{ title }}</h4>
      <button @click="$emit('add')" class="btn btn-primary">
        Add {{ curveType }}
      </button>
    </div>

    <div v-if="curves.length === 0" class="empty-state">
      <p>No {{ curveType }} curves yet</p>
    </div>

    <div v-else class="curves-grid">
      <div
        v-for="(curve, index) in curves"
        :key="index"
        class="curve-card"
        @click="selectCurve(index)"
        :class="{ selected: selectedIndex === index }"
      >
        <div class="curve-id">
          <strong>Curve #{{ index + 1 }}</strong>
        </div>
        <div class="card-content">
          <div class="curve-info">
            <div v-for="(value, key) in getCurveInfo(curve)" :key="key" class="info-row">
              <span class="info-label">{{ formatLabel(key) }}:</span>
              <span class="info-value">{{ formatValue(key, value) }}</span>
            </div>
          </div>

          <span v-if="showPreview && hasGraphData(curve)" class="curve-preview">{{ getDataPointCount(curve) }} points</span>
        </div>
        <div class="card-actions">
          <button
            @click.stop="$emit('edit', index)"
            class="btn-icon"
            title="Edit"
            aria-label="Edit Curve"
          >
            Edit
          </button>
          <button
            @click.stop="$emit('view', index)"
            class="btn-icon"
            title="View Curve"
          >
            View
          </button>
          <button
            @click.stop="confirmDelete(index)"
            class="btn-icon btn-danger"
            title="Delete"
          >
            Delete
          </button>
        </div>
      </div>
    </div>

    <!-- Delete Confirmation Modal -->
    <div v-if="deleteConfirmIndex !== null" class="modal-overlay" @click="cancelDelete">
      <div class="modal-content" role="dialog" aria-modal="true" aria-labelledby="curve-delete-title" @click.stop>
        <h3 id="curve-delete-title">Confirm Delete</h3>
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
        if (!key.startsWith('graph_') && !key.startsWith('_') && value !== null && value !== undefined) {
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
  border: 1px solid var(--border-color);
  border-radius: 4px;
}

.list-header, .curve-card {
  display: flex;
  align-items: center;
  gap: .6rem;
}

.list-header {
  justify-content: space-between;
  flex-wrap: wrap;
  padding: .45rem .65rem;
  background: var(--bg-tertiary);
}

.list-header h4 { font-size: .8rem; }
.curve-card, .empty-state { padding: .45rem .65rem; border-top: 1px solid var(--border-color); }
.empty-state { color: var(--text-secondary); }

.curve-card {
  flex-wrap: wrap;
  justify-content: space-between;
  cursor: pointer;
}

.curve-card:hover, .curve-card.selected { background: var(--accent-blue-light); }
.curve-id { min-width: 4.5rem; }
.card-content { flex: 1; min-width: 0; display: flex; flex-wrap: wrap; gap: .35rem 1rem; }
.curve-info, .card-actions { display: flex; flex-wrap: wrap; gap: .35rem .65rem; }
.info-row { display: flex; align-items: baseline; gap: .22rem; }
.info-label, .curve-preview { color: var(--text-secondary); }
.info-value { font-family: ui-monospace, monospace; overflow-wrap: anywhere; }

.btn, .btn-icon { padding: .24rem .55rem; font: inherit; }
.btn-primary { background: var(--action-bg); border-color: var(--action-bg); }
.btn-primary:hover:not(:disabled) { background: var(--action-hover); }
.btn-icon {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
}
.btn-icon:hover { border-color: var(--accent-blue); }
.btn-icon.btn-danger { color: var(--danger-color); }
.btn-icon.btn-danger:hover { border-color: var(--danger-color); }

.modal-overlay {
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  padding: 1rem;
  background: rgb(15 23 42 / 48%);
  z-index: 600;
}

.modal-content {
  padding: .9rem;
  max-width: 26rem;
  border: 1px solid var(--border-color);
  border-radius: 5px;
  background: var(--bg-secondary);
}

.warning-text { margin-top: .35rem; color: var(--text-secondary); }
.modal-actions { display: flex; justify-content: flex-end; gap: .45rem; margin-top: .65rem; }

@media (max-width: 680px) {
  .card-content { flex-basis: 100%; }
}
</style>
