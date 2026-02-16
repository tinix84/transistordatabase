<template>
  <div class="plot-card">
    <div class="plot-header">
      <h3>{{ title }}</h3>
      <div class="plot-actions">
        <button
          v-if="!isFullscreen"
          @click="toggleDataTable"
          class="btn-icon"
          title="Toggle Data Table"
        >
          📊
        </button>
        <button @click="downloadPlot('png')" class="btn-icon" title="Download PNG">
          💾
        </button>
        <button
          v-if="allowFullscreen"
          @click="toggleFullscreen"
          class="btn-icon"
          title="Fullscreen"
        >
          {{ isFullscreen ? '↙️' : '⤢' }}
        </button>
      </div>
    </div>

    <div class="plot-content" :class="{ fullscreen: isFullscreen }">
      <CurvePlotter
        v-if="plotData && plotData.curves && plotData.curves.length > 0"
        :ref="plotterRef"
        :title="title"
        :datasets="chartDatasets"
        :x-label="plotData.xlabel || 'X'"
        :y-label="plotData.ylabel || 'Y'"
        :x-log-scale="plotData.log_scale_x || false"
        :y-log-scale="plotData.log_scale_y || false"
        :show-legend="true"
      />
      <div v-else-if="plotData && plotData.breakdown" class="breakdown-chart">
        <BreakdownChart :data="plotData.breakdown" />
      </div>
      <div v-else class="no-data">
        <p>📭 No data available for this comparison</p>
        <p class="help-text">
          Select transistors with {{ dataRequirement }} data
        </p>
      </div>

      <!-- Data Table View -->
      <div v-if="showDataTable && plotData" class="data-table-panel">
        <div class="table-header">
          <h4>📋 Data Table</h4>
          <button @click="showDataTable = false" class="btn-close">✕</button>
        </div>
        <div class="table-wrapper">
          <table class="data-table">
            <thead>
              <tr>
                <th>{{ plotData.xlabel }}</th>
                <th v-for="(curve, idx) in plotData.curves" :key="idx">
                  {{ curve.name }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, rowIdx) in tableData" :key="rowIdx">
                <td>{{ row.x }}</td>
                <td v-for="(val, colIdx) in row.y" :key="colIdx">
                  {{ val }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <button @click="downloadData('csv')" class="btn btn-sm btn-secondary">
          💾 Download CSV
        </button>
      </div>
    </div>

    <div v-if="plotData && plotData.config" class="plot-footer">
      <div class="config-badges">
        <span class="badge">T_j: {{ plotData.config.t_j }}°C</span>
        <span v-if="plotData.config.v_supply" class="badge">
          V_supply: {{ plotData.config.v_supply.join(', ') }}V
        </span>
        <span v-if="plotData.config.r_g_on" class="badge">
          R_g,on: {{ plotData.config.r_g_on.join(', ') }}Ω
        </span>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import CurvePlotter from './CurvePlotter.vue'
import BreakdownChart from './BreakdownChart.vue'

export default {
  name: 'PlotCard',
  components: {
    CurvePlotter,
    BreakdownChart
  },
  props: {
    title: {
      type: String,
      required: true
    },
    plotData: {
      type: Object,
      default: null
    },
    dataRequirement: {
      type: String,
      default: 'characteristic'
    },
    allowFullscreen: {
      type: Boolean,
      default: true
    }
  },
  setup(props) {
    const isFullscreen = ref(false)
    const showDataTable = ref(false)
    const plotterRef = ref(null)

    const chartDatasets = computed(() => {
      if (!props.plotData || !props.plotData.curves) return []

      const colors = [
        'rgb(59, 130, 246)',   // Blue
        'rgb(239, 68, 68)',    // Red
        'rgb(34, 197, 94)',    // Green
        'rgb(251, 146, 60)',   // Orange
        'rgb(168, 85, 247)',   // Purple
      ]

      return props.plotData.curves.map((curve, idx) => {
        let data = []

        // Handle different data formats
        if (curve.v_data && curve.i_data) {
          data = curve.v_data.map((v, i) => ({
            x: v,
            y: curve.i_data[i]
          }))
        } else if (curve.i_data && curve.e_data) {
          data = curve.i_data.map((i, index) => ({
            x: i,
            y: curve.e_data[index]
          }))
        } else if (curve.q_data && curve.v_data) {
          data = curve.q_data.map((q, i) => ({
            x: q,
            y: curve.v_data[i]
          }))
        } else if (curve.time_data && curve.z_th_data) {
          data = curve.time_data.map((t, i) => ({
            x: t,
            y: curve.z_th_data[i]
          }))
        } else if (curve.current_data && curve.efficiency_data) {
          data = curve.current_data.map((i, index) => ({
            x: i,
            y: curve.efficiency_data[index]
          }))
        } else if (curve.c_data && curve.v_data) {
          data = curve.v_data.map((v, i) => ({
            x: v,
            y: curve.c_data[i]
          }))
        }

        return {
          label: curve.name,
          data: data,
          borderColor: colors[idx % colors.length],
          backgroundColor: colors[idx % colors.length].replace('rgb', 'rgba').replace(')', ', 0.1)'),
          borderWidth: 2,
          pointRadius: 2
        }
      })
    })

    const tableData = computed(() => {
      if (!props.plotData || !props.plotData.curves || props.plotData.curves.length === 0) {
        return []
      }

      const firstCurve = props.plotData.curves[0]
      let xData = []

      if (firstCurve.v_data) xData = firstCurve.v_data
      else if (firstCurve.i_data) xData = firstCurve.i_data
      else if (firstCurve.q_data) xData = firstCurve.q_data
      else if (firstCurve.time_data) xData = firstCurve.time_data
      else if (firstCurve.current_data) xData = firstCurve.current_data

      return xData.slice(0, 20).map((x, idx) => ({
        x: typeof x === 'number' ? x.toExponential(3) : x,
        y: props.plotData.curves.map(curve => {
          let yData = curve.i_data || curve.e_data || curve.v_data || curve.z_th_data || curve.efficiency_data || curve.c_data
          return yData && yData[idx] ? (typeof yData[idx] === 'number' ? yData[idx].toExponential(3) : yData[idx]) : 'N/A'
        })
      }))
    })

    const toggleFullscreen = () => {
      isFullscreen.value = !isFullscreen.value
    }

    const toggleDataTable = () => {
      showDataTable.value = !showDataTable.value
    }

    const downloadPlot = (format) => {
      // Trigger download from CurvePlotter
      console.log(`Downloading plot as ${format}`)
    }

    const downloadData = (format) => {
      if (!props.plotData || !props.plotData.curves) return

      let csvContent = ''

      // Header
      const headers = [props.plotData.xlabel || 'X']
      props.plotData.curves.forEach(curve => headers.push(curve.name))
      csvContent += headers.join(',') + '\n'

      // Data rows
      const firstCurve = props.plotData.curves[0]
      const xData = firstCurve.v_data || firstCurve.i_data || firstCurve.q_data || firstCurve.time_data || firstCurve.current_data || []

      xData.forEach((x, idx) => {
        const row = [x]
        props.plotData.curves.forEach(curve => {
          const yData = curve.i_data || curve.e_data || curve.v_data || curve.z_th_data || curve.efficiency_data || curve.c_data || []
          row.push(yData[idx] || '')
        })
        csvContent += row.join(',') + '\n'
      })

      // Download
      const blob = new Blob([csvContent], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${props.title.replace(/\s+/g, '_')}.csv`
      link.click()
      URL.revokeObjectURL(url)
    }

    return {
      isFullscreen,
      showDataTable,
      plotterRef,
      chartDatasets,
      tableData,
      toggleFullscreen,
      toggleDataTable,
      downloadPlot,
      downloadData
    }
  }
}
</script>

<style scoped>
.plot-card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  margin-bottom: 24px;
}

.plot-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.plot-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}

.plot-actions {
  display: flex;
  gap: 8px;
}

.btn-icon {
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 6px;
  padding: 6px 12px;
  cursor: pointer;
  font-size: 16px;
  color: white;
  transition: all 0.2s;
}

.btn-icon:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: scale(1.05);
}

.plot-content {
  padding: 20px;
  min-height: 400px;
  position: relative;
}

.plot-content.fullscreen {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 1000;
  background: white;
  padding: 40px;
  overflow: auto;
}

.no-data {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  color: #9ca3af;
}

.no-data p {
  margin: 8px 0;
  font-size: 16px;
}

.help-text {
  font-size: 14px;
  color: #6b7280;
}

.breakdown-chart {
  padding: 20px;
}

.data-table-panel {
  margin-top: 24px;
  border-top: 2px solid #e5e7eb;
  padding-top: 20px;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.table-header h4 {
  margin: 0;
  color: #374151;
}

.btn-close {
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: #6b7280;
  padding: 4px;
}

.btn-close:hover {
  color: #1f2937;
}

.table-wrapper {
  overflow-x: auto;
  margin-bottom: 16px;
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.data-table th {
  position: sticky;
  top: 0;
  background: #f3f4f6;
  padding: 10px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #d1d5db;
  white-space: nowrap;
}

.data-table td {
  padding: 8px 10px;
  border-bottom: 1px solid #e5e7eb;
  font-family: 'Courier New', monospace;
  font-size: 12px;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-sm {
  padding: 6px 12px;
  font-size: 13px;
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}

.plot-footer {
  padding: 12px 20px;
  background: #f9fafb;
  border-top: 1px solid #e5e7eb;
}

.config-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.badge {
  display: inline-block;
  padding: 4px 10px;
  background: #e0e7ff;
  color: #3730a3;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}
</style>
