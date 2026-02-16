<template>
  <div class="breakdown-chart">
    <div class="chart-container">
      <canvas ref="chartCanvas"></canvas>
    </div>

    <div class="breakdown-table">
      <table>
        <thead>
          <tr>
            <th>Transistor</th>
            <th>Conduction [W]</th>
            <th>E_on [W]</th>
            <th>E_off [W]</th>
            <th>Total [W]</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(item, index) in data" :key="index">
            <td><strong>{{ item.name }}</strong></td>
            <td>{{ item.conduction.toFixed(2) }}</td>
            <td>{{ item.switching_on.toFixed(2) }}</td>
            <td>{{ item.switching_off.toFixed(2) }}</td>
            <td><strong>{{ item.total.toFixed(2) }}</strong></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, watch } from 'vue'
import { Chart, registerables } from 'chart.js/auto'

Chart.register(...registerables)

export default {
  name: 'BreakdownChart',
  props: {
    data: {
      type: Array,
      required: true
      // Format: [{ name, conduction, switching_on, switching_off, total }]
    }
  },
  setup(props) {
    const chartCanvas = ref(null)
    let chart = null

    const createChart = () => {
      if (!chartCanvas.value || props.data.length === 0) return

      const ctx = chartCanvas.value.getContext('2d')

      const labels = props.data.map(item => item.name)
      const conductionData = props.data.map(item => item.conduction)
      const switchingOnData = props.data.map(item => item.switching_on)
      const switchingOffData = props.data.map(item => item.switching_off)

      if (chart) {
        chart.destroy()
      }

      chart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Conduction Loss',
              data: conductionData,
              backgroundColor: 'rgba(59, 130, 246, 0.7)',
              borderColor: 'rgb(59, 130, 246)',
              borderWidth: 1
            },
            {
              label: 'Switching Loss (E_on)',
              data: switchingOnData,
              backgroundColor: 'rgba(239, 68, 68, 0.7)',
              borderColor: 'rgb(239, 68, 68)',
              borderWidth: 1
            },
            {
              label: 'Switching Loss (E_off)',
              data: switchingOffData,
              backgroundColor: 'rgba(251, 146, 60, 0.7)',
              borderColor: 'rgb(251, 146, 60)',
              borderWidth: 1
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'top',
            },
            tooltip: {
              callbacks: {
                footer: (tooltipItems) => {
                  const index = tooltipItems[0].dataIndex
                  const total = props.data[index].total
                  return `Total: ${total.toFixed(2)} W`
                }
              }
            }
          },
          scales: {
            x: {
              stacked: true,
              title: {
                display: true,
                text: 'Transistor'
              }
            },
            y: {
              stacked: true,
              title: {
                display: true,
                text: 'Power Loss [W]'
              },
              beginAtZero: true
            }
          }
        }
      })
    }

    onMounted(() => {
      createChart()
    })

    watch(() => props.data, () => {
      createChart()
    }, { deep: true })

    return {
      chartCanvas
    }
  }
}
</script>

<style scoped>
.breakdown-chart {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.chart-container {
  position: relative;
  height: 300px;
  width: 100%;
}

.breakdown-table {
  overflow-x: auto;
}

.breakdown-table table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.breakdown-table th {
  background: #f3f4f6;
  padding: 12px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #d1d5db;
  color: #374151;
}

.breakdown-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #e5e7eb;
  color: #1f2937;
}

.breakdown-table td:not(:first-child) {
  text-align: right;
  font-family: 'Courier New', monospace;
}

.breakdown-table tbody tr:hover {
  background: #f9fafb;
}
</style>
