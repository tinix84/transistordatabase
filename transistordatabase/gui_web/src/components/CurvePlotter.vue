<template>
  <div class="curve-plotter">
    <div class="plotter-header">
      <h4>{{ title }}</h4>
      <div class="plotter-actions">
        <button @click="toggleZoom" class="btn-icon" title="Toggle Zoom">
          🔍
        </button>
        <button @click="resetZoom" class="btn-icon" title="Reset Zoom">
          🔄
        </button>
        <button @click="downloadImage" class="btn-icon" title="Download PNG">
          💾
        </button>
      </div>
    </div>

    <div class="chart-container">
      <canvas ref="chartCanvas"></canvas>
    </div>

    <div v-if="showLegend && datasets.length > 1" class="chart-legend">
      <div
        v-for="(dataset, index) in datasets"
        :key="index"
        class="legend-item"
      >
        <span
          class="legend-color"
          :style="{ backgroundColor: dataset.borderColor }"
        ></span>
        <span class="legend-label">{{ dataset.label }}</span>
      </div>
    </div>
  </div>
</template>

<script>
import { Chart, registerables } from 'chart.js/auto'
import zoomPlugin from 'chartjs-plugin-zoom'

Chart.register(...registerables, zoomPlugin)

export default {
  name: 'CurvePlotter',
  props: {
    title: {
      type: String,
      default: 'Plot'
    },
    datasets: {
      type: Array,
      required: true,
      // Each dataset: { label, data: [{x, y}], borderColor, backgroundColor }
    },
    xLabel: {
      type: String,
      default: 'X'
    },
    yLabel: {
      type: String,
      default: 'Y'
    },
    xLogScale: {
      type: Boolean,
      default: false
    },
    yLogScale: {
      type: Boolean,
      default: false
    },
    showLegend: {
      type: Boolean,
      default: true
    },
    gridLines: {
      type: Boolean,
      default: true
    }
  },
  data() {
    return {
      chart: null,
      zoomEnabled: false
    }
  },
  mounted() {
    this.createChart()
  },
  watch: {
    datasets: {
      handler() {
        this.updateChart()
      },
      deep: true
    }
  },
  beforeUnmount() {
    if (this.chart) {
      this.chart.destroy()
    }
  },
  methods: {
    createChart() {
      const ctx = this.$refs.chartCanvas.getContext('2d')

      const chartData = {
        datasets: this.datasets.map((ds, index) => ({
          label: ds.label,
          data: ds.data,
          borderColor: ds.borderColor || this.getColor(index),
          backgroundColor: ds.backgroundColor || this.getColor(index, 0.1),
          borderWidth: ds.borderWidth || 2,
          pointRadius: ds.pointRadius || 3,
          pointHoverRadius: ds.pointHoverRadius || 5,
          tension: ds.tension || 0,
          fill: ds.fill || false
        }))
      }

      this.chart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: 'nearest',
            intersect: false
          },
          plugins: {
            legend: {
              display: false // Use custom legend
            },
            tooltip: {
              enabled: true,
              mode: 'nearest',
              intersect: false,
              callbacks: {
                label: (context) => {
                  const label = context.dataset.label || ''
                  return `${label}: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`
                }
              }
            },
            zoom: {
              zoom: {
                wheel: {
                  enabled: false
                },
                pinch: {
                  enabled: false
                },
                mode: 'xy'
              },
              pan: {
                enabled: false,
                mode: 'xy'
              }
            }
          },
          scales: {
            x: {
              type: this.xLogScale ? 'logarithmic' : 'linear',
              title: {
                display: true,
                text: this.xLabel,
                font: {
                  size: 14,
                  weight: 'bold'
                }
              },
              grid: {
                display: this.gridLines,
                color: 'rgba(0, 0, 0, 0.1)'
              }
            },
            y: {
              type: this.yLogScale ? 'logarithmic' : 'linear',
              title: {
                display: true,
                text: this.yLabel,
                font: {
                  size: 14,
                  weight: 'bold'
                }
              },
              grid: {
                display: this.gridLines,
                color: 'rgba(0, 0, 0, 0.1)'
              }
            }
          }
        }
      })
    },

    updateChart() {
      if (!this.chart) return

      this.chart.data.datasets = this.datasets.map((ds, index) => ({
        label: ds.label,
        data: ds.data,
        borderColor: ds.borderColor || this.getColor(index),
        backgroundColor: ds.backgroundColor || this.getColor(index, 0.1),
        borderWidth: ds.borderWidth || 2,
        pointRadius: ds.pointRadius || 3,
        tension: ds.tension || 0,
        fill: ds.fill || false
      }))

      this.chart.update()
    },

    toggleZoom() {
      this.zoomEnabled = !this.zoomEnabled
      if (this.chart) {
        this.chart.options.plugins.zoom.zoom.wheel.enabled = this.zoomEnabled
        this.chart.options.plugins.zoom.zoom.pinch.enabled = this.zoomEnabled
        this.chart.options.plugins.zoom.pan.enabled = this.zoomEnabled
        this.chart.update()
      }
    },

    resetZoom() {
      if (this.chart) {
        this.chart.resetZoom()
      }
    },

    downloadImage() {
      if (this.chart) {
        const url = this.chart.toBase64Image()
        const link = document.createElement('a')
        link.download = `${this.title.replace(/\s+/g, '_')}.png`
        link.href = url
        link.click()
      }
    },

    getColor(index, alpha = 1) {
      const colors = [
        `rgba(75, 192, 192, ${alpha})`,
        `rgba(255, 99, 132, ${alpha})`,
        `rgba(54, 162, 235, ${alpha})`,
        `rgba(255, 206, 86, ${alpha})`,
        `rgba(153, 102, 255, ${alpha})`,
        `rgba(255, 159, 64, ${alpha})`,
        `rgba(199, 199, 199, ${alpha})`
      ]
      return colors[index % colors.length]
    }
  }
}
</script>

<style scoped>
.curve-plotter {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
  background: white;
}

.plotter-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.plotter-header h4 {
  margin: 0;
  font-size: 16px;
  color: #333;
}

.plotter-actions {
  display: flex;
  gap: 8px;
}

.btn-icon {
  background: none;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 6px 10px;
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.btn-icon:hover {
  background: #f0f0f0;
  border-color: #999;
}

.chart-container {
  position: relative;
  height: 400px;
  width: 100%;
}

.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #eee;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.legend-color {
  width: 20px;
  height: 3px;
  border-radius: 2px;
}

.legend-label {
  font-size: 13px;
  color: #666;
}
</style>
