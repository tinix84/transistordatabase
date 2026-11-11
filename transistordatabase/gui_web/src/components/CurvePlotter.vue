<template>
  <div class="curve-plotter">
    <div class="plotter-header">
      <h4>{{ title }}</h4>
      <div class="plotter-actions">
        <button v-if="allowZoom" @click="toggleZoom" class="btn-icon" title="Toggle Zoom">
          🔍
        </button>
        <button v-if="allowZoom" @click="resetZoom" class="btn-icon" title="Reset Zoom">
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
import { markRaw } from 'vue'

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
    xValueDecimals: {
      type: Number,
      default: 3
    },
    xValueScientific: {
      type: Boolean,
      default: false
    },
    yValueScientific: {
      type: Boolean,
      default: false
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
    compactScientificYTicks: { type: Boolean, default: false },
    showLegend: {
      type: Boolean,
      default: true
    },
    gridLines: {
      type: Boolean,
      default: true
    },
    allowZoom: { type: Boolean, default: true },
    xMin: { type: Number, default: null },
    xMax: { type: Number, default: null
    },
    cursorX: { type: Number, default: null },
    yMin: { type: Number, default: null },
    yMax: { type: Number, default: null }
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
    xMin() { this.updateRange() },
    xMax() { this.updateRange() },
    cursorX() { this.chart?.draw() },
    yMin() { this.updateYRange() },
    yMax() { this.updateYRange() },
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
    drawCursor(chart) {
      if (!Number.isFinite(this.cursorX)) return
      const { ctx, chartArea, scales } = chart
      const x = scales.x.getPixelForValue(this.cursorX)
      if (x < chartArea.left || x > chartArea.right) return
      ctx.save()
      ctx.beginPath()
      ctx.setLineDash([4, 3])
      ctx.strokeStyle = '#7c3aed'
      ctx.lineWidth = 1.5
      ctx.moveTo(x, chartArea.top)
      ctx.lineTo(x, chartArea.bottom)
      ctx.stroke()
      ctx.restore()
    },
    formatCompactScientificTick(value) {
      const number = Number(value)
      if (number === 0) return '0'
      if (!Number.isFinite(number)) return String(value)
      const magnitude = Math.abs(number)
      return magnitude < 1e-3 || magnitude >= 1e6
        ? number.toExponential(2) : Number(number.toPrecision(3)).toString()
    },
    createChart() {
      const ctx = this.$refs.chartCanvas.getContext('2d')

      const chartData = {
        datasets: this.datasets.map((ds, index) => ({
          label: ds.label,
          data: ds.data,
          borderColor: ds.borderColor || this.getColor(index),
          backgroundColor: ds.backgroundColor || this.getColor(index, 0.1),
          borderWidth: ds.borderWidth || 2,
          pointRadius: ds.pointRadius ?? 3,
          showLine: ds.showLine ?? true,
          pointHoverRadius: ds.pointHoverRadius || 5,
          tension: ds.tension || 0,
          fill: ds.fill || false
        }))
      }

      this.chart = markRaw(new Chart(ctx, {
        type: 'line',
        data: chartData,
        plugins: [{ id: 'sharedTimeCursor', afterDatasetsDraw: chart => this.drawCursor(chart) }],
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
                  if (this.compactScientificYTicks) {
                    return `${label}: (${context.parsed.x.toPrecision(6)}, ${context.parsed.y.toPrecision(6)})`
                  }
                  const xValue = this.xValueScientific
                    ? context.parsed.x.toExponential(3)
                    : context.parsed.x.toFixed(this.xValueDecimals)
                  const needsScientificY = this.yValueScientific &&
                    context.parsed.y !== 0 && Math.abs(context.parsed.y) < 0.001
                  const yValue = needsScientificY
                    ? context.parsed.y.toExponential(3)
                    : context.parsed.y.toFixed(3)
                  return `${label}: (${xValue}, ${yValue})`
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
              min: this.xMin ?? undefined,
              max: this.xMax ?? undefined,
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
              min: this.yMin ?? undefined,
              max: this.yMax ?? undefined,
              type: this.yLogScale ? 'logarithmic' : 'linear',
              ...(this.compactScientificYTicks ? { ticks: { callback: this.formatCompactScientificTick } } : {}),
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
      }))
    },

    updateChart() {
      if (!this.chart) return

      this.chart.data.datasets = this.datasets.map((ds, index) => ({
        label: ds.label,
        data: ds.data,
        borderColor: ds.borderColor || this.getColor(index),
        backgroundColor: ds.backgroundColor || this.getColor(index, 0.1),
        borderWidth: ds.borderWidth || 2,
        pointRadius: ds.pointRadius ?? 3,
        showLine: ds.showLine ?? true,
        tension: ds.tension || 0,
        fill: ds.fill || false
      }))

      this.chart.update()
    },

    updateRange() {
      if (!this.chart) return
      // Cancel targets computed with the old scales before drawing new bounds.
      // Otherwise an initial animation can later restore stale point pixels.
      this.chart.stop()
      this.chart.options.scales.x.min = this.xMin ?? undefined
      this.chart.options.scales.x.max = this.xMax ?? undefined
      this.chart.update('none')
    },

    updateYRange() {
      if (!this.chart) return
      this.chart.stop()
      this.chart.options.scales.y.min = this.yMin ?? undefined
      this.chart.options.scales.y.max = this.yMax ?? undefined
      this.chart.update('none')
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
