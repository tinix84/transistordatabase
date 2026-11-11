// Inspect the mounted production chart without requiring Vue development hooks.
export function inspectRenderedChart(node, selectedTime = null) {
  function find(vnode) {
    if (!vnode) return null
    if (vnode.component) {
      const component = vnode.component
      if (component.subTree?.el === node && component.proxy?.chart) return component
      const child = find(component.subTree)
      if (child) return child
    }
    if (Array.isArray(vnode.children)) for (const child of vnode.children) {
      const component = find(child)
      if (component) return component
    }
    return null
  }
  const component = find(document.getElementById('app')._vnode)
  if (!component) throw new Error('The mounted plot has no Chart instance.')
  const chart = component.proxy.chart
  const cursorTime = selectedTime && typeof selectedTime === 'object' ? selectedTime.cursorTime ?? null : selectedTime
  const probes = selectedTime && typeof selectedTime === 'object' ? selectedTime.probes ?? [] : []
  let purplePixels = 0
  if (cursorTime !== null) {
    const scale = chart.currentDevicePixelRatio, x = chart.scales.x.getPixelForValue(cursorTime)
    const left = Math.max(0, Math.round(x * scale) - 2), top = Math.ceil(chart.chartArea.top * scale)
    const height = Math.floor(chart.chartArea.bottom * scale) - top
    const pixels = chart.ctx.getImageData(left, top, 5, height).data
    for (let i = 0; i < pixels.length; i += 4) if (Math.abs(pixels[i] - 124) < 25 && Math.abs(pixels[i + 1] - 58) < 25 && Math.abs(pixels[i + 2] - 237) < 25 && pixels[i + 3] > 100) purplePixels++
  }
  return {
    chartArea: { left: chart.chartArea.left, right: chart.chartArea.right, top: chart.chartArea.top, bottom: chart.chartArea.bottom },
    cursor: component.proxy.cursorX,
    xLabel: chart.options.scales.x.title.text,
    xMin: chart.scales.x.min, xMax: chart.scales.x.max,
    yMin: chart.scales.y.min, yMax: chart.scales.y.max,
    drawnTime: cursorTime === null ? null : chart.scales.x.getValueForPixel(chart.scales.x.getPixelForValue(cursorTime)),
    purplePixels,
    probes: probes.map(probe => {
      const scale = chart.currentDevicePixelRatio
      const x = Math.round(chart.scales.x.getPixelForValue(probe.time) * scale), y = Math.round(chart.scales.y.getPixelForValue(probe.value) * scale)
      const pixels = chart.ctx.getImageData(x - 3, y - 3, 7, 7).data
      let linePixels = 0
      for (let i = 0; i < pixels.length; i += 4) if (Math.abs(pixels[i] - 75) < 25 && Math.abs(pixels[i + 1] - 192) < 25 && Math.abs(pixels[i + 2] - 192) < 25 && pixels[i + 3] > 100) linePixels++
      return { ...probe, linePixels }
    }),
    datasets: chart.data.datasets.map((dataset, index) => ({ label: dataset.label, showLine: dataset.showLine,
      points: chart.getDatasetMeta(index).data.map((point, i) => ({ input: dataset.data[i], value: chart.scales.y.getValueForPixel(point.y), pixel: { x: point.x, y: point.y }, radius: point.options.radius, borderWidth: point.options.borderWidth })) })),
  }
}
