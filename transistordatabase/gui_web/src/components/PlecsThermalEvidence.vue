<script setup>
defineProps({ networks: { type: Array, required: true }, scope: { type: String, required: true } })
const display = value => value === null || value === undefined ? 'Unknown' : String(value)
</script>
<template>
  <section v-if="networks.length" class="thermal-evidence">
    <h4>{{ scope }} thermal networks</h4>
    <section v-for="(network, index) in networks" :key="index">
      <p><strong>{{ display(network.network_type) }}</strong> · {{ network.status }}</p>
      <p v-if="network.network_type?.toLowerCase() === 'cauer'">Cauer chain retained as source. Its elements are not interpreted as independent Foster poles.</p>
      <div class="table-scroll"><table><caption>{{ network.source_path }}</caption><thead><tr><th>Element</th><th>R (K/W)</th><th>Tau (s)</th><th>C (J/K)</th></tr></thead><tbody><tr v-for="(element, i) in network.elements" :key="i"><td>{{ i + 1 }}</td><td>{{ display(element.R) }}</td><td>{{ display(element.Tau) }}</td><td>{{ display(element.C) }}</td></tr></tbody></table></div>
      <details><summary>Thermal source XML</summary><pre>{{ network.xml }}</pre></details>
    </section>
  </section>
</template>
<style scoped>
.thermal-evidence { padding: .8rem 0; border-top: 1px solid #d4dbe5; }
.table-scroll { overflow-x: auto; max-width: 100%; margin: .7rem 0; }
table { border-collapse: collapse; min-width: 500px; width: 100%; }
th, td { padding: .6rem; border: 1px solid #c6cfdb; text-align: left; }
caption { text-align: left; margin-bottom: .4rem; }
details { padding: .6rem 0; } summary { cursor: pointer; font-weight: 600; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
</style>
