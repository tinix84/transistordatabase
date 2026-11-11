<script setup>
import { computed, ref, watch } from 'vue'
import { displayNumber as number } from './analysisFiles.js'

const props = defineProps({ merit: { type: Object, required: true } })
const page = ref(0), pageSize = 32
watch(() => props.merit, () => { page.value = 0 })
const rows = computed(() => props.merit.rows.slice(page.value * pageSize, (page.value + 1) * pageSize))
const ranking = computed(() => props.merit.ranking.slice(0, pageSize))
function downloadRaw(value, filename, type) {
  const url = URL.createObjectURL(new Blob([value], { type }))
  const link = document.createElement('a'); link.href = url; link.download = filename; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
</script>

<template>
  <section class="merit-evidence" data-testid="merit-evidence">
    <h3>Weighted merit penalty comparison</h3>
    <p><strong>Lower dimensionless penalty is better for the selected policy.</strong> Zero is the best possible penalty. The penalty depends on the objectives, weights, directions, and eligible population.</p>
    <p>For minimize, the objective penalty is (mean − minimum) / (maximum − minimum). Maximize reverses this scale. A constant objective has penalty zero.</p>
    <p>Python normalizes positive objective weights and sums their weighted penalties. Zero-weight objectives are inactive. Only complete full means enter the common population.</p>
    <p>Status: <strong>{{ merit.status }}</strong> · {{ merit.method_version }}. Pair filters and table pages do not change this comparison or its downloads.</p>
    <p>Downloads contain every configuration/objective audit row. CSV uses {{ merit.null_csv }} for null.</p>
    <p class="hash">Grid input SHA-256: {{ merit.grid_input_sha256 }}<br>Population SHA-256: {{ merit.population_sha256 }}</p>
    <section data-testid="merit-population"><h4>Common comparison population</h4>
      <p>Eligible configurations: {{ merit.population.configuration_ids.join(', ') || 'None' }}. Coverage {{ number(merit.population.configuration_coverage) }} · {{ merit.population.eligible_configuration_count }} / {{ merit.population.configuration_count }} configurations.</p>
      <p>Positive-weight operating points: {{ merit.population.operating_point_ids.join(', ') || 'None' }}.</p>
      <p>Zero-weight operating points excluded from merit: {{ merit.population.zero_weight_operating_point_ids.join(', ') || 'None' }}.</p>
    </section>
    <div class="actions"><button @click="downloadRaw(merit.rows_json, 'grid-merit.json', 'application/json')">Download merit JSON</button><button @click="downloadRaw(merit.rows_csv, 'grid-merit.csv', 'text/csv')">Download merit CSV</button></div>
    <h4>Objective weights and population extrema</h4>
    <div class="table-scroll" data-testid="merit-objectives"><table><thead><tr><th>Objective / unit</th><th>Direction / active</th><th>Weight / normalized weight</th><th>Minimum / all configuration IDs</th><th>Maximum / all configuration IDs</th></tr></thead><tbody><tr v-for="objective in merit.objectives" :key="objective.name"><td>{{ objective.name }}<br>{{ objective.unit }}</td><td>{{ objective.direction }}<br>{{ objective.active ? 'Active' : 'Inactive' }}</td><td>{{ number(objective.weight) }} / {{ number(objective.normalized_weight) }}</td><td>{{ number(objective.minimum) }} {{ objective.unit }}<br>{{ objective.minimum_configuration_ids.join(', ') || 'None' }}</td><td>{{ number(objective.maximum) }} {{ objective.unit }}<br>{{ objective.maximum_configuration_ids.join(', ') || 'None' }}</td></tr></tbody></table></div>
    <h4>Penalty ranking</h4><p v-if="merit.status !== 'computed'" role="status">No configuration has complete means for every active objective. Scores and ranks are unavailable.</p>
    <ol v-else class="ranking" data-testid="merit-ranking"><li v-for="row in ranking" :key="row.configuration_id"><span>Rank {{ row.rank }} · {{ row.configuration_id }} · penalty {{ number(row.score) }}</span><div class="bar-track"><div :style="{ width: `${100 * row.score}%` }" :data-score="row.score"></div></div></li></ol>
    <p>Equal scores share the same competition rank. Every tied configuration remains visible.</p>
    <h4>Configuration means, coverage, and contributions</h4>
    <p>{{ merit.rows.length }} configurations · page {{ page + 1 }} · at most {{ pageSize }} rows per page.</p>
    <div class="actions"><button :disabled="page === 0" @click="page--">Previous merit rows</button><button :disabled="(page + 1) * pageSize >= merit.rows.length" @click="page++">Next merit rows</button></div>
    <div class="table-scroll" data-testid="merit-configurations"><table><thead><tr><th>Configuration / status</th><th>Score / rank</th><th>Raw full means and coverage</th><th>Objective penalties and weighted contributions</th><th>Exclusions / missing operating points</th></tr></thead><tbody><tr v-for="row in rows" :key="row.configuration_id"><td>{{ row.configuration_id }}<br>{{ row.status }}</td><td>{{ number(row.score) }}<br>Rank {{ row.rank ?? 'Unavailable' }}</td><td><ul><li v-for="objective in row.objectives" :key="objective.name">{{ objective.name }}: {{ number(objective.full_mean) }} {{ objective.unit }}<br>{{ objective.complete ? 'Complete' : 'Incomplete' }} · coverage {{ number(objective.coverage) }} · retained {{ number(objective.retained_weight) }} / {{ number(objective.total_weight) }}<br>Missing points: {{ objective.missing_operating_point_ids.join(', ') || 'None' }}</li></ul></td><td><ul><li v-for="objective in row.objectives" :key="objective.name">{{ objective.name }}<br>Penalty {{ number(objective.penalty) }} · contribution {{ number(objective.contribution) }}</li></ul></td><td><ul v-if="row.exclusion_reasons.length"><li v-for="reason in row.exclusion_reasons" :key="reason.name">{{ reason.name }}: missing {{ reason.missing_operating_point_ids.join(', ') }}</li></ul><span v-else>None</span></td></tr></tbody></table></div>
  </section>
</template>

<style scoped>
.merit-evidence { min-width:0; border:1px solid var(--border-color,#aaa); border-radius:8px; padding:1rem; margin:1rem 0; }p { line-height:1.6; overflow-wrap:anywhere; }.hash { font-family:monospace; }.actions { display:flex; flex-wrap:wrap; gap:.4rem; }button { padding:.55rem .8rem; border:1px solid #7c98ae; border-radius:5px; color:inherit; background:var(--bg-secondary); cursor:pointer; }button:disabled { opacity:.5; cursor:default; }
.table-scroll { max-width:100%; overflow-x:auto; }table { border-collapse:collapse; min-width:800px; width:100%; }th,td { padding:.6rem; text-align:left; vertical-align:top; border-bottom:1px solid var(--border-color,#aaa); }td { min-width:120px; max-width:320px; overflow-wrap:anywhere; }ul { list-style:none; padding:0; margin:0; }li { margin:.4rem 0; }.ranking { padding-left:1.5rem; }.bar-track { background:var(--bg-tertiary,#eee); height:12px; width:100%; margin:.4rem 0; }.bar-track div { height:100%; background:#1686bd; }
@media(max-width:600px) { .merit-evidence { padding:.6rem; } }
</style>
