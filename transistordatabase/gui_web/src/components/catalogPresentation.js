/** Export supplied scientific values unchanged; spreadsheet protection applies to text only. */
export function catalogCsv(result, metrics, candidates) {
  const keys = Object.keys(metrics)
  const columns = ['id', 'part_number', 'manufacturer', 'technology', 'source_id', 'sheet', 'row', 'source_sha256', 'currency', 'snapshot', 'quote_quantity', 'min_quantity', 'source_snapshot_id', 'quoted_price_original', 'resistance_condition', 'gate_charge_condition', 'conditions_json', 'quote_context_json', 'cost_cohort_json', 'normalization_mode', 'pareto', 'row_kind', 'population_evidence', ...keys,
    ...keys.flatMap(k => [`${k}_relative`, `${k}_denominator`, `${k}_population_hash`, `${k}_population_ids`, `${k}_denominator_ids`, `${k}_exclusion`, `${k}_population_exclusions_json`]), 'sources_json', 'workbook_evidence_json', 'metric_exclusions_json', 'workbook_derivation_json']
  const quote = value => {
    const text = typeof value === 'number' ? String(value) : String(value ?? '').replace(/^[=+@-]/, "'$&")
    return `"${text.replaceAll('"', '""')}"`
  }
  return [columns, ...(candidates.length ? candidates : [{}]).map((p, index) => {
    const source = p.source || result.sources?.find(s => s.id === p.source_id) || result.source || {}
    const row = { ...p, ...p.metrics, sheet: source.sheet, source_sha256: source.sha256, currency: Object.hasOwn(p, 'currency') ? p.currency : source.currency, snapshot: source.snapshot,
      resistance_condition: p.conditions?.resistance, gate_charge_condition: p.conditions?.gate_charge,
      conditions_json: JSON.stringify(p.conditions || {}), quote_context_json: JSON.stringify(p.quote_context || {}),
      source_snapshot_id: source.snapshot_id, quoted_price_original: p.price,
      cost_cohort_json: JSON.stringify(result.cost_cohort || {}), normalization_mode: result.normalization?.mode,
      row_kind: candidates.length ? 'candidate' : 'evidence_only', population_evidence: index === 0 ? 'full_ids' : 'hash_reference' }
    row.sources_json = index === 0 ? JSON.stringify(result.sources || [result.source].filter(Boolean)) : ''
    row.workbook_evidence_json = JSON.stringify(p.workbook_evidence || {})
    row.metric_exclusions_json = JSON.stringify(p.metric_exclusions || {})
    row.workbook_derivation_json = index === 0 ? JSON.stringify(result.workbook_derivation || {}) : ''
    for (const key of keys) {
      const evidence = result.normalization?.metrics?.[key]
      row[`${key}_relative`] = p.normalization?.[key]?.relative
      row[`${key}_denominator`] = evidence?.denominator
      row[`${key}_population_hash`] = evidence?.population_hash
      // Population and tie lists are shared evidence, never repeated per candidate.
      row[`${key}_population_ids`] = evidence && index === 0 ? JSON.stringify(evidence.population_ids) : ''
      row[`${key}_denominator_ids`] = evidence && index === 0 ? JSON.stringify(evidence.denominator_ids) : ''
      row[`${key}_exclusion`] = p.normalization?.[key]?.reason || p.metric_exclusions?.[key]
      row[`${key}_population_exclusions_json`] = evidence && index === 0 ? JSON.stringify(evidence.exclusions || []) : ''
    }
    return columns.map(k => row[k])
  })].map(row => row.map(quote).join(',')).join('\r\n')
}
