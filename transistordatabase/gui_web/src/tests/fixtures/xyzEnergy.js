export const xyzText = '9,600,90\n3,200,30\n0,600,12\n9,200,60\n0,200,0'
export function xyzResult() {
  const rows = [[9,600,90],[3,200,30],[0,600,12],[9,200,60],[0,200,0]].map(([i,v,e], index) => ({
    id: `line-${index + 1}`, line_number: index + 1, raw_record: `${i},${v},${e}`, lexemes: [String(i),String(v),String(e)],
    current_decimal_key: String(i), voltage_decimal_key: String(v), current_a: i, voltage_v: v, energy_j: e * 1e-6,
    status: 'available', cell_status: 'available', reason: null,
  }))
  return { method_version: 'xyz-energy-map-v1', energy_kind: 'e_on', conditions: { energy_convention: 'Recorded switching energy', t_j_c: null },
    source: { name: 'pasted-xyz.txt', source: 'Manual fixture', source_kind: 'manual', input_kind: 'paste', sha256: 'a'.repeat(64), hash_scope: 'utf8-paste-bytes', byte_count: xyzText.length, encoding: 'utf-8' },
    options: { delimiter: 'csv', decimal: '.', has_header: false, columns: { current: 0, voltage: 1, energy: 2 }, units: { current: 'A', voltage: 'V', energy: 'µJ' }, missing_energy_tokens: [] },
    request_sha256: 'b'.repeat(64), currents_a: [0,3,9], voltages_v: [200,600], current_decimal_keys: ['0','3','9'], voltage_decimal_keys: ['200','600'], units: { current: 'A', voltage: 'V', energy: 'J' },
    cells: [200,600].map(v => [0,3,9].map(i => { const row = rows.find(row => row.current_a === i && row.voltage_v === v); return { current_a: i, voltage_v: v, energy_j: row?.energy_j ?? null, status: row ? 'available' : 'missing', row_ids: row ? [row.id] : [], reason: row ? null : 'no-source-row' } })),
    rows, header: null, census: { source_rows: 5, available_rows: 5, unknown_rows: 0, available_cells: 5, missing_cells: 1, unknown_cells: 0, ambiguous_cells: 0, duplicate_coordinates: 0, total_cells: 6 }, limits: { max_source_bytes: 4_000_000 }, assumptions: [],
  }
}
