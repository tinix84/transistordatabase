import { finiteDecimal } from './analysisFiles.js'

export const lossComponents = ['conduction', 'turn_on', 'turn_off']
export const currentDirections = ['positive', 'negative', 'zero']

// Inventory and input serialization only. Python assigns and calculates all heat.
export function stageMembers(bindings) {
  return bindings.flatMap(binding => binding.members.map(id => ({
    id, role: binding.role, layout_complete: false, junctions: [],
    loss_owners: Object.fromEntries(lossComponents.map(component => [component,
      Object.fromEntries(currentDirections.map(direction => [direction, { die: null, junction_id: null }]))])),
  })))
}

export function serializeMembers(members) {
  return members.map(member => ({ ...member,
    loss_owners: Object.fromEntries(lossComponents.map(component => [component,
      Object.fromEntries(currentDirections.map(direction => {
        const owner = member.loss_owners[component][direction]
        return [direction, { die: owner.die || null, junction_id: owner.junction_id || null }]
      }))])),
    junctions: member.junctions.map(junction => ({ ...junction,
      rtheta_k_w: junction.rtheta_k_w === '' || junction.rtheta_k_w === null ? null : finiteDecimal(junction.rtheta_k_w, 'Thermal resistance'),
      source: junction.source?.trim() || null,
    })),
  }))
}
