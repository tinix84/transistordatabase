<script setup>
const props = defineProps({ modelValue: { type: Array, required: true } })
const emit = defineEmits(['update:modelValue'])
function update(index, key, value) { emit('update:modelValue', props.modelValue.map((row, i) => i === index ? { ...row, [key]: value } : row)) }
function add() {
  let suffix = 1
  while (props.modelValue.some(row => row.interface_id === `contact-${suffix}`)) suffix++
  emit('update:modelValue', [...props.modelValue, { interface_id: `contact-${suffix}`, path: 'common', rho: '', area: '', source: '', conditionsText: '' }])
}
</script>
<template>
  <section class="contact-list" data-testid="thermal-contacts">
    <p>Each row represents one identified physical contact. Distinct contacts on the same path form a series stack. Common contacts carry total heat. Branch contacts carry only their branch heat.</p>
    <fieldset v-for="(row, i) in modelValue" :key="i" :data-testid="`thermal-contact-${i}`">
      <legend>Physical contact {{ i + 1 }}</legend><div class="contact-fields">
      <label>Unique interface ID<input :value="row.interface_id" @input="update(i, 'interface_id', $event.target.value)" data-testid="contact-id"></label>
      <label>Physical path<select :value="row.path" @change="update(i, 'path', $event.target.value)" data-testid="contact-path"><option>common</option><option>forward</option><option>reverse</option></select></label>
      <label>Area-specific resistivity (K·m²/W)<input :value="row.rho" @input="update(i, 'rho', $event.target.value)" inputmode="decimal" data-testid="contact-rho"></label>
      <label>Contact area (m²)<input :value="row.area" @input="update(i, 'area', $event.target.value)" inputmode="decimal" data-testid="contact-area"></label>
      <label>Contact source<input :value="row.source" @input="update(i, 'source', $event.target.value)" data-testid="contact-source"></label>
      <label>Conditions: name,value (blank value means unknown)<textarea :value="row.conditionsText" @input="update(i, 'conditionsText', $event.target.value)" rows="2" data-testid="contact-conditions"></textarea></label>
      </div><button type="button" :disabled="modelValue.length === 1" @click="emit('update:modelValue', modelValue.filter((_, index) => index !== i))">Remove contact row {{ i + 1 }}</button>
    </fieldset>
    <button type="button" :disabled="modelValue.length >= 32" @click="add" data-testid="thermal-contact-add">Add physical contact</button>
    <p>All entries use SI units. R = resistivity / area. A sourced zero resistivity declares an ideal contact. Missing values are rejected.</p>
  </section>
</template>
<style scoped>
.contact-list { min-width: 0; grid-column: 1 / -1; } fieldset { min-width: 0; padding: .8rem; margin: .8rem 0; border: 1px solid #94a3b8; }
.contact-fields { display: grid; grid-template-columns: repeat(auto-fit,minmax(min(100%,220px),1fr)); gap: .7rem; }
label { display: grid; gap: .4rem; } input, select, textarea { box-sizing: border-box; width: 100%; min-width: 0; padding: .5rem; font: inherit; }
p { line-height: 1.5; overflow-wrap: anywhere; } button { margin: .6rem 0; padding: .6rem; font: inherit; }
</style>
