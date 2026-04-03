<template>
  <section class="examples-wrap">
    <div class="examples-list">
      <button
        v-for="item in currentExamples"
        :key="selectedDomain + '-' + item"
        class="example-chip"
        :class="selectedDomain"
        @click="$emit('select-example', item)"
      >
        {{ item }}
      </button>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  selectedDomain: {
    type: String,
    required: true,
  },
})

defineEmits(['select-example'])

const exampleMap = {
  medical: [
    'What is hypertension?',
    'What causes acute pancreatitis?',
    'What is the difference between type 1 and type 2 diabetes?',
    'What are common symptoms of anemia?',
  ],
  finance: [
    'What is the difference between stocks and bonds?',
    'What does inflation mean?',
    'How do interest rates affect the economy?',
    'What is a balance sheet?',
  ],
  legal: [
    'What is the difference between civil law and criminal law?',
    'What is a breach of contract?',
    'What does negligence mean in law?',
    'What is intellectual property?',
  ],
  general: [
    'What is machine learning?',
    'How does photosynthesis work?',
    'What causes climate change?',
    'What is the difference between DNA and RNA?',
  ],
  multidomain: [
    'How can AI be used in healthcare?',
    'What legal issues can arise from financial fraud?',
    'How does public health policy affect the economy?',
    'What are the risks of using AI in law and medicine?',
  ],
}

const currentExamples = computed(() => {
  return exampleMap[props.selectedDomain] || exampleMap.multidomain
})
</script>

<style scoped>
.examples-wrap {
  width: 100%;
}

.examples-list {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.example-chip {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.035);
  color: #cfd8e6;
  border-radius: 999px;
  padding: 10px 13px;
  font-size: 12.5px;
  line-height: 1.35;
  cursor: pointer;
  transition: all 0.18s ease;
  text-align: left;
}

.example-chip:hover {
  transform: translateY(-1px);
  background: rgba(255,255,255,0.08);
  color: #ffffff;
}

.example-chip.medical:hover {
  border-color: rgba(91,124,255,0.55);
}
.example-chip.finance:hover {
  border-color: rgba(16,185,129,0.55);
}
.example-chip.legal:hover {
  border-color: rgba(139,92,246,0.55);
}
.example-chip.general:hover {
  border-color: rgba(245,158,11,0.55);
}
.example-chip.multidomain:hover {
  border-color: rgba(236,72,153,0.55);
}

@media (max-width: 700px) {
  .examples-list {
    gap: 8px;
  }

  .example-chip {
    width: 100%;
    border-radius: 16px;
  }
}
</style>
