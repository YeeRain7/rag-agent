<script setup lang="ts">
defineProps<{ index: number; docName?: string; snippet?: string }>()
</script>

<template>
  <div class="source-card">
    <span class="source-index">[^{{ index }}]</span>
    <span class="source-text">{{ docName || `来源 ${index}` }}</span>
  </div>
</template>

<style scoped>
.source-card {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  background: var(--bg-sidebar);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  font-size: 12px;
  color: var(--text-secondary);
  cursor: default;
  transition: border-color var(--transition-fast);
}

.source-card:hover {
  border-color: var(--border-active);
}

.source-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 16px;
  padding: 0 4px;
  background: var(--text-primary);
  color: #fff;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
  font-family: var(--font-mono);
}

.source-text {
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
