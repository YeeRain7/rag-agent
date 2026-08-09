<script setup lang="ts">
import type { DocFile } from '../../composables/useKnowledge'

defineProps<{ documents: DocFile[]; loading: boolean }>()
defineEmits<{ delete: [filename: string] }>()
</script>

<template>
  <div class="doc-list">
    <div v-if="loading" class="loading">加载中...</div>

    <div v-else-if="!documents.length" class="empty">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.25">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      <span>暂无文档</span>
    </div>

    <div v-else class="table">
      <div v-for="doc in documents" :key="doc.filename" class="table-row">
        <span class="col-name">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="file-icon"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          {{ doc.filename }}
        </span>
        <span class="col-size">{{ doc.size_kb }} KB</span>
        <button class="delete-btn" @click="$emit('delete', doc.filename)" aria-label="删除">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.doc-list { min-height: 160px; }

.table { width: 100%; }

.table-row {
  display: flex;
  align-items: center;
  padding: 11px 0;
  border-bottom: 1px solid var(--border-subtle);
  font-size: 14px;
}

.table-row:hover {
  background: var(--bg-sidebar);
  margin: 0 -12px;
  padding-left: 12px;
  padding-right: 12px;
  border-radius: 6px;
}

.col-name {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-icon { flex-shrink: 0; opacity: 0.35; }

.col-size {
  width: 72px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  font-size: 12px;
  text-align: right;
  padding-right: var(--space-md);
}

.col-action { width: 40px; text-align: right; }
.col-name { padding-right: var(--space-md); }

.delete-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  background: none;
  border: 1px solid transparent;
  border-radius: 6px;
  color: var(--text-tertiary);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.delete-btn:hover {
  background: #fef2f2;
  border-color: #fecaca;
  color: #ef4444;
}

.loading, .empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 40px 0;
  color: var(--text-tertiary);
  font-size: 14px;
}
</style>
