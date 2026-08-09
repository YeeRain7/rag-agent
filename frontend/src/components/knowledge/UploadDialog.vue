<script setup lang="ts">
import { ref } from 'vue'
import { useKnowledge } from '../../composables/useKnowledge'

const emit = defineEmits<{ uploaded: [] }>()
const visible = ref(false)
const uploading = ref(false)
const dragOver = ref(false)
const message = ref('')

function open() { visible.value = true; message.value = '' }
function close() { visible.value = false }

async function handleFile(file: File) {
  const ext = file.name.split('.').pop()?.toLowerCase()
  if (!ext || !['md', 'txt', 'pdf', 'docx'].includes(ext)) {
    message.value = `不支持 .${ext} 格式`
    return
  }
  uploading.value = true
  message.value = ''
  try {
    const { uploadFile } = useKnowledge()
    const result = await uploadFile(file)
    message.value = result.message
    if (result.success) emit('uploaded')
  } catch {
    message.value = '上传失败'
  } finally {
    uploading.value = false
  }
}

function onDrop(e: DragEvent) {
  dragOver.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file) handleFile(file)
}

function onFileInput(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (file) handleFile(file)
}

defineExpose({ open, close })
</script>

<template>
  <div v-if="visible" class="overlay" @click.self="close">
    <div class="dialog glass-elevated">
      <div class="dialog-header">
        <h3>上传文档</h3>
        <button class="close-btn" @click="close" aria-label="关闭">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>

      <div
        :class="['drop-zone', { 'drag-over': dragOver }]"
        @dragover.prevent="dragOver = true"
        @dragleave="dragOver = false"
        @drop.prevent="onDrop"
      >
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
        <p>拖拽文件到此处，或点击选择</p>
        <div class="format-badges">
          <span class="badge">PDF</span>
          <span class="badge">Word</span>
          <span class="badge">Markdown</span>
          <span class="badge">TXT</span>
        </div>
        <input type="file" accept=".md,.txt,.pdf,.docx" class="file-input" @change="onFileInput" />
      </div>

      <div v-if="uploading" class="status">上传中...</div>
      <div v-if="message" :class="['status', message.includes('失败') || message.includes('不支持') ? 'error' : 'ok']">{{ message }}</div>
    </div>
  </div>
</template>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.25);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.dialog {
  width: 420px;
  max-width: 90vw;
  padding: var(--space-xl);
}

.dialog-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--space-lg);
}

.dialog-header h3 {
  font-family: var(--font-heading);
  margin: 0;
  font-size: 17px;
  font-weight: 600;
  color: var(--text-primary);
}

.close-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  background: none;
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  color: var(--text-secondary);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.close-btn:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

.drop-zone {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 40px;
  border: 2px dashed var(--border-card);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-size: 14px;
  cursor: pointer;
  transition: all var(--transition-fast);
}

.drop-zone:hover,
.drop-zone.drag-over {
  border-color: var(--border-active);
  background: var(--bg-sidebar);
}

.format-badges {
  display: flex;
  gap: 6px;
  margin-top: 4px;
}

.badge {
  padding: 2px 10px;
  background: var(--bg-sidebar);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  letter-spacing: 0.3px;
}

.file-input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}

.status {
  margin-top: var(--space-md);
  text-align: center;
  font-size: 13px;
  color: var(--text-secondary);
}

.status.ok { color: #059669; }
.status.error { color: #dc2626; }
</style>
