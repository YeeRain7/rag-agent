<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ streaming: boolean }>()
const emit = defineEmits<{ send: [content: string]; stop: [] }>()
const input = ref('')

function handleSend() {
  const text = input.value.trim()
  if (!text) return
  emit('send', text)
  input.value = ''
}

function handleStop() {
  emit('stop')
}
</script>

<template>
  <div class="chat-input">
    <input
      v-model="input"
      class="glass-input input-field"
      :placeholder="streaming ? 'AI 正在生成...' : '输入你的问题，Enter 发送...'"
      @keyup.enter="streaming ? null : handleSend()"
    />
    <!-- 停止按钮 -->
    <button v-if="streaming" class="stop-btn" @click="handleStop" aria-label="停止生成">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
    </button>
    <!-- 发送按钮 -->
    <button v-else class="send-btn" @click="handleSend" aria-label="发送">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <line x1="12" y1="5" x2="12" y2="19" />
        <polyline points="5 12 12 5 19 12" />
      </svg>
    </button>
  </div>
</template>

<style scoped>
.chat-input {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}

.input-field {
  flex: 1;
  padding: 12px 20px;
  font-size: 16px;
  font-family: var(--font-body);
  height: 52px;
}

.input-field::placeholder {
  color: var(--text-placeholder);
}

.send-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 52px;
  height: 52px;
  flex-shrink: 0;
  background: var(--text-primary);
  border: none;
  border-radius: 50%;
  color: var(--bg-chat);
  cursor: pointer;
  transition: opacity var(--transition-fast);
}

.send-btn:hover {
  opacity: 0.85;
}

.stop-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 52px;
  height: 52px;
  flex-shrink: 0;
  background: var(--text-primary);
  border: none;
  border-radius: 50%;
  color: var(--bg-chat);
  cursor: pointer;
  transition: opacity var(--transition-fast);
}

.stop-btn:hover {
  opacity: 0.85;
}
</style>
