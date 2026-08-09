<script setup lang="ts">
import { computed, ref } from 'vue'
import type { SourceInfo } from '../../composables/useWebSocket'
import SourceCard from './SourceCard.vue'

const props = defineProps<{
  role: 'user' | 'assistant' | 'status'
  content: string
  sources?: SourceInfo[]
  stage?: string
  isLast?: boolean
}>()

const stageLabel = computed(() => {
  const map: Record<string, string> = {
    routing: '分析中...',
    retrieving: '检索中...',
    generating: '生成回答...'
  }
  return map[props.stage || ''] || props.stage
})

// 标记 "推理补充" 段落
const renderedContent = computed(() => {
  if (props.role !== 'assistant' || !props.content) return props.content
  let html = props.content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    // 代码块
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // 上角标引用 [^N] → 可点击
    .replace(/\[\^(\d+)\]/g, '<sup class="cite-ref" data-ref="$1">[$1]</sup>')
    // 推理补充标记
    .replace(/「推理补充」/g, '<span class="infer-tag">推理补充</span>')
    // 粗体
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // ## 标题前加分割线
    .replace(/\n+(#{2,3}\s)/g, '<hr class="soft-hr">$1')
    // --- → 浅色分割线
    .replace(/^---+$/gm, '<hr class="soft-hr">')
    // 双换行保留间距
    .replace(/\n\n+/g, '<br><br>')
    // 单换行
    .replace(/\n/g, '<br>')
  return html
})

// 来源去重（同文档合并）
const dedupedSources = computed(() => {
  if (!props.sources) return []
  const seen = new Set<string>()
  return props.sources.filter(s => {
    const key = s.doc_name || s.snippet.slice(0, 40)
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
})

const expandedIdx = ref<number | null>(null)
const showFollowInput = ref(false)
const followQuestion = ref('')
const copied = ref(false)

const emit = defineEmits<{
  copy: []
  retry: []
  followUp: [question: string]
}>()

function handleCopy() {
  navigator.clipboard.writeText(props.content).then(() => {
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  }).catch(() => {})
}

function handleFollowUpSend() {
  const q = followQuestion.value.trim()
  if (!q) return
  emit('followUp', q)
  followQuestion.value = ''
  showFollowInput.value = false
}

function closeFollowInput(e: MouseEvent) {
  const target = e.target as HTMLElement
  if (!target.closest('.actions-bar')) {
    showFollowInput.value = false
    document.removeEventListener('click', closeFollowInput)
  }
}

function toggleFollowInput() {
  showFollowInput.value = !showFollowInput.value
  if (showFollowInput.value) {
    setTimeout(() => document.addEventListener('click', closeFollowInput), 0)
  }
}
</script>

<template>
  <div :class="['message', role]">
    <div v-if="role === 'status'" class="status-badge">
      <div class="spinner"><span /><span /><span /><span /></div>
      {{ stageLabel }}
    </div>

    <div v-else-if="role === 'user'" class="bubble user-bubble">
      {{ content }}
    </div>

    <div v-else class="bubble ai-bubble">
      <div class="markdown-body" v-html="renderedContent" />

      <!-- 参考来源 -->
      <div v-if="dedupedSources.length" class="sources-section">
        <div class="sources-title">参考来源</div>
        <div
          v-for="s in dedupedSources"
          :key="s.index"
          class="source-item"
          :class="{ expanded: expandedIdx === s.index }"
          @click="expandedIdx = expandedIdx === s.index ? null : s.index"
        >
          <div class="source-summary">
            <span class="source-idx">[^{{ s.index }}]</span>
            <span class="source-doc">{{ s.doc_name || '未知文档' }}</span>
            <span v-if="s.doc_type" class="source-type">{{ s.doc_type }}</span>
          </div>
          <div v-if="expandedIdx === s.index && s.snippet" class="source-snippet">
            {{ s.snippet }}
          </div>
        </div>
      </div>

      <!-- 无来源提示 -->
      <div v-else-if="role === 'assistant' && content && !content.includes('分析中')" class="no-source">
        本回答基于模型知识，未引用内部文档
      </div>

      <!-- 操作栏 + 复制成功提示 -->
      <div v-if="role === 'assistant' && content" :class="['actions-bar', { 'always-show': isLast || showFollowInput }]">
        <button class="action-btn" @click="handleCopy">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          <span class="action-label">{{ copied ? '已复制' : '复制' }}</span>
        </button>
        <button class="action-btn" @click="$emit('retry')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
          <span class="action-label">重新生成</span>
        </button>
        <button class="action-btn" @click="toggleFollowInput">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          <span class="action-label">追问</span>
        </button>
        <!-- 追问输入框 -->
        <div v-if="showFollowInput" class="follow-input-row">
          <input
            v-model="followQuestion"
            class="follow-input"
            placeholder="基于此回答继续追问..."
            @keyup.enter="handleFollowUpSend"
          />
          <button class="follow-send" @click="handleFollowUpSend">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.message {
  display: flex;
  margin-bottom: 35px;
}

.message.user { justify-content: flex-end; }
.message.assistant, .message.status { justify-content: flex-start; }

/* 状态 */
.status-badge {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
  color: var(--text-tertiary);
  font-size: 13px;
}

/* 黑白旋转光环 */
.spinner {
  position: relative;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  animation: rotate 1.2s linear infinite;
  background: conic-gradient(var(--text-primary) 0deg, transparent 120deg, transparent 240deg, var(--text-primary) 360deg);
}
.spinner span { position: absolute; border-radius: 50%; height: 100%; width: 100%; background: inherit; }
.spinner span:nth-of-type(1) { filter: blur(2px); }
.spinner span:nth-of-type(2) { filter: blur(4px); }
.spinner span:nth-of-type(3) { filter: blur(6px); }
.spinner span:nth-of-type(4) { filter: blur(8px); }
.spinner::after {
  content: "";
  position: absolute;
  top: 3px; left: 3px; right: 3px; bottom: 3px;
  background: var(--bg-chat);
  border-radius: 50%;
}
@keyframes rotate { to { transform: rotate(360deg); } }

/* 气泡 */
.bubble { max-width: 100%; font-size: 16px; line-height: 1.8; }
.user-bubble {
  background: var(--bg-hover);
  color: var(--text-primary);
  padding: 10px 18px;
  border-radius: var(--radius-lg);
  border-bottom-right-radius: 4px;
  border: 1px solid var(--border-subtle);
}
.ai-bubble { color: var(--text-primary); padding: 0; background: var(--bg-chat); }

/* Markdown */
.markdown-body :deep(pre) {
  background: #f5f5f7;
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 14px 16px;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 14px;
  line-height: 1.6;
  margin: 10px 0;
}
.markdown-body :deep(code) {
  font-family: var(--font-mono);
  font-size: 14px;
  background: #f0f0f3;
  padding: 2px 6px;
  border-radius: 4px;
}
.markdown-body :deep(pre code) { background: none; padding: 0; }
.markdown-body :deep(strong) { font-weight: 600; }

/* 上角标引用 */
.markdown-body :deep(.cite-ref) {
  font-size: 0.75em;
  color: var(--accent);
  cursor: pointer;
  vertical-align: super;
  font-weight: 600;
}
.markdown-body :deep(.cite-ref:hover) {
  text-decoration: underline;
}

/* 推理补充 */
.markdown-body :deep(.infer-tag) {
  display: inline-block;
  font-size: 11px;
  padding: 1px 6px;
  background: var(--accent-dim);
  color: var(--text-secondary);
  border-radius: 3px;
  margin-right: 2px;
}

/* 内容块分割线 */
.markdown-body :deep(.soft-hr) {
  border: none;
  border-top: 1px solid var(--border-card);
  margin: 8px 0;
}

/* 参考来源区 */
.sources-section {
  margin-top: 20px;
  padding-top: 14px;
  border-top: 2px solid var(--border-card);
}

.sources-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 10px;
}

.source-item {
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background var(--transition-fast);
}

.source-item:hover { background: var(--bg-sidebar); }
.source-item.expanded { background: var(--bg-sidebar); }

.source-summary {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}

.source-idx {
  color: var(--accent);
  font-weight: 600;
  font-size: 12px;
}

.source-doc {
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.source-type {
  font-size: 10px;
  padding: 1px 6px;
  background: var(--accent-dim);
  color: var(--text-tertiary);
  border-radius: 3px;
  flex-shrink: 0;
}

.source-snippet {
  margin-top: 8px;
  padding: 10px 12px;
  background: #f5f5f7;
  border-left: 2px solid var(--accent);
  border-radius: 0 6px 6px 0;
  font-size: 13px;
  line-height: 1.6;
  color: var(--text-secondary);
}

.no-source {
  margin-top: 14px;
  font-size: 12px;
  color: var(--text-tertiary);
  font-style: italic;
  margin-bottom: 4px;
}

/* 操作栏 */
.actions-bar {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 8px;
  margin-bottom: 0;
  flex-wrap: wrap;
  opacity: 0;
  transition: opacity var(--transition-fast);
}

.actions-bar.always-show,
.message:hover .actions-bar {
  opacity: 1;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  background: none;
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  color: var(--text-tertiary);
  font-size: 11px;
  font-family: var(--font-body);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.action-btn:hover {
  background: var(--bg-sidebar);
  border-color: var(--border-card);
  color: var(--text-secondary);
}

.action-label {
  max-width: 0;
  overflow: hidden;
  white-space: nowrap;
  transition: max-width 0.2s ease;
}

.action-btn:hover .action-label {
  max-width: 60px;
}

/* 追问输入 */
.follow-input-row {
  display: flex;
  gap: 4px;
  width: 100%;
  margin-top: 6px;
}

.follow-input {
  flex: 1;
  padding: 6px 10px;
  background: var(--bg-sidebar);
  border: 1px solid var(--border-card);
  border-radius: 6px;
  font-size: 13px;
  font-family: var(--font-body);
  color: var(--text-primary);
  outline: none;
}

.follow-input::placeholder {
  color: var(--text-tertiary);
}

.follow-send {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  background: var(--text-primary);
  border: none;
  border-radius: 6px;
  color: var(--bg-chat);
  cursor: pointer;
  flex-shrink: 0;
}
</style>
