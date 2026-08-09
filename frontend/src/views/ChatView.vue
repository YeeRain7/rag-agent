<script setup lang="ts">
import { ref, computed, nextTick, watch, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useTheme } from '../composables/useTheme'
import { useWebSocket } from '../composables/useWebSocket'
import ChatMessage from '../components/chat/ChatMessage.vue'
import ChatInput from '../components/chat/ChatInput.vue'
import SessionList from '../components/chat/SessionList.vue'

const router = useRouter()
const { isDark, toggle: toggleTheme } = useTheme()

const sessions = ref<{ id: string; title: string; pinned?: boolean }[]>(
  JSON.parse(localStorage.getItem('chat_sessions') || 'null') || [{ id: 'default', title: '新对话' }]
)

// 持久化 hooks
onMounted(() => {
  const saved = localStorage.getItem('chat_sessions')
  if (saved) {
    try { sessions.value = JSON.parse(saved) } catch {}
  }
  const savedHistory = localStorage.getItem('chat_history')
  if (savedHistory) {
    try { sessionHistory.value = JSON.parse(savedHistory) } catch {}
  }
  // 恢复当前活跃会话的对话记录
  nextTick(() => {
    const history = sessionHistory.value[activeSession.value]
    if (history && history.length) {
      messages.value.push(...history)
      history.forEach((msg, i) => {
        if (msg.role === 'assistant' && msg.content) {
          typedContents.value[i] = msg.content
        }
      })
    }
  })
})
function persistSessions() {
  localStorage.setItem('chat_active', activeSession.value)
  localStorage.setItem('chat_sessions', JSON.stringify(sessions.value))
  // 只持久化最近 50 条消息的会话，控制存储量
  const compact: Record<string, typeof messages.value> = {}
  for (const [id, msgs] of Object.entries(sessionHistory.value)) {
    compact[id] = msgs.slice(-50)
  }
  try { localStorage.setItem('chat_history', JSON.stringify(compact)) } catch {}
}
const activeSession = ref(localStorage.getItem('chat_active') || 'default')
const searchQuery = ref('')
const showFeedback = ref(false)
const feedbackText = ref('')
const feedbackSent = ref(false)

const filteredSessions = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  if (!q) return sessions.value
  return sessions.value.filter(s => s.title.toLowerCase().includes(q))
})

const { messages, isConnected, isStreaming, send, abort } = useWebSocket(activeSession)

const localStreaming = computed(() => isStreaming.value || activeTypewriter.value !== null)

// 每会话独立消息历史
const sessionHistory = ref<Record<string, typeof messages.value>>({})

const msgContainer = ref<HTMLElement>()

function switchSession(newId: string) {
  const oldId = activeSession.value
  if (oldId === newId) return

  // 先停打字机
  if (typewriterTimer) { clearInterval(typewriterTimer); typewriterTimer = null }
  typedContents.value = {}

  // 保存旧会话——过滤掉 transient 状态消息
  if (oldId) {
    sessionHistory.value[oldId] = messages.value.filter(m => m.role !== 'status')
  }

  // 切换（触发 useWebSocket 清空 + 重连）
  activeSession.value = newId
  localStorage.setItem('chat_active', newId)

  // 恢复新会话历史
  nextTick(() => {
    if (sessionHistory.value[newId]) {
      const restored = sessionHistory.value[newId]
      messages.value.push(...restored)
      restored.forEach((msg, i) => {
        if (msg.role === 'assistant' && msg.content) {
          typedContents.value[i] = msg.content
        }
      })
    }
  })
}

// 打字机效果 — 每条 assistant 消息逐字显示
const typedContents = ref<Record<number, string>>({})
const activeTypewriter = ref<number | null>(null)

watch(
  () => messages.value,
  (newMsgs) => {
    newMsgs.forEach((msg, i) => {
      if (msg.role !== 'assistant' || !msg.content) return
      // 跳过已经打过的消息
      if (typedContents.value[i] === msg.content) return
      // 新消息或有更新
      startTypewriter(i, msg.content)
    })
  },
  { deep: true }
)

let typewriterTimer: ReturnType<typeof setInterval> | null = null

onUnmounted(() => {
  if (typewriterTimer) clearInterval(typewriterTimer)
  sessionHistory.value[activeSession.value] = [...messages.value]
  persistSessions()
})

function startTypewriter(index: number, fullText: string) {
  if (typewriterTimer) clearInterval(typewriterTimer)
  activeTypewriter.value = index
  typedContents.value[index] = ''
  let pos = 0
  // 速度：每 30ms 一次，每次前进 2-5 个字符
  typewriterTimer = setInterval(() => {
    pos += Math.floor(Math.random() * 4) + 2
    if (pos >= fullText.length) {
      pos = fullText.length
      if (typewriterTimer) clearInterval(typewriterTimer)
      typewriterTimer = null
      activeTypewriter.value = null
    }
    typedContents.value[index] = fullText.slice(0, pos)
  }, 30)
}

watch(
  () => messages.value.length,
  async () => {
    await nextTick()
    if (msgContainer.value) {
      msgContainer.value.scrollTop = msgContainer.value.scrollHeight
    }
  }
)

function handleSend(content: string) {
  send(content)
  const s = sessions.value.find(x => x.id === activeSession.value)
  if (!s) {
    // 当前是临时会话，首次提问后才加入列表
    sessions.value.unshift({ id: activeSession.value, title: content.slice(0, 30) })
    persistSessions()
  } else if (s.title === '新对话') {
    s.title = content.slice(0, 30)
    persistSessions()
  }
}

function handleNewSession() {
  const id = Date.now().toString()
  // 不加入 sessions 列表，仅在提问后才出现
  switchSession(id)
}

function handlePinSession(id: string) {
  const idx = sessions.value.findIndex(s => s.id === id)
  if (idx === -1) return
  const [item] = sessions.value.splice(idx, 1)
  item.pinned = true
  sessions.value.unshift(item)
  persistSessions()
}

function handleUnpinSession(id: string) {
  const s = sessions.value.find(s => s.id === id)
  if (s) s.pinned = false
  persistSessions()
}

function handleRenameSession(id: string, title: string) {
  const s = sessions.value.find(s => s.id === id)
  if (s) { s.title = title; persistSessions() }
}

function handleCopy() {
  // 复制已在 ChatMessage 中通过 navigator.clipboard 完成
}

function handleRetry() {
  // 找到上一条用户消息重新发送
  const userMsgs = messages.value.filter(m => m.role === 'user')
  if (userMsgs.length) {
    send(userMsgs[userMsgs.length - 1].content)
  }
}

function handleFollowUp(question: string) {
  send(question)
}

function handleStop() {
  if (typewriterTimer) { clearInterval(typewriterTimer); typewriterTimer = null }
  activeTypewriter.value = null
  // 去掉还在转的状态消息
  const lastIdx = messages.value.length - 1
  if (lastIdx >= 0 && messages.value[lastIdx].role === 'status') {
    messages.value.pop()
  }
  abort()
}

function handleFeedbackSubmit() {
  if (!feedbackText.value.trim()) return
  // TODO: 后续对接后端反馈接口
  console.log('反馈:', feedbackText.value)
  feedbackSent.value = true
  setTimeout(() => {
    showFeedback.value = false
    feedbackText.value = ''
    feedbackSent.value = false
  }, 1000)
}

function handleDeleteSession(id: string) {
  const idx = sessions.value.findIndex(s => s.id === id)
  if (idx === -1) return
  sessions.value.splice(idx, 1)
  persistSessions()
  delete sessionHistory.value[id]
  if (activeSession.value === id) {
    if (!sessions.value.length) {
      sessions.value.push({ id: 'default', title: '新对话' })
      persistSessions()
    }
    switchSession(sessions.value[0].id)
  }
}
</script>

<template>
  <div class="chat-view">
    <!-- 侧边栏：品牌 + 会话列表 + 底部知识库入口 -->
    <aside class="sidebar">
      <div class="sidebar-top">
        <span class="brand">Alter-RAG知识库平台</span>
        <button class="new-chat-btn" @click="handleNewSession">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          新对话
        </button>
        <div class="search-box">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="search-icon"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <input
            v-model="searchQuery"
            class="search-input"
            placeholder="搜索会话..."
          />
        </div>
        <SessionList
          :sessions="filteredSessions"
          :active-id="activeSession"
          @select="switchSession($event)"
          @pin="handlePinSession"
          @unpin="handleUnpinSession"
          @rename="handleRenameSession"
          @delete="handleDeleteSession"
        />
      </div>

      <div class="sidebar-bottom">
        <button class="knowledge-link" @click="router.push('/knowledge')">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
          知识库
        </button>
      </div>
    </aside>

    <!-- 对话工作区 -->
    <main class="workspace">
      <!-- 右上角：反馈 + 主题切换 -->
      <div class="top-actions">
        <button class="action-btn" @click="showFeedback = true" aria-label="反馈">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          反馈
        </button>
        <button class="action-btn" @click="toggleTheme" :aria-label="isDark ? '切换亮色' : '切换暗色'">
          <svg v-if="isDark" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
          <svg v-else width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
        </button>
      </div>

      <!-- 反馈弹窗 -->
      <div v-if="showFeedback" class="feedback-overlay" @click.self="showFeedback = false">
        <div class="feedback-dialog glass-elevated">
          <h3>提交反馈</h3>
          <textarea
            v-model="feedbackText"
            class="feedback-input"
            placeholder="请描述您遇到的问题或改进建议…"
            rows="5"
          />
          <div class="feedback-actions">
            <button class="feedback-cancel" @click="showFeedback = false">取消</button>
            <button class="feedback-submit" @click="handleFeedbackSubmit">
              {{ feedbackSent ? '已提交' : '提交' }}
            </button>
          </div>
        </div>
      </div>

      <!-- 会话标题栏 -->
      <div class="chat-header">
        <span class="chat-title">{{ sessions.find(s => s.id === activeSession)?.title || '新对话' }}</span>
      </div>

      <div class="msg-wrapper" ref="msgContainer">
        <div class="msg-column">
          <div v-if="messages.length === 0" class="welcome">
            <h1>我能根据知识库回答问题，有什么可以帮助您的？</h1>
            <p class="welcome-hint">支持上传 PDF、Word、Markdown、TXT 文档构建知识库</p>
          </div>

          <ChatMessage
            v-for="(msg, i) in messages"
            :key="i"
            :role="msg.role"
            :content="msg.role === 'assistant' ? (typedContents[i] ?? msg.content) : msg.content"
            :sources="msg.sources"
            :stage="msg.stage"
            :is-last="msg.role === 'assistant' && i === messages.length - 1"
            @copy="handleCopy"
            @retry="handleRetry"
            @follow-up="handleFollowUp"
          />
        </div>
      </div>

      <div class="input-wrapper">
        <div class="input-row">
          <ChatInput :streaming="localStreaming" @send="handleSend" @stop="handleStop" />
        </div>
        <div class="status-bar">
          <span :class="['conn-dot', isConnected ? 'online' : 'offline']" />
          {{ isConnected ? '就绪' : '连接中...' }}
        </div>
      </div>
    </main>
  </div>
</template>

<style scoped>
.chat-view {
  display: flex;
  flex: 1;
  min-width: 0;
  overflow: hidden;
}

/* === 侧边栏 === */
.sidebar {
  width: 260px;
  flex-shrink: 0;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border-sidebar);
  display: flex;
  flex-direction: column;
}

.sidebar-top {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: var(--space-md) var(--space-sm);
  overflow: hidden;
}

.brand {
  font-family: var(--font-heading);
  font-size: 16px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.3px;
  padding: var(--space-xs) var(--space-sm) var(--space-md);
  flex-shrink: 0;
}

.new-chat-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 9px 12px;
  margin-bottom: var(--space-sm);
  background: var(--bg-chat);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  font-size: 13px;
  font-weight: 500;
  font-family: var(--font-body);
  cursor: pointer;
  transition: background var(--transition-fast);
  flex-shrink: 0;
}

.new-chat-btn:hover { background: var(--bg-hover); }

/* 搜索框 */
.search-box {
  position: relative;
  margin-bottom: var(--space-sm);
  flex-shrink: 0;
}

.search-icon {
  position: absolute;
  left: 10px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-tertiary);
  pointer-events: none;
}

.search-input {
  width: 100%;
  padding: 7px 10px 7px 30px;
  background: var(--bg-chat);
  border: 1px solid var(--border-subtle);
  border-radius: 7px;
  font-size: 12px;
  font-family: var(--font-body);
  color: var(--text-primary);
  outline: none;
  transition: border-color var(--transition-fast);
}

.search-input::placeholder {
  color: var(--text-tertiary);
}

.search-input:focus {
  border-color: var(--border-active);
}

/* 底部知识库入口 */
.sidebar-bottom {
  padding: var(--space-sm);
  border-top: 1px solid var(--border-sidebar);
  flex-shrink: 0;
}

.knowledge-link {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 9px 12px;
  background: none;
  border: none;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  font-family: var(--font-body);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.knowledge-link:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

/* === 工作区 === */
.workspace {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--bg-chat);
  min-width: 0;
  overflow: hidden;
  position: relative;
}

/* 会话标题栏 */
.chat-header {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px var(--space-lg);
  border-bottom: 1px solid var(--border-subtle);
  flex-shrink: 0;
}

.chat-title {
  font-family: var(--font-heading);
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
  max-width: 400px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.msg-wrapper {
  flex: 1;
  overflow-y: auto;
  display: flex;
  justify-content: center;
  scroll-behavior: smooth;
  padding-bottom: var(--space-2xl);
}

.msg-column {
  width: 100%;
  max-width: 1000px;
  padding: var(--space-xl) var(--space-lg);
}

.welcome {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 50vh;
}

.welcome h1 {
  font-family: var(--font-heading);
  font-size: 22px;
  font-weight: 500;
  color: var(--text-secondary);
  letter-spacing: -0.2px;
}

.welcome-hint {
  margin-top: 10px;
  font-size: 13px;
  color: var(--text-tertiary);
}

.input-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: var(--space-md) var(--space-lg) var(--space-md);
  background: var(--bg-chat);
}

.input-row {
  width: 100%;
  max-width: 1000px;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 5px;
  padding-top: 6px;
  font-size: 11px;
  color: var(--text-tertiary);
}

.conn-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
}

.conn-dot.online { background: #34d399; }
.conn-dot.offline { background: #f87171; }

/* 右上角操作按钮组 */
.top-actions {
  position: absolute;
  top: 48px;
  right: var(--space-lg);
  display: flex;
  gap: 6px;
  z-index: 10;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 6px 12px;
  background: none;
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  color: var(--text-tertiary);
  font-size: 12px;
  font-family: var(--font-body);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.action-btn:hover {
  border-color: var(--border-active);
  color: var(--text-secondary);
  background: var(--bg-sidebar);
}

/* 反馈弹窗 */
.feedback-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.25);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.feedback-dialog {
  width: 420px;
  max-width: 90vw;
  padding: var(--space-xl);
}

.feedback-dialog h3 {
  font-family: var(--font-heading);
  margin: 0 0 var(--space-md);
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.feedback-input {
  width: 100%;
  padding: 12px;
  background: var(--bg-sidebar);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  font-size: 14px;
  font-family: var(--font-body);
  resize: vertical;
  outline: none;
  transition: border-color var(--transition-fast);
}

.feedback-input:focus {
  border-color: var(--border-active);
}

.feedback-input::placeholder {
  color: var(--text-tertiary);
}

.feedback-actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-sm);
  margin-top: var(--space-md);
}

.feedback-cancel {
  padding: 7px 16px;
  background: none;
  border: 1px solid var(--border-card);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 13px;
  font-family: var(--font-body);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.feedback-cancel:hover {
  background: var(--bg-hover);
}

.feedback-submit {
  padding: 7px 20px;
  background: var(--text-primary);
  border: none;
  border-radius: var(--radius-sm);
  color: var(--bg-chat);
  font-size: 13px;
  font-weight: 500;
  font-family: var(--font-body);
  cursor: pointer;
  transition: opacity var(--transition-fast);
}

.feedback-submit:hover {
  opacity: 0.85;
}
</style>
