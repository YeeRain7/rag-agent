<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  sessions: { id: string; title: string; pinned?: boolean }[]
  activeId?: string
}>()

const emit = defineEmits<{
  select: [id: string]
  pin: [id: string]
  unpin: [id: string]
  rename: [id: string, title: string]
  delete: [id: string]
}>()

const openMenuId = ref<string | null>(null)
const showRenameModal = ref(false)
const renameId = ref('')
const renameTitle = ref('')

function toggleMenu(id: string) {
  openMenuId.value = openMenuId.value === id ? null : id
}

function closeMenu() {
  openMenuId.value = null
}

function openRename(id: string, currentTitle: string) {
  renameId.value = id
  renameTitle.value = currentTitle
  showRenameModal.value = true
  mouseDownOutside = false
  closeMenu()
}

function confirmRename() {
  const name = renameTitle.value.trim()
  if (name) {
    emit('rename', renameId.value, name)
  }
  showRenameModal.value = false
  mouseDownOutside = false
}

// 区分点击外部 vs 拖拽选文字：只有 mousedown+up 都在弹窗外才关闭
let mouseDownOutside = false

function onOverlayMouseDown(e: MouseEvent) {
  mouseDownOutside = (e.target as HTMLElement).classList.contains('rename-overlay')
}

function onOverlayMouseUp(e: MouseEvent) {
  const isOutside = (e.target as HTMLElement).classList.contains('rename-overlay')
  if (mouseDownOutside && isOutside) {
    showRenameModal.value = false
    mouseDownOutside = false
  }
}
</script>

<template>
  <div class="session-list">
    <div v-if="!sessions.length" class="empty">暂无会话</div>

    <div class="sessions">
      <div
        v-for="s in sessions"
        :key="s.id"
        :class="['session-row', { active: s.id === activeId, pinned: s.pinned }]"
      >
        <button class="session-item" @click="emit('select', s.id)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="chat-icon"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          <span class="session-title">{{ s.title || '新对话' }}</span>
        </button>

        <!-- 置顶图标：hover 时隐藏，显示三点菜单 -->
        <svg v-if="s.pinned" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" class="pin-icon-fixed"><path d="M16 12V4h1V2H7v2h1v8l-2 2v2h5.2v6h1.6v-6H18v-2l-2-2z"/></svg>

        <!-- 三点菜单按钮 -->
        <button
          :class="['menu-btn', { 'show-always': s.pinned }]"
          @click.stop="toggleMenu(s.id)"
          aria-label="更多操作"
        >
          <svg width="16" height="4" viewBox="0 0 24 4" fill="currentColor"><circle cx="4" cy="2" r="2"/><circle cx="12" cy="2" r="2"/><circle cx="20" cy="2" r="2"/></svg>
        </button>

        <!-- 下拉菜单 -->
        <div v-if="openMenuId === s.id" class="dropdown">
          <button v-if="s.pinned" class="dropdown-item" @click.stop="emit('unpin', s.id); closeMenu()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="4" y1="20" x2="20" y2="4"/><path d="M16 12V4h1V2H7v2h1v5l-2 2v2h5.2v6h1.6v-6H18v-2l-2-2z"/></svg>
            取消置顶
          </button>
          <button v-else class="dropdown-item" @click.stop="emit('pin', s.id); closeMenu()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 12V4h1V2H7v2h1v8l-2 2v2h5.2v6h1.6v-6H18v-2l-2-2z"/></svg>
            置顶
          </button>
          <button class="dropdown-item" @click.stop="openRename(s.id, s.title)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>
            重命名
          </button>
          <button class="dropdown-item danger" @click.stop="emit('delete', s.id); closeMenu()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            删除
          </button>
        </div>

        <!-- 点击空白关闭菜单 -->
        <div v-if="openMenuId === s.id" class="menu-overlay" @click="closeMenu" />
      </div>
    </div>
  </div>

  <!-- 重命名弹窗 -->
  <div
    v-if="showRenameModal"
    class="rename-overlay"
    @mousedown="onOverlayMouseDown"
    @mouseup="onOverlayMouseUp"
  >
    <div class="rename-dialog" @mousedown.stop @mouseup.stop>
      <h3>编辑会话名称</h3>
      <input
        v-model="renameTitle"
        class="rename-input"
        @keyup.enter="confirmRename"
        autofocus
      />
      <div class="rename-actions">
        <button class="rename-cancel" @click="showRenameModal = false">取消</button>
        <button class="rename-confirm" @click="confirmRename">确定</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.session-list {
  display: flex;
  flex-direction: column;
  flex: 1;
  overflow: hidden;
}

.sessions {
  flex: 1;
  overflow-y: auto;
  padding: 0 var(--space-sm);
}

.session-row {
  position: relative;
  display: flex;
  align-items: center;
  margin-bottom: 1px;
  border-radius: var(--radius-sm);
  transition: background var(--transition-fast);
}

.session-row:hover {
  background: var(--bg-hover);
}

.session-row.active {
  background: var(--bg-active);
}

.session-row.pinned {
  background: var(--bg-hover);
}

.session-row.pinned.active {
  background: var(--bg-active);
}

.session-item {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  padding: 9px 0 9px 12px;
  background: none;
  border: none;
  color: var(--text-secondary);
  font-size: 13px;
  font-family: var(--font-body);
  text-align: left;
  cursor: pointer;
  min-width: 0;
}

.session-row.active .session-item {
  color: var(--text-primary);
  font-weight: 500;
}

.chat-icon {
  flex-shrink: 0;
  opacity: 0.5;
}

.pin-icon-fixed {
  flex-shrink: 0;
  opacity: 0.55;
  color: var(--text-secondary);
  margin-right: 6px;
}

.session-row:hover .pin-icon-fixed {
  display: none;
}

.menu-btn.show-always {
  display: none;
  opacity: 0;
}

.session-row:hover .menu-btn.show-always {
  display: flex;
  opacity: 1;
}

.session-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 三点菜单按钮 */
.menu-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  flex-shrink: 0;
  background: none;
  border: none;
  border-radius: 6px;
  color: var(--text-tertiary);
  cursor: pointer;
  opacity: 0;
  margin-right: 2px;
  transition: all var(--transition-fast);
}

.session-row:hover .menu-btn {
  opacity: 1;
}

.menu-btn:hover {
  background: var(--border-card);
  color: var(--text-primary);
}

/* 下拉菜单 */
.dropdown {
  position: absolute;
  right: 4px;
  top: 100%;
  z-index: 100;
  min-width: 120px;
  background: var(--bg-chat);
  border: 1px solid var(--border-card);
  border-radius: 8px;
  box-shadow: var(--shadow-lg);
  padding: 4px;
}

.dropdown-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px 12px;
  background: none;
  border: none;
  border-radius: 5px;
  color: var(--text-primary);
  font-size: 13px;
  font-family: var(--font-body);
  cursor: pointer;
  transition: background var(--transition-fast);
}

.dropdown-item:hover {
  background: var(--bg-sidebar);
}

.dropdown-item.danger {
  color: #ef4444;
}

.dropdown-item.danger:hover {
  background: #fef2f2;
}

/* 关闭菜单的透明遮罩 */
.menu-overlay {
  position: fixed;
  inset: 0;
  z-index: 99;
}

.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  color: var(--text-tertiary);
  font-size: 13px;
}

/* 重命名弹窗 */
.rename-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}

.rename-dialog {
  width: 340px;
  background: var(--bg-chat);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  padding: var(--space-xl);
}

.rename-dialog h3 {
  font-family: var(--font-heading);
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 var(--space-md);
}

.rename-input {
  width: 100%;
  padding: 10px 12px;
  background: var(--bg-sidebar);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-sm);
  font-size: 14px;
  font-family: var(--font-body);
  color: var(--text-primary);
  outline: none;
  transition: border-color var(--transition-fast);
}

.rename-input:focus {
  border-color: var(--border-active);
}

.rename-actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-sm);
  margin-top: var(--space-md);
}

.rename-cancel {
  padding: 6px 16px;
  background: none;
  border: 1px solid var(--border-card);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 13px;
  font-family: var(--font-body);
  cursor: pointer;
}

.rename-cancel:hover {
  background: var(--bg-hover);
}

.rename-confirm {
  padding: 6px 18px;
  background: var(--text-primary);
  border: none;
  border-radius: var(--radius-sm);
  color: var(--bg-chat);
  font-size: 13px;
  font-weight: 500;
  font-family: var(--font-body);
  cursor: pointer;
}

.rename-confirm:hover {
  opacity: 0.85;
}
</style>
