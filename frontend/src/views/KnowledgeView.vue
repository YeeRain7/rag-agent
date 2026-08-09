<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useTheme } from '../composables/useTheme'
import { useKnowledge } from '../composables/useKnowledge'

useTheme()

const router = useRouter()
import DocumentList from '../components/knowledge/DocumentList.vue'
import UploadDialog from '../components/knowledge/UploadDialog.vue'

const { documents, loading, stats, fetchDocuments, fetchStats, deleteDocument } = useKnowledge()
const uploadRef = ref<InstanceType<typeof UploadDialog>>()

onMounted(() => {
  fetchDocuments()
  fetchStats()
})

async function handleDelete(filename: string) {
  const ok = await deleteDocument(filename)
  if (!ok) {
    alert('删除失败，请稍后重试')
  }
}

function handleUploaded() {
  fetchDocuments()
  fetchStats()
}
</script>

<template>
  <div class="knowledge-view">
    <div class="knowledge-inner">
    <div class="page-header">
      <div>
        <router-link to="/chat" class="back-btn" aria-label="返回对话">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 18 9 12 15 6"/></svg>
        </router-link>
        <h1>知识库</h1>
        <p class="subtitle">管理文档库，构建你的专属检索范围</p>
      </div>
      <button class="upload-btn" @click="uploadRef?.open()">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        上传文档
      </button>
    </div>

    <!-- 统计卡片 -->
    <div class="stats">
      <div class="stat-card">
        <span class="stat-num">{{ stats.document_count }}</span>
        <span class="stat-label">文档总数</span>
      </div>
      <div class="stat-card">
        <span class="stat-num">{{ stats.total_chunks }}</span>
        <span class="stat-label">知识片段</span>
      </div>
      <div class="stat-card hint-card">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
        <span class="hint-text">支持 PDF、Word、Markdown、TXT 格式</span>
      </div>
    </div>

    <!-- 文档列表 -->
    <div class="doc-header">
      <h2>文档列表</h2>
      <span class="count">{{ documents.length }} 个文件</span>
    </div>
    <div class="doc-section glass-card">
      <div class="table-header">
        <span class="col-name">文件名</span>
        <span class="col-size">大小</span>
        <span class="col-action"></span>
      </div>
      <div class="doc-scroll">
        <DocumentList
          :documents="documents"
          :loading="loading"
          @delete="handleDelete"
        />
      </div>
    </div>

    <UploadDialog ref="uploadRef" @uploaded="handleUploaded" />
    </div>
  </div>
</template>

<style scoped>
.knowledge-view {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.knowledge-inner {
  max-width: 900px;
  margin: 0 auto;
  padding: var(--space-2xl) var(--space-xl) var(--space-lg);
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: var(--space-xl);
}

.back-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background: var(--bg-sidebar);
  color: var(--text-secondary);
  text-decoration: none;
  margin-bottom: 14px;
  transition: all var(--transition-fast);
}

.back-btn:hover {
  background: var(--bg-active);
  color: var(--text-primary);
}

.page-header h1 {
  font-family: var(--font-heading);
  font-size: 26px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
  letter-spacing: -0.4px;
}

.subtitle {
  font-size: 14px;
  color: var(--text-tertiary);
  margin-top: 6px;
}

.upload-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  background: var(--text-primary);
  border: none;
  border-radius: var(--radius-sm);
  color: var(--bg-chat);
  font-size: 14px;
  font-weight: 500;
  font-family: var(--font-body);
  cursor: pointer;
  transition: opacity var(--transition-fast);
}

.upload-btn:hover { opacity: 0.88; }

/* === 统计卡片 === */
.stats {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: var(--space-md);
  margin-bottom: var(--space-xl);
}

.stat-card {
  padding: 24px;
  background: var(--bg-sidebar);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-num {
  font-size: 34px;
  font-weight: 600;
  color: var(--text-primary);
  font-family: var(--font-mono);
  letter-spacing: -1.5px;
}

.stat-label {
  font-size: 13px;
  color: var(--text-tertiary);
}

/* 提示卡片 */
.hint-card {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 10px;
  color: var(--text-tertiary);
}

.hint-text {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.4;
}

/* === 文档区 === */
.doc-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--space-md);
  flex-shrink: 0;
}

.doc-header h2 {
  font-family: var(--font-heading);
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.count {
  font-size: 12px;
  color: var(--text-tertiary);
}

.doc-section {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.table-header {
  display: flex;
  align-items: center;
  padding: var(--space-md) var(--space-2xl) 12px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: var(--text-tertiary);
  flex-shrink: 0;
  border-bottom: 1px solid var(--border-subtle);
}

.col-name { flex: 1; padding-right: var(--space-md); }
.col-size { width: 72px; text-align: right; padding-right: var(--space-md); }
.col-action { width: 40px; }

.doc-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 0 var(--space-2xl) var(--space-md);
}
</style>
