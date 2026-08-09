import { ref } from 'vue'

export interface DocFile {
  filename: string
  size_kb: number
  modified: string
}

export function useKnowledge() {
  const documents = ref<DocFile[]>([])
  const loading = ref(false)
  const stats = ref({ document_count: 0, total_chunks: 0 })

  async function fetchDocuments() {
    loading.value = true
    try {
      const res = await fetch('/api/knowledge/documents')
      const data = await res.json()
      documents.value = data.documents || []
    } finally {
      loading.value = false
    }
  }

  async function fetchStats() {
    const res = await fetch('/api/knowledge/stats')
    if (res.ok) {
      stats.value = await res.json()
    }
  }

  async function uploadFile(file: File): Promise<{ success: boolean; message: string }> {
    const form = new FormData()
    form.append('file', file)

    const res = await fetch('/api/knowledge/upload', {
      method: 'POST',
      body: form
    })

    if (!res.ok) {
      const err = await res.json()
      return { success: false, message: err.detail || '上传失败' }
    }

    await Promise.all([fetchDocuments(), fetchStats()])
    return await res.json()
  }

  async function deleteDocument(filename: string): Promise<boolean> {
    const res = await fetch(`/api/knowledge/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    })

    if (res.ok) {
      await Promise.all([fetchDocuments(), fetchStats()])
    }
    return res.ok
  }

  return { documents, loading, stats, fetchDocuments, fetchStats, uploadFile, deleteDocument }
}
