import { ref, watch, onUnmounted, type Ref } from 'vue'

export interface ChatMessage {
  role: 'user' | 'assistant' | 'status'
  content: string
  stage?: string
  sources?: SourceInfo[]
  timestamp: number
}

export interface SourceInfo {
  index: number
  doc_name: string
  doc_type: string
  snippet: string
}

export function useWebSocket(sessionId: Ref<string>) {
  const messages = ref<ChatMessage[]>([])
  const isConnected = ref(false)
  const isStreaming = ref(false)
  let ws: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  function connect() {
    disconnect()

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${location.host}/ws/chat?session_id=${sessionId.value}`

    ws = new WebSocket(url)

    ws.onopen = () => {
      isConnected.value = true
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      switch (data.type) {
        case 'status':
          isStreaming.value = true
          messages.value.push({
            role: 'status',
            content: '',
            stage: data.stage,
            timestamp: Date.now()
          })
          break

        case 'answer':
          isStreaming.value = false
          const lastIdx = messages.value.length - 1
          if (lastIdx >= 0 && messages.value[lastIdx].role === 'status') {
            messages.value.pop()
          }
          messages.value.push({
            role: 'assistant',
            content: data.content,
            sources: data.sources || [],
            timestamp: Date.now()
          })
          break

        case 'error':
          isStreaming.value = false
          messages.value.push({
            role: 'assistant',
            content: `❌ ${data.message}`,
            timestamp: Date.now()
          })
          break
      }
    }

    ws.onclose = () => {
      isConnected.value = false
      reconnectTimer = setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      ws?.close()
    }
  }

  function send(content: string) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    messages.value.push({
      role: 'user',
      content,
      timestamp: Date.now()
    })

    ws.send(JSON.stringify({
      type: 'query',
      content
    }))
  }

  function clearMessages() {
    messages.value = []
  }

  function abort() {
    // 去掉残留状态消息
    const lastIdx = messages.value.length - 1
    if (lastIdx >= 0 && messages.value[lastIdx].role === 'status') {
      messages.value.pop()
    }
    if (ws) {
      ws.onclose = null
      ws.close()
      ws = null
    }
    isStreaming.value = false
    connect()
  }

  function disconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    if (ws) {
      ws.onclose = null
      ws.close()
      ws = null
    }
  }

  // 会话切换 → 断开旧连接 + 重连新 session_id
  watch(sessionId, () => {
    messages.value = []
    isStreaming.value = false
    connect()
  })

  // 初始连接
  connect()

  onUnmounted(disconnect)

  return { messages, isConnected, isStreaming, send, disconnect, clearMessages, abort }
}
