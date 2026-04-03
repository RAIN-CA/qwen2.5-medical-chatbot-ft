<template>
  <div class="app-shell">
    <SettingsPanel
      :open="sidebarOpen"
      :models="models"
      :selected-model="selectedModel"
      :max-new-tokens="maxNewTokens"
      :temperature="temperature"
      :top-p="topP"
      :use-rag="useRag"
      :rag-top-k="ragTopK"
      :rag-chunk-size="ragChunkSize"
      :rag-overlap="ragOverlap"
      :uploaded-files="uploadedFiles"
      :selected-files="selectedFiles"
      :pending-file="pendingFile"
      :uploading="uploading"
      :histories="histories"
      :current-history-id="currentHistoryId"
      :pending-delete-history="pendingDeleteHistory"
      @close="sidebarOpen = false"
      @new-chat="startNewChat"
      @open-history="openHistory"
      @delete-history="requestDeleteHistory"
      @confirm-delete-history="confirmDeleteHistory"
      @cancel-delete-history="cancelDeleteHistory"
      @update:selectedModel="selectedModel = $event"
      @update:maxNewTokens="maxNewTokens = $event"
      @update:temperature="temperature = $event"
      @update:topP="topP = $event"
      @update:useRag="useRag = $event"
      @update:ragTopK="ragTopK = $event"
      @update:ragChunkSize="ragChunkSize = $event"
      @update:ragOverlap="ragOverlap = $event"
      @file-change="handleFileChange"
      @upload-file="uploadSelectedFile"
      @toggle-file="toggleFileSelection"
      @delete-file="deleteUploadedFile"
    />

    <ChatWindow
      :messages="messages"
      :status-items="statusItems"
      :loading="loading"
      :query="query"
      :use-rag="useRag"
      :streaming-message-id="streamingMessageId"
      :format-message="formatMessage"
      :selected-domain="selectedDomain"
      :domain-options="domainOptions"
      :has-conversation="hasConversation"
      :selected-model-label="selectedModelLabel"
      :domain-models="domainModels"
      :selected-model="selectedModel"
      @toggle-sidebar="sidebarOpen = !sidebarOpen"
      @open-settings="sidebarOpen = true"
      @new-chat="startNewChat"
      @clear-chat="clearChat"
      @send-message="sendMessage"
      @select-domain="selectDomain"
      @select-model="selectModel"
      @select-example="applyExampleQuestion"
      @update:query="query = $event"
      @update:useRag="useRag = $event"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import SettingsPanel from './components/SettingsPanel.vue'
import ChatWindow from './components/ChatWindow.vue'

const API_BASE = 'http://127.0.0.1:5000'
const HISTORY_KEY = 'medchat_vue_histories_v3'

const domainOptions = [
  { key: 'medical', label: 'Medical', colorClass: 'medical' },
  { key: 'finance', label: 'Finance', colorClass: 'finance' },
  { key: 'legal', label: 'Legal', colorClass: 'legal' },
  { key: 'general', label: 'General', colorClass: 'general' },
  { key: 'multidomain', label: 'Multi-domain', colorClass: 'multidomain' },
]

const models = ref([])
const uploadedFiles = ref([])
const selectedFiles = ref([])

const sidebarOpen = ref(false)
const selectedDomain = ref('multidomain')
const selectedModel = ref('')
const useRag = ref(false)

const loading = ref(false)
const uploading = ref(false)
const pendingFile = ref(null)

const query = ref('')
const maxNewTokens = ref(160)
const temperature = ref(0.8)
const topP = ref(0.5)

const ragTopK = ref(4)
const ragChunkSize = ref(800)
const ragOverlap = ref(120)

const streamingMessageId = ref(null)

const histories = ref([])
const currentHistoryId = ref(null)

const messageCounter = ref(1)
const statusCounter = ref(1)

const initialAssistantMessage = () => ({
  id: messageCounter.value++,
  role: 'assistant',
  content: 'Hello. Select a domain to begin.',
})

const messages = ref([initialAssistantMessage()])
const statusItems = ref([])

const hasConversation = computed(() =>
  messages.value.some(msg => msg.role === 'user')
)

const domainModels = computed(() => {
  return models.value.filter(m => m.domain === selectedDomain.value || m.domain === 'all')
})

const selectedModelLabel = computed(() => {
  const item = models.value.find(m => m.key === selectedModel.value)
  return item?.label || 'No model selected'
})

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function formatMessage(text) {
  return escapeHtml(text).replace(/\n/g, '<br>')
}

function addStatus(id, text) {
  const existing = [...statusItems.value].reverse().find(
    item => item.id === id && item.state === 'active'
  )
  if (existing) {
    existing.text = text
    return
  }

  if (id.startsWith('rag_')) {
    statusItems.value.forEach((item) => {
      if (item.id.startsWith('rag_') && item.id !== id && item.state === 'active') {
        item.state = 'done'
        setTimeout(() => {
          statusItems.value = statusItems.value.filter(x => x.uid !== item.uid)
        }, 1400)
      }
    })
  }

  const uid = `${id}-${statusCounter.value++}`
  statusItems.value.push({
    uid,
    id,
    text,
    state: 'active',
  })
}

function completeStatus(id, text) {
  const target = [...statusItems.value].reverse().find(
    item => item.id === id && item.state === 'active'
  )

  if (!target) {
    const uid = `${id}-${statusCounter.value++}`
    statusItems.value.push({
      uid,
      id,
      text,
      state: 'done',
    })

    setTimeout(() => {
      statusItems.value = statusItems.value.filter(item => item.uid !== uid)
    }, 1400)
    return
  }

  target.text = text
  target.state = 'done'

  setTimeout(() => {
    statusItems.value = statusItems.value.filter(item => item.uid !== target.uid)
  }, 1400)
}

function resetStatuses() {
  statusItems.value = []
}

function completeAllActiveStatusByPrefix(prefix) {
  statusItems.value.forEach((item) => {
    if (item.id.startsWith(prefix) && item.state === 'active') {
      item.state = 'done'
      setTimeout(() => {
        statusItems.value = statusItems.value.filter(x => x.uid !== item.uid)
      }, 1400)
    }
  })
}

function choosePreferredModel(domainKey) {
  const domainCandidates = models.value.filter(
    m => m.domain === domainKey || m.domain === 'all'
  )

  const matchers = [
    m => m.family === 'qwen25' && m.size === '0.5b' && m.domain === domainKey && m.variant.includes('_ft'),
    m => m.family === 'qwen25' && m.size === '3b' && m.domain === domainKey && m.variant.includes('_ft'),
    m => m.family === 'qwen25' && m.size === '0.5b' && m.domain === 'multidomain' && m.variant.includes('balanced_multidomain_ft'),
    m => m.family === 'qwen25' && m.size === '3b' && m.domain === 'multidomain' && m.variant.includes('balanced_multidomain_ft'),
    m => m.family === 'qwen25' && m.size === '0.5b' && m.domain === 'all',
    m => m.family === 'qwen25' && m.size === '3b' && m.domain === 'all',
    m => m.domain === domainKey,
    m => m.domain === 'all',
  ]

  for (const matcher of matchers) {
    const found = domainCandidates.find(matcher)
    if (found) return found.key
  }

  return models.value[0]?.key || ''
}

function selectDomain(domainKey) {
  selectedDomain.value = domainKey
  const best = choosePreferredModel(domainKey)
  if (best) selectedModel.value = best
}

function selectModel(modelKey) {
  selectedModel.value = modelKey
}

function applyExampleQuestion(questionText) {
  query.value = questionText
}

async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/models`)
  const data = await res.json()
  models.value = data.models || []

  if (!selectedModel.value && models.value.length) {
    selectedModel.value = choosePreferredModel(selectedDomain.value)
  }
}

async function fetchFiles() {
  const res = await fetch(`${API_BASE}/api/rag/files`)
  const data = await res.json()
  uploadedFiles.value = data.files || []
}

function handleFileChange(event) {
  pendingFile.value = event.target.files?.[0] || null
}

async function uploadSelectedFile() {
  if (!pendingFile.value || uploading.value) return

  uploading.value = true
  try {
    const formData = new FormData()
    formData.append('file', pendingFile.value)

    const res = await fetch(`${API_BASE}/api/rag/files/upload`, {
      method: 'POST',
      body: formData,
    })

    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Upload failed')

    uploadedFiles.value = data.files || []
    if (!selectedFiles.value.includes(data.filename)) {
      selectedFiles.value.push(data.filename)
    }
    pendingFile.value = null
  } catch (err) {
    alert(`Upload failed: ${err.message}`)
  } finally {
    uploading.value = false
  }
}

function toggleFileSelection(filename) {
  if (selectedFiles.value.includes(filename)) {
    selectedFiles.value = selectedFiles.value.filter(f => f !== filename)
  } else {
    selectedFiles.value.push(filename)
  }
}

async function deleteUploadedFile(filename) {
  if (!confirm(`Delete file "${filename}"?`)) return

  try {
    const res = await fetch(`${API_BASE}/api/rag/files/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Delete failed')

    uploadedFiles.value = data.files || []
    selectedFiles.value = selectedFiles.value.filter(f => f !== filename)
  } catch (err) {
    alert(`Delete failed: ${err.message}`)
  }
}

async function startGeneration(payload) {
  const res = await fetch(`${API_BASE}/api/chat/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  const data = await res.json()
  if (!res.ok) {
    throw new Error(data.error || 'Failed to start generation')
  }
  return data.stream_id
}

function updateAssistantMessage(messageId, updater) {
  const idx = messages.value.findIndex(m => m.id === messageId)
  if (idx === -1) return
  const current = messages.value[idx]
  const next = typeof updater === 'function' ? updater(current) : updater
  messages.value.splice(idx, 1, next)
}

function consumeEventStream(streamId, assistantMessageId) {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`${API_BASE}/api/chat/events/${streamId}`)

    es.onmessage = (event) => {
      let payload
      try {
        payload = JSON.parse(event.data)
      } catch {
        return
      }

      const { type, data } = payload

      if (type === 'status') {
        if (data.id === 'prompt' || data.id === 'generate') {
          completeAllActiveStatusByPrefix('rag_')
        }
        addStatus(data.id, data.text)
      } else if (type === 'status_update') {
        addStatus(data.id, data.text)
      } else if (type === 'status_done') {
        completeStatus(data.id, data.text)
      } else if (type === 'chunk') {
        updateAssistantMessage(assistantMessageId, (msg) => ({
          ...msg,
          content: msg.content + data.text,
        }))
      } else if (type === 'done') {
        completeAllActiveStatusByPrefix('rag_')
        es.close()
        resolve()
      } else if (type === 'error') {
        updateAssistantMessage(assistantMessageId, (msg) => ({
          ...msg,
          content: `Error: ${data.message}`,
        }))
        es.close()
        reject(new Error(data.message))
      }
    }

    es.onerror = () => {
      es.close()
      resolve()
    }
  })
}

function buildHistoryTitle() {
  const firstUser = messages.value.find(m => m.role === 'user')
  if (!firstUser) return `New ${selectedDomain.value} chat`
  const text = firstUser.content.trim()
  return text.length > 42 ? `${text.slice(0, 42)}...` : text
}

function persistHistories() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(histories.value))
}

function saveCurrentConversation() {
  if (!messages.value.some(m => m.role === 'user')) return

  const item = {
    id: currentHistoryId.value || `chat-${Date.now()}`,
    title: buildHistoryTitle(),
    domain: selectedDomain.value,
    modelKey: selectedModel.value,
    useRag: useRag.value,
    maxNewTokens: maxNewTokens.value,
    temperature: temperature.value,
    topP: topP.value,
    ragTopK: ragTopK.value,
    ragChunkSize: ragChunkSize.value,
    ragOverlap: ragOverlap.value,
    selectedFiles: [...selectedFiles.value],
    updatedAt: Date.now(),
    messages: JSON.parse(JSON.stringify(messages.value)),
  }

  currentHistoryId.value = item.id
  const idx = histories.value.findIndex(h => h.id === item.id)
  if (idx >= 0) {
    histories.value[idx] = item
  } else {
    histories.value.unshift(item)
  }

  histories.value = [...histories.value].sort((a, b) => b.updatedAt - a.updatedAt)
  persistHistories()
}

function loadHistories() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    histories.value = raw ? JSON.parse(raw) : []
  } catch {
    histories.value = []
  }
}

function openHistory(item) {
  currentHistoryId.value = item.id
  selectedDomain.value = item.domain || 'medical'
  selectedModel.value = item.modelKey || choosePreferredModel(selectedDomain.value)
  useRag.value = !!item.useRag
  maxNewTokens.value = item.maxNewTokens ?? 160
  temperature.value = item.temperature ?? 0.8
  topP.value = item.topP ?? 0.5
  ragTopK.value = item.ragTopK ?? 4
  ragChunkSize.value = item.ragChunkSize ?? 800
  ragOverlap.value = item.ragOverlap ?? 120
  selectedFiles.value = item.selectedFiles || []
  messages.value = JSON.parse(JSON.stringify(item.messages || [initialAssistantMessage()]))
  query.value = ''
  resetStatuses()
  sidebarOpen.value = false
}

function clearChat() {
  messages.value = [initialAssistantMessage()]
  query.value = ''
  resetStatuses()
  streamingMessageId.value = null
  currentHistoryId.value = null
}

function startNewChat() {
  saveCurrentConversation()
  clearChat()
  sidebarOpen.value = false
}

function requestDeleteHistory(item) {
  pendingDeleteHistory.value = item
}

function cancelDeleteHistory() {
  pendingDeleteHistory.value = null
}

const pendingDeleteHistory = ref(null)

function confirmDeleteHistory() {
  if (!pendingDeleteHistory.value) return
  const deleteId = pendingDeleteHistory.value.id

  histories.value = histories.value.filter(h => h.id !== deleteId)

  if (currentHistoryId.value === deleteId) {
    clearChat()
  }

  persistHistories()
  pendingDeleteHistory.value = null
}

async function sendMessage() {
  const text = query.value.trim()
  if (!text || loading.value) return

  resetStatuses()

  const userMsg = {
    id: messageCounter.value++,
    role: 'user',
    content: text,
  }
  messages.value.push(userMsg)

  const assistantMsg = {
    id: messageCounter.value++,
    role: 'assistant',
    content: '',
  }
  messages.value.push(assistantMsg)

  query.value = ''
  loading.value = true
  streamingMessageId.value = assistantMsg.id

  try {
    const streamId = await startGeneration({
      model_key: selectedModel.value,
      query: text,
      max_new_tokens: maxNewTokens.value,
      temperature: temperature.value,
      top_p: topP.value,
      use_rag: useRag.value,
      selected_files: selectedFiles.value,
      rag_top_k: ragTopK.value,
      rag_chunk_size: ragChunkSize.value,
      rag_overlap: ragOverlap.value,
    })

    await consumeEventStream(streamId, assistantMsg.id)

    const finalMsg = messages.value.find(m => m.id === assistantMsg.id)
    if (!finalMsg?.content.trim()) {
      updateAssistantMessage(assistantMsg.id, (msg) => ({
        ...msg,
        content: 'No response returned.',
      }))
    }
    saveCurrentConversation()
  } catch (err) {
    updateAssistantMessage(assistantMsg.id, (msg) => ({
      ...msg,
      content: `Error: ${err.message}`,
    }))
  } finally {
    loading.value = false
    streamingMessageId.value = null
    saveCurrentConversation()
  }
}

onMounted(async () => {
  loadHistories()
  await fetchModels()
  await fetchFiles()
})
</script>

<style>
:root {
  --bg: #0a0b0e;
  --bg2: #111318;
  --panel: rgba(20, 23, 30, 0.86);
  --panel-strong: rgba(18, 21, 28, 0.96);
  --border: rgba(255,255,255,0.08);
  --text: #f5f7fb;
  --muted: #9aa4b2;
  --primary: #5b7cff;
  --primary2: #7b5cff;
  --assistant-bg: #171b23;
  --assistant-border: rgba(255,255,255,0.08);
  --shadow: 0 20px 60px rgba(0,0,0,0.38);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background: radial-gradient(circle at top, #171922 0%, #0c0e13 40%, #08090c 100%);
}

* {
  box-sizing: border-box;
}

html, body, #app {
  margin: 0;
  min-height: 100%;
  background: var(--bg);
  color: var(--text);
}

body {
  overflow: hidden;
}

button, input, textarea, select {
  font: inherit;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  backdrop-filter: blur(14px);
}

.app-shell {
  min-height: 100vh;
  width: 100%;
  overflow: hidden;
}
</style>
