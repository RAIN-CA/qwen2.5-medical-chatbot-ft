<template>
  <div class="app-shell page-enter">
    <SettingsPanel
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
      :example-questions="exampleQuestions"
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
      @pick-example="query = $event"
    />

    <ChatWindow
      :messages="messages"
      :status-items="statusItems"
      :loading="loading"
      :query="query"
      :streaming-message-id="streamingMessageId"
      :format-message="formatMessage"
      @clear-chat="clearChat"
      @send-message="sendMessage"
      @update:query="query = $event"
    />
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import SettingsPanel from './components/SettingsPanel.vue'
import ChatWindow from './components/ChatWindow.vue'

const API_BASE = 'http://127.0.0.1:5000'

const models = ref([])
const uploadedFiles = ref([])
const selectedFiles = ref([])
const selectedModel = ref('ft_3b')
const useRag = ref(false)

const loading = ref(false)
const uploading = ref(false)
const pendingFile = ref(null)

const query = ref('')
const maxNewTokens = ref(384)
const temperature = ref(0.2)
const topP = ref(0.85)

const ragTopK = ref(4)
const ragChunkSize = ref(800)
const ragOverlap = ref(120)

const streamingMessageId = ref(null)

const exampleQuestions = [
  'What are the common symptoms of diabetes?',
  'What is hypertension?',
  'What is the difference between CT and MRI?',
  'What are common risk factors for heart disease?',
  'What is anemia?',
]

const messageCounter = ref(1)
const statusCounter = ref(1)

const messages = ref([
  {
    id: messageCounter.value++,
    role: 'assistant',
    content: 'Hello. Ask a medical knowledge question to start the demo.',
  },
])

const statusItems = ref([])

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

  const uid = `${id}-${statusCounter.value++}`
  statusItems.value.push({
    uid,
    id,
    text,
    state: 'active',
  })
}

function completeStatus(id, text) {
  const target = [...statusItems.value].reverse().find(item => item.id === id && item.state === 'active')
  if (!target) {
    statusItems.value.push({
      uid: `${id}-${statusCounter.value++}`,
      id,
      text,
      state: 'done',
    })
    return
  }

  target.text = text
  target.state = 'done'

  setTimeout(() => {
    statusItems.value = statusItems.value.filter(item => item.uid !== target.uid)
  }, 900)
}

function resetStatuses() {
  statusItems.value = []
}

async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/models`)
  const data = await res.json()
  models.value = data.models
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

function consumeEventStream(streamId, assistantMsg) {
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
        addStatus(data.id, data.text)
      } else if (type === 'status_update') {
        addStatus(data.id, data.text)
      } else if (type === 'status_done') {
        completeStatus(data.id, data.text)
      } else if (type === 'chunk') {
        assistantMsg.content += data.text
      } else if (type === 'done') {
        es.close()
        resolve()
      } else if (type === 'error') {
        assistantMsg.content = `Error: ${data.message}`
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

async function sendMessage() {
  const text = query.value.trim()
  if (!text || loading.value) return

  loading.value = true
  resetStatuses()

  messages.value.push({
    id: messageCounter.value++,
    role: 'user',
    content: text,
  })

  messages.value.push({
    id: messageCounter.value++,
    role: 'assistant',
    content: '',
  })
  const assistantMsg = messages.value[messages.value.length - 1]
  streamingMessageId.value = assistantMsg.id

  query.value = ''

  try {
    const streamId = await startGeneration({
      model_key: selectedModel.value,
      query: text,
      max_new_tokens: Number(maxNewTokens.value),
      temperature: Number(temperature.value),
      top_p: Number(topP.value),
      use_rag: useRag.value,
      selected_files: selectedFiles.value,
      rag_top_k: Number(ragTopK.value),
      rag_chunk_size: Number(ragChunkSize.value),
      rag_overlap: Number(ragOverlap.value),
    })

    await consumeEventStream(streamId, assistantMsg)
  } catch (err) {
    assistantMsg.content = `Error: ${err.message || 'Unknown error'}`
  } finally {
    loading.value = false
    streamingMessageId.value = null
    await nextTick()
  }
}

function clearChat() {
  if (loading.value) return
  messages.value = [
    {
      id: messageCounter.value++,
      role: 'assistant',
      content: 'Hello. Ask a medical knowledge question to start the demo.',
    },
  ]
  resetStatuses()
}

onMounted(async () => {
  await fetchModels()
  await fetchFiles()
})
</script>

<style>
:root {
  --bg: #f4f7fb;
  --panel: rgba(255, 255, 255, 0.88);
  --text: #172033;
  --muted: #667085;
  --primary: #2563eb;
  --primary2: #4f46e5;
  --assistant-bg: #f8fafc;
  --assistant-border: #e2e8f0;
}

* { box-sizing: border-box; }

html, body, #app {
  margin: 0;
  min-height: 100%;
  font-family: Inter, Arial, sans-serif;
}

body {
  color: var(--text);
  background:
    radial-gradient(circle at top left, #dbeafe, transparent 30%),
    radial-gradient(circle at bottom right, #e0e7ff, transparent 30%),
    var(--bg);
  overflow: hidden;
}

.app-shell {
  min-height: 100vh;
  display: flex;
  gap: 20px;
  padding: 20px;
}

.panel {
  background: var(--panel);
  border: 1px solid rgba(255,255,255,0.72);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow: 0 16px 40px rgba(37, 99, 235, 0.08);
}

.page-enter {
  animation: pageFade 0.55s ease-out both;
}

.enter-up {
  opacity: 0;
  transform: translateY(18px);
  animation: enterUp 0.6s ease-out forwards;
}

.delay-1 { animation-delay: 0.05s; }
.delay-2 { animation-delay: 0.14s; }
.delay-3 { animation-delay: 0.22s; }

@keyframes pageFade {
  from { opacity: 0; filter: blur(6px); }
  to { opacity: 1; filter: blur(0); }
}

@keyframes enterUp {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 1024px) {
  body {
    overflow: auto;
  }

  .app-shell {
    flex-direction: column;
    padding: 14px;
  }
}
</style>
