<template>
  <div class="app-shell page-enter">
    <aside class="sidebar panel enter-up delay-1">
      <div class="brand">
        <div class="icon">🩺</div>
        <div>
          <h2>Medical Chatbot</h2>
          <p>Vue frontend for the coursework demo</p>
        </div>
      </div>

      <div class="section">
        <label>Choose model</label>
        <select v-model="selectedModel">
          <option v-for="model in models" :key="model.key" :value="model.key">
            {{ model.label }}
          </option>
        </select>
      </div>

      <div class="section">
        <label>Generation Settings</label>
        <div class="form-row">
          <span>Max new tokens</span>
          <span>{{ maxNewTokens }}</span>
        </div>
        <input v-model="maxNewTokens" type="range" min="128" max="640" step="32" />

        <div class="form-row">
          <span>Temperature</span>
          <span>{{ temperature.toFixed(2) }}</span>
        </div>
        <input v-model="temperature" type="range" min="0.1" max="1" step="0.05" />

        <div class="form-row">
          <span>Top-p</span>
          <span>{{ topP.toFixed(2) }}</span>
        </div>
        <input v-model="topP" type="range" min="0.5" max="1" step="0.05" />
      </div>

      <div class="section">
        <label class="toggle-row">
          <input type="checkbox" v-model="useRag" />
          <span>Enable RAG retrieval</span>
        </label>
      </div>

      <div class="section">
        <label>Upload document</label>
        <input type="file" @change="handleFileChange" />
        <p class="upload-hint">
          Supported formats: TXT, MD, PDF, DOCX. Files are normalized into text chunks for retrieval.
        </p>
        <button class="primary-btn small-btn" @click="uploadSelectedFile" :disabled="!pendingFile || uploading">
          {{ uploading ? 'Uploading...' : 'Upload File' }}
        </button>
      </div>

      <div class="section">
        <label>Available documents</label>
        <div class="file-list">
          <label v-for="f in uploadedFiles" :key="f.name" class="file-card">
            <div class="file-left">
              <input type="checkbox" :value="f.name" v-model="selectedFiles" />
              <div>
                <div class="file-name">{{ f.name }}</div>
                <div class="file-meta">{{ f.type.toUpperCase() }} · {{ f.size_kb }} KB</div>
              </div>
            </div>
          </label>
        </div>
      </div>

      <div class="section">
        <label>Example questions</label>
        <button
          v-for="q in exampleQuestions"
          :key="q"
          class="example-btn"
          @click="query = q"
        >
          {{ q }}
        </button>
      </div>

      <div class="notice">
        <strong>Academic Use Only</strong>
        <p>This demo is for coursework only and does not provide diagnosis or treatment advice.</p>
      </div>
    </aside>

    <main class="main enter-up delay-2">
      <header class="topbar panel">
        <div>
          <h1>Medical Chatbot Demo</h1>
          <p>Vue + SSE Python backend + optional RAG</p>
        </div>
        <button class="ghost-btn" @click="clearChat" :disabled="loading">Clear Chat</button>
      </header>

      <section class="status-panel panel enter-up delay-3" v-if="loading || statusItems.length">
        <div class="status-header">
          <span class="thinking-dot"></span>
          <span>{{ loading ? 'System status' : 'Recent progress' }}</span>
        </div>

        <transition-group name="status" tag="div" class="status-list">
          <div
            v-for="item in statusItems"
            :key="item.uid"
            class="status-item"
            :class="item.state"
          >
            <span class="status-icon">
              <span v-if="item.state === 'active'" class="spinner-mini"></span>
              <span v-else>✓</span>
            </span>
            <span>{{ item.text }}</span>
          </div>
        </transition-group>
      </section>

      <section ref="chatBoxRef" class="chat-box panel">
        <transition-group name="bubble" tag="div">
          <div
            v-for="msg in messages"
            :key="msg.id"
            class="message"
            :class="msg.role"
          >
            <div v-if="msg.role === 'assistant'" class="avatar assistant-avatar">AI</div>

            <div class="bubble" :class="msg.role === 'assistant' ? 'assistant-bubble' : 'user-bubble'">
              <span v-html="formatMessage(msg.content)"></span>
              <span
                v-if="loading && streamingMessageId === msg.id && msg.role === 'assistant'"
                class="typing-cursor"
              >
                ▌
              </span>
            </div>

            <div v-if="msg.role === 'user'" class="avatar user-avatar">You</div>
          </div>
        </transition-group>
      </section>

      <section class="composer panel enter-up delay-3">
        <textarea
          v-model="query"
          placeholder="Enter your medical question..."
          @keydown.enter.exact.prevent="sendMessage"
          :disabled="loading"
        />
        <div class="actions">
          <button class="primary-btn" @click="sendMessage" :disabled="loading">
            {{ loading ? 'Generating response...' : 'Generate Response' }}
          </button>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'

const API_BASE = 'http://127.0.0.1:5000'

const models = ref([])
const uploadedFiles = ref([])
const selectedFiles = ref([])
const selectedModel = ref('ft_3b')
const useRag = ref(false)

const loading = ref(false)
const uploading = ref(false)
const pendingFile = ref(null)

const query = ref('What are the common symptoms of diabetes?')
const maxNewTokens = ref(384)
const temperature = ref(0.2)
const topP = ref(0.85)

const chatBoxRef = ref(null)
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

let pendingChunkBuffer = ''
let flushTimer = null
let scrollScheduled = false

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function formatMessage(text) {
  return escapeHtml(text).replace(/\n/g, '<br>')
}

function scrollToBottom() {
  const el = chatBoxRef.value
  if (el) el.scrollTop = el.scrollHeight
}

function scheduleScroll() {
  if (scrollScheduled) return
  scrollScheduled = true
  requestAnimationFrame(() => {
    scrollScheduled = false
    scrollToBottom()
  })
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

function startFlushLoop(assistantMsg) {
  stopFlushLoop()

  flushTimer = setInterval(() => {
    if (!pendingChunkBuffer) return
    assistantMsg.content += pendingChunkBuffer
    pendingChunkBuffer = ''
    scheduleScroll()
  }, 50)
}

function stopFlushLoop() {
  if (flushTimer) {
    clearInterval(flushTimer)
    flushTimer = null
  }
}

async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/models`)
  const data = await res.json()
  models.value = data.models
}

async function fetchFiles() {
  const res = await fetch(`${API_BASE}/api/files`)
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

    const res = await fetch(`${API_BASE}/api/upload`, {
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
    let doneReceived = false

    function tryFinish() {
      if (doneReceived && !pendingChunkBuffer) {
        stopFlushLoop()
        es.close()
        resolve()
      }
    }

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
        pendingChunkBuffer += data.text
      } else if (type === 'done') {
        doneReceived = true
        setTimeout(tryFinish, 60)
      } else if (type === 'error') {
        stopFlushLoop()
        assistantMsg.content = `Error: ${data.message}`
        es.close()
        reject(new Error(data.message))
      }
    }

    es.onerror = () => {
      if (doneReceived) {
        stopFlushLoop()
        es.close()
        resolve()
      }
    }

    const watcher = setInterval(() => {
      if (pendingChunkBuffer) {
        // let the flush loop handle UI updates
      }
      if (doneReceived && !pendingChunkBuffer) {
        clearInterval(watcher)
        tryFinish()
      }
    }, 50)
  })
}

async function sendMessage() {
  const text = query.value.trim()
  if (!text || loading.value) return

  loading.value = true
  resetStatuses()
  pendingChunkBuffer = ''
  stopFlushLoop()

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
  startFlushLoop(assistantMsg)

  query.value = ''
  scrollToBottom()

  try {
    const streamId = await startGeneration({
      model_key: selectedModel.value,
      query: text,
      max_new_tokens: Number(maxNewTokens.value),
      temperature: Number(temperature.value),
      top_p: Number(topP.value),
      use_rag: useRag.value,
      selected_files: selectedFiles.value,
    })

    await consumeEventStream(streamId, assistantMsg)
  } catch (err) {
    assistantMsg.content = `Error: ${err.message || 'Unknown error'}`
  } finally {
    stopFlushLoop()
    loading.value = false
    streamingMessageId.value = null
    scrollToBottom()
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
  scrollToBottom()
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

.sidebar {
  width: 340px;
  border-radius: 24px;
  padding: 24px;
  display: flex;
  flex-direction: column;
}

.brand {
  display: flex;
  gap: 14px;
  align-items: center;
  margin-bottom: 24px;
}

.icon {
  width: 48px;
  height: 48px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #dbeafe, #e0e7ff);
  font-size: 22px;
}

.brand h2 { margin: 0; }
.brand p {
  margin: 6px 0 0 0;
  color: var(--muted);
  font-size: 14px;
}

.section { margin-bottom: 22px; }

.section label {
  display: block;
  margin-bottom: 10px;
  font-weight: 700;
  font-size: 14px;
}

.toggle-row {
  display: flex !important;
  align-items: center;
  gap: 10px;
}

.form-row {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: var(--muted);
  margin: 8px 0 4px;
}

select, textarea, input[type="range"], input[type="file"] {
  width: 100%;
}

select, textarea {
  border-radius: 16px;
  border: 1px solid #d8dee9;
  padding: 12px 14px;
  font-size: 14px;
  background: rgba(255,255,255,0.96);
}

textarea {
  min-height: 110px;
  resize: vertical;
}

.upload-hint {
  margin: 8px 0 0 0;
  font-size: 12px;
  color: var(--muted);
  line-height: 1.45;
}

.small-btn {
  margin-top: 10px;
  width: 100%;
}

.file-list {
  max-height: 180px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-right: 4px;
}

.file-card {
  display: block;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(148,163,184,0.18);
  cursor: pointer;
}

.file-left {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}

.file-name {
  font-size: 14px;
  font-weight: 600;
  word-break: break-all;
}

.file-meta {
  font-size: 12px;
  color: var(--muted);
  margin-top: 4px;
}

.example-btn {
  width: 100%;
  margin-bottom: 10px;
  text-align: left;
  border: 1px solid rgba(99,102,241,0.12);
  background: rgba(238,242,255,0.9);
  padding: 11px 13px;
  border-radius: 14px;
  cursor: pointer;
  transition: transform 0.16s ease, box-shadow 0.16s ease;
}

.example-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 18px rgba(99,102,241,0.12);
}

.notice {
  margin-top: auto;
  padding: 14px 16px;
  border-radius: 18px;
  background: rgba(255,247,237,0.95);
  border: 1px solid rgba(253,186,116,0.5);
  font-size: 14px;
  line-height: 1.55;
}

.notice p { margin-bottom: 0; }

.main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.topbar, .status-panel, .chat-box, .composer {
  border-radius: 24px;
}

.topbar {
  padding: 20px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.topbar h1 { margin: 0; }
.topbar p {
  margin: 6px 0 0 0;
  color: var(--muted);
}

.status-panel {
  padding: 14px 18px;
}

.status-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  margin-bottom: 10px;
}

.thinking-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(135deg, var(--primary), var(--primary2));
  animation: pulse 1.1s infinite;
}

.status-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 14px;
  font-size: 14px;
}

.status-item.active {
  background: rgba(219, 234, 254, 0.55);
  color: #1d4ed8;
}

.status-item.done {
  background: rgba(236, 253, 245, 0.75);
  color: #047857;
}

.status-icon {
  width: 18px;
  display: flex;
  justify-content: center;
}

.spinner-mini {
  width: 14px;
  height: 14px;
  border-radius: 999px;
  border: 2px solid rgba(37, 99, 235, 0.18);
  border-top-color: var(--primary);
  animation: spin 0.9s linear infinite;
}

.chat-box {
  flex: 1;
  min-height: 480px;
  max-height: calc(100vh - 320px);
  overflow-y: auto;
  padding: 24px;
  scroll-behavior: smooth;
}

.chat-box::-webkit-scrollbar {
  width: 10px;
}

.chat-box::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.35);
  border-radius: 999px;
}

.message {
  display: flex;
  gap: 10px;
  margin-bottom: 18px;
  align-items: flex-end;
}

.message.user {
  justify-content: flex-end;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
}

.assistant-avatar {
  background: linear-gradient(135deg, #dbeafe, #e0e7ff);
  color: #1d4ed8;
}

.user-avatar {
  background: linear-gradient(135deg, var(--primary), var(--primary2));
  color: white;
}

.bubble {
  max-width: 76%;
  padding: 14px 16px;
  border-radius: 20px;
  line-height: 1.65;
  white-space: pre-wrap;
  word-wrap: break-word;
  position: relative;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.assistant-bubble {
  background: var(--assistant-bg);
  border: 1px solid var(--assistant-border);
  border-bottom-left-radius: 8px;
}

.user-bubble {
  background: linear-gradient(135deg, var(--primary), var(--primary2));
  color: white;
  border-bottom-right-radius: 8px;
}

.assistant-bubble::before {
  content: "";
  position: absolute;
  left: -7px;
  bottom: 8px;
  width: 14px;
  height: 14px;
  background: var(--assistant-bg);
  border-left: 1px solid var(--assistant-border);
  border-bottom: 1px solid var(--assistant-border);
  transform: rotate(45deg);
}

.user-bubble::before {
  content: "";
  position: absolute;
  right: -7px;
  bottom: 8px;
  width: 14px;
  height: 14px;
  background: #3b5bfa;
  transform: rotate(45deg);
}

.composer {
  padding: 16px;
}

.actions {
  margin-top: 14px;
  display: flex;
  justify-content: flex-end;
}

.primary-btn, .ghost-btn {
  border: none;
  border-radius: 14px;
  padding: 12px 18px;
  cursor: pointer;
  font-weight: 600;
  transition: transform 0.16s ease, opacity 0.16s ease;
}

.primary-btn {
  background: linear-gradient(135deg, var(--primary), var(--primary2));
  color: white;
}

.ghost-btn {
  background: white;
}

.primary-btn:hover, .ghost-btn:hover {
  transform: translateY(-1px);
}

.primary-btn:disabled, .ghost-btn:disabled {
  opacity: 0.65;
  cursor: not-allowed;
  transform: none;
}

.typing-cursor {
  display: inline-block;
  margin-left: 2px;
  animation: blink 0.9s infinite;
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

.bubble-enter-active, .bubble-leave-active {
  transition: all 0.28s ease;
}
.bubble-enter-from {
  opacity: 0;
  transform: translateY(8px) scale(0.985);
}
.bubble-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

.status-enter-active, .status-leave-active {
  transition: all 0.3s ease;
}
.status-enter-from {
  opacity: 0;
  transform: translateY(8px);
}
.status-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

@keyframes pageFade {
  from { opacity: 0; filter: blur(6px); }
  to { opacity: 1; filter: blur(0); }
}

@keyframes enterUp {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes blink {
  0%, 49% { opacity: 1; }
  50%, 100% { opacity: 0; }
}

@keyframes spin {
  from { transform: rotate(0); }
  to { transform: rotate(360deg); }
}

@keyframes pulse {
  0% { transform: scale(1); opacity: 0.8; }
  50% { transform: scale(1.25); opacity: 1; }
  100% { transform: scale(1); opacity: 0.8; }
}

@media (max-width: 1024px) {
  body {
    overflow: auto;
  }

  .app-shell {
    flex-direction: column;
    padding: 14px;
  }

  .sidebar {
    width: 100%;
  }

  .chat-box {
    max-height: none;
    min-height: 420px;
  }

  .bubble {
    max-width: 88%;
  }
}
</style>
