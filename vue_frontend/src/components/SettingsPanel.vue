<template>
  <transition name="drawer-overlay">
    <div v-if="open" class="drawer-backdrop" @click.self="$emit('close')">
      <transition name="drawer-slide">
        <aside v-if="open" class="drawer panel">
          <div class="drawer-header">
            <div class="brand">
              <div class="icon">◉</div>
              <div>
                <h2>Control Center</h2>
                <p>Models, RAG, documents and chat history</p>
              </div>
            </div>
            <button class="close-btn" @click="$emit('close')">✕</button>
          </div>

          <div class="quick-actions">
            <button class="quick-btn" :class="{ active: activeTab === 'settings' }" @click="activeTab = 'settings'">
              Settings
            </button>
            <button class="quick-btn" :class="{ active: activeTab === 'docs' }" @click="activeTab = 'docs'">
              Documents
            </button>
            <button class="quick-btn new-chat-btn" @click="$emit('new-chat')">
              New Chat
            </button>
          </div>

          <div class="drawer-body">
            <div v-if="activeTab === 'settings'" class="tab-panel">
              <div class="section">
                <label>Choose model</label>
                <div class="custom-select">
                  <button class="custom-select-trigger" @click="toggleModelMenu">
                    <span>{{ selectedModelLabel }}</span>
                    <span class="caret">⌄</span>
                  </button>

                  <transition name="menu-fade">
                    <div v-if="modelMenuOpen" class="custom-select-menu">
                      <button
                        v-for="model in models"
                        :key="model.key"
                        class="custom-select-item"
                        :class="{ active: selectedModel === model.key }"
                        @click="pickModel(model.key)"
                      >
                        {{ model.label }}
                      </button>
                    </div>
                  </transition>
                </div>
              </div>

              <div class="section">
                <label>Generation Settings</label>

                <div class="form-row">
                  <span>Max new tokens</span>
                  <span>{{ maxNewTokens }}</span>
                </div>
                <input :value="maxNewTokens" @input="$emit('update:maxNewTokens', Number($event.target.value))" type="range" min="64" max="320" step="16" />

                <div class="form-row">
                  <span>Temperature</span>
                  <span>{{ Number(temperature).toFixed(2) }}</span>
                </div>
                <input :value="temperature" @input="$emit('update:temperature', Number($event.target.value))" type="range" min="0.1" max="1" step="0.05" />

                <div class="form-row">
                  <span>Top-p</span>
                  <span>{{ Number(topP).toFixed(2) }}</span>
                </div>
                <input :value="topP" @input="$emit('update:topP', Number($event.target.value))" type="range" min="0.5" max="1" step="0.05" />
              </div>

              <div class="section">
                <label>RAG Settings</label>

                <div class="toggle-box">
                  <button
                    class="rag-state-btn"
                    :class="{ active: useRag }"
                    @click="$emit('update:useRag', !useRag)"
                  >
                    {{ useRag ? 'RAG ON' : 'RAG OFF' }}
                  </button>
                </div>

                <div class="form-row">
                  <span>Top-k</span>
                  <span>{{ ragTopK }}</span>
                </div>
                <input :value="ragTopK" @input="$emit('update:ragTopK', Number($event.target.value))" type="range" min="1" max="8" step="1" />

                <div class="form-row">
                  <span>Chunk size</span>
                  <span>{{ ragChunkSize }}</span>
                </div>
                <input :value="ragChunkSize" @input="$emit('update:ragChunkSize', Number($event.target.value))" type="range" min="300" max="1200" step="50" />

                <div class="form-row">
                  <span>Chunk overlap</span>
                  <span>{{ ragOverlap }}</span>
                </div>
                <input :value="ragOverlap" @input="$emit('update:ragOverlap', Number($event.target.value))" type="range" min="0" max="300" step="20" />
              </div>
            </div>

            <div v-if="activeTab === 'docs'" class="tab-panel">
              <div class="section">
                <label>Upload document</label>

                <input
                  ref="fileInputRef"
                  class="hidden-file-input"
                  type="file"
                  @change="$emit('file-change', $event)"
                />

                <div class="file-picker-row">
                  <button class="file-pick-btn" @click="openFilePicker">
                    Choose File
                  </button>
                  <div class="file-picked-name">
                    {{ pendingFile?.name || 'No file selected' }}
                  </div>
                </div>

                <p class="upload-hint">
                  Supported: TXT, MD, PDF, DOCX. Files are chunked for retrieval.
                </p>

                <button class="upload-btn" @click="$emit('upload-file')" :disabled="!pendingFile || uploading">
                  {{ uploading ? 'Uploading...' : 'Upload File' }}
                </button>
              </div>

              <div class="section">
                <label>Available documents</label>
                <div class="file-list">
                  <div v-for="f in uploadedFiles" :key="f.name" class="file-card">
                    <div class="file-left">
                      <input
                        type="checkbox"
                        :checked="selectedFiles.includes(f.name)"
                        @change="$emit('toggle-file', f.name)"
                      />
                      <div class="file-info">
                        <div class="file-name">{{ f.name }}</div>
                        <div class="file-meta">{{ f.type.toUpperCase() }} · {{ f.size_kb }} KB</div>
                      </div>
                    </div>
                    <button class="danger-btn" @click="$emit('delete-file', f.name)">Delete</button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="history-panel">
            <div class="history-header">
              <span>History</span>
              <span class="history-count">{{ histories.length }}</span>
            </div>

            <transition-group name="history-item" tag="div" class="history-list">
              <div
                v-for="item in histories"
                :key="item.id"
                class="history-item-wrap"
              >
                <button
                  class="history-item"
                  :class="{ active: currentHistoryId === item.id }"
                  @click="$emit('open-history', item)"
                >
                  <div class="history-title">{{ item.title }}</div>
                  <div class="history-meta">
                    <span>{{ item.domain }}</span>
                    <span>•</span>
                    <span>{{ formatTime(item.updatedAt) }}</span>
                  </div>
                </button>

                <button
                  class="history-delete-btn"
                  @click.stop="$emit('delete-history', item)"
                  aria-label="Delete history"
                  title="Delete history"
                >
                  <svg viewBox="0 0 24 24" fill="none" class="trash-icon">
                    <path d="M9 3h6l1 2h4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M4 5h16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                    <path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M10 11v6M14 11v6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                  </svg>
                </button>
              </div>
            </transition-group>

            <div v-if="!histories.length" class="history-empty">
              No chat history yet.
            </div>
          </div>
        </aside>
      </transition>

      <transition name="confirm-fade">
        <div v-if="pendingDeleteItem" class="confirm-overlay">
          <div class="confirm-panel panel">
            <h3>Delete this conversation?</h3>
            <p>This action cannot be undone.</p>

            <div class="confirm-target">
              {{ pendingDeleteItem.title }}
            </div>

            <div class="confirm-actions">
              <button class="confirm-cancel-btn" @click="$emit('cancel-delete-history')">
                Cancel
              </button>
              <button class="confirm-delete-btn" @click="$emit('confirm-delete-history')">
                Delete
              </button>
            </div>
          </div>
        </div>
      </transition>
    </div>
  </transition>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'

const activeTab = ref('settings')
const fileInputRef = ref(null)
const modelMenuOpen = ref(false)

const props = defineProps({
  open: Boolean,
  models: Array,
  selectedModel: String,
  maxNewTokens: Number,
  temperature: Number,
  topP: Number,
  useRag: Boolean,
  ragTopK: Number,
  ragChunkSize: Number,
  ragOverlap: Number,
  uploadedFiles: Array,
  selectedFiles: Array,
  pendingFile: Object,
  uploading: Boolean,
  histories: Array,
  currentHistoryId: String,
  pendingDeleteHistory: Object,
})

const emit = defineEmits([
  'close',
  'new-chat',
  'open-history',
  'delete-history',
  'confirm-delete-history',
  'cancel-delete-history',
  'update:selectedModel',
  'update:maxNewTokens',
  'update:temperature',
  'update:topP',
  'update:useRag',
  'update:ragTopK',
  'update:ragChunkSize',
  'update:ragOverlap',
  'file-change',
  'upload-file',
  'toggle-file',
  'delete-file',
])

const selectedModelLabel = computed(() => {
  const found = props.models.find(m => m.key === props.selectedModel)
  return found?.label || 'Select model'
})

const pendingDeleteItem = computed(() => props.pendingDeleteHistory || null)

function formatTime(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  return d.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function openFilePicker() {
  fileInputRef.value?.click()
}

function toggleModelMenu() {
  modelMenuOpen.value = !modelMenuOpen.value
}

function pickModel(modelKey) {
  emit('update:selectedModel', modelKey)
  modelMenuOpen.value = false
}

function handleOutsideClick(event) {
  const target = event.target
  if (!target.closest('.custom-select')) {
    modelMenuOpen.value = false
  }
}

onMounted(() => {
  window.addEventListener('click', handleOutsideClick)
})

onBeforeUnmount(() => {
  window.removeEventListener('click', handleOutsideClick)
})
</script>

<style scoped>
.drawer-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(3, 5, 10, 0.42);
  z-index: 2000;
  display: flex;
  justify-content: flex-start;
}

.drawer {
  width: min(420px, 92vw);
  height: 100vh;
  background: rgba(14, 17, 22, 0.98);
  border-right: 1px solid rgba(255,255,255,0.08);
  padding: 22px 18px 18px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 18px;
}

.brand {
  display: flex;
  gap: 14px;
  align-items: center;
}

.icon {
  width: 46px;
  height: 46px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #5b7cff, #7b5cff);
  color: white;
  font-size: 18px;
}

.brand h2 {
  margin: 0;
  color: #fff;
}

.brand p {
  margin: 6px 0 0 0;
  color: #9aa4b2;
  font-size: 13px;
}

.close-btn {
  width: 40px;
  height: 40px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  color: white;
  cursor: pointer;
}

.quick-actions {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
  margin-bottom: 16px;
}

.quick-btn {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  color: #eaf0fb;
  border-radius: 14px;
  padding: 12px 12px;
  font-weight: 700;
  cursor: pointer;
}

.quick-btn.active,
.new-chat-btn {
  background: linear-gradient(135deg, #5b7cff, #7b5cff);
  border-color: transparent;
}

.drawer-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding-right: 4px;
}

.tab-panel {
  padding-right: 2px;
}

.section {
  margin-bottom: 18px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px;
  padding: 14px;
}

.section label {
  display: block;
  margin-bottom: 10px;
  font-weight: 700;
  font-size: 14px;
  color: #fff;
}

.form-row {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: #aab4c4;
  margin: 8px 0 4px;
}

.custom-select {
  position: relative;
}

.custom-select-trigger {
  width: 100%;
  text-align: left;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  color: #f3f6fb;
  border-radius: 16px;
  padding: 12px 16px;
  cursor: pointer;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.custom-select-menu {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  right: 0;
  background: rgba(15, 18, 25, 0.98);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 8px;
  box-shadow: 0 18px 40px rgba(0,0,0,0.35);
  z-index: 2100;
  max-height: 260px;
  overflow-y: auto;
}

.custom-select-item {
  width: 100%;
  text-align: left;
  border: none;
  background: transparent;
  color: #dde6f4;
  border-radius: 12px;
  padding: 11px 12px;
  cursor: pointer;
  font-weight: 600;
}

.custom-select-item:hover {
  background: rgba(255,255,255,0.05);
}

.custom-select-item.active {
  background: linear-gradient(135deg, rgba(91,124,255,0.26), rgba(123,92,255,0.22));
  color: white;
}

.caret {
  color: #98a3b6;
  flex-shrink: 0;
}

.toggle-box {
  margin-bottom: 12px;
}

.rag-state-btn {
  width: 100%;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  color: #c9d2e3;
  border-radius: 14px;
  padding: 12px 16px;
  cursor: pointer;
  font-weight: 700;
  transition: all 0.18s ease;
}

.rag-state-btn.active {
  background: linear-gradient(135deg, #0ea5e9, #2563eb);
  color: white;
  border-color: transparent;
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
}

input[type="range"] {
  width: 100%;
}

.hidden-file-input {
  display: none;
}

.file-picker-row {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}

.file-pick-btn {
  flex-shrink: 0;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.06);
  color: #eef2ff;
  border-radius: 14px;
  padding: 11px 16px;
  cursor: pointer;
  font-weight: 700;
  transition: all 0.18s ease;
}

.file-pick-btn:hover {
  background: rgba(255,255,255,0.1);
}

.file-picked-name {
  flex: 1;
  min-width: 0;
  color: #b6c0d0;
  font-size: 13px;
  padding: 11px 14px;
  border-radius: 14px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.05);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.upload-hint {
  margin: 10px 0 0 0;
  font-size: 12px;
  color: #9aa4b2;
  line-height: 1.45;
}

.upload-btn,
.danger-btn {
  border: none;
  border-radius: 14px;
  padding: 11px 14px;
  cursor: pointer;
  font-weight: 700;
}

.upload-btn {
  margin-top: 12px;
  width: 100%;
  background: linear-gradient(135deg, #5b7cff, #7b5cff);
  color: white;
  box-shadow: 0 12px 28px rgba(91, 124, 255, 0.24);
}

.upload-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.file-list {
  max-height: 240px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.file-card {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.06);
}

.file-left {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  flex: 1;
}

.file-info {
  min-width: 0;
}

.file-name {
  font-size: 14px;
  font-weight: 600;
  word-break: break-all;
  color: white;
}

.file-meta {
  font-size: 12px;
  color: #9aa4b2;
  margin-top: 4px;
}

.danger-btn {
  background: rgba(239, 68, 68, 0.14);
  color: #ffb4b4;
}

.history-panel {
  border-top: 1px solid rgba(255,255,255,0.08);
  padding-top: 14px;
  margin-top: 12px;
}

.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #fff;
  font-weight: 700;
  margin-bottom: 10px;
}

.history-count {
  color: #9aa4b2;
  font-size: 13px;
}

.history-list {
  max-height: 260px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.history-item-wrap {
  position: relative;
  display: flex;
  align-items: stretch;
}

.history-item {
  width: 100%;
  text-align: left;
  border: 1px solid rgba(255,255,255,0.06);
  background: rgba(255,255,255,0.03);
  color: white;
  border-radius: 14px;
  padding: 12px 48px 12px 12px;
  cursor: pointer;
  transition: border-color 0.18s ease, background 0.18s ease;
}

.history-item.active {
  border-color: rgba(91,124,255,0.6);
  background: rgba(91,124,255,0.12);
}

.history-delete-btn {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  width: 34px;
  height: 34px;
  border: none;
  border-radius: 10px;
  background: rgba(239, 68, 68, 0.08);
  color: rgba(255, 99, 99, 0.92);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.18s ease, transform 0.18s ease, background 0.18s ease;
}

.history-item-wrap:hover .history-delete-btn {
  opacity: 1;
  pointer-events: auto;
}

.history-delete-btn:hover {
  background: rgba(239, 68, 68, 0.16);
  transform: translateY(-50%) scale(1.04);
}

.trash-icon {
  width: 16px;
  height: 16px;
}

.history-title {
  font-weight: 700;
  font-size: 14px;
  line-height: 1.4;
}

.history-meta {
  margin-top: 6px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  color: #97a3b4;
  font-size: 12px;
  text-transform: capitalize;
}

.history-empty {
  color: #8c96a8;
  font-size: 13px;
  padding: 10px 2px;
}

.confirm-overlay {
  position: fixed;
  inset: 0;
  z-index: 2400;
  background: rgba(0, 0, 0, 0.38);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.confirm-panel {
  width: min(420px, 92vw);
  background: rgba(16, 18, 24, 0.98);
  border-radius: 22px;
  padding: 22px;
  border: 1px solid rgba(255,255,255,0.08);
  text-align: center;
}

.confirm-panel h3 {
  margin: 0 0 10px;
  color: #fff;
  font-size: 22px;
}

.confirm-panel p {
  margin: 0 0 14px;
  color: #aab4c4;
}

.confirm-target {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.06);
  padding: 12px 14px;
  border-radius: 14px;
  color: #f5f7fb;
  margin-bottom: 18px;
  word-break: break-word;
}

.confirm-actions {
  display: flex;
  gap: 10px;
  justify-content: center;
}

.confirm-cancel-btn,
.confirm-delete-btn {
  border: none;
  border-radius: 14px;
  padding: 11px 16px;
  font-weight: 700;
  cursor: pointer;
}

.confirm-cancel-btn {
  background: rgba(255,255,255,0.06);
  color: white;
}

.confirm-delete-btn {
  background: linear-gradient(135deg, #ef4444, #dc2626);
  color: white;
}

.drawer-overlay-enter-active,
.drawer-overlay-leave-active {
  transition: opacity 0.2s ease;
}

.drawer-overlay-enter-from,
.drawer-overlay-leave-to {
  opacity: 0;
}

.drawer-slide-enter-active,
.drawer-slide-leave-active {
  transition: transform 0.28s ease;
}

.drawer-slide-enter-from,
.drawer-slide-leave-to {
  transform: translateX(-100%);
}

.menu-fade-enter-active,
.menu-fade-leave-active,
.confirm-fade-enter-active,
.confirm-fade-leave-active {
  transition: all 0.22s ease;
}

.menu-fade-enter-from,
.menu-fade-leave-to,
.confirm-fade-enter-from,
.confirm-fade-leave-to {
  opacity: 0;
  transform: translateY(8px);
}

.history-item-enter-active,
.history-item-leave-active,
.history-item-move {
  transition: all 0.28s ease;
}

.history-item-enter-from {
  opacity: 0;
  transform: translateY(12px);
}

.history-item-leave-to {
  opacity: 0;
  transform: translateX(-48px);
}

.history-item-leave-active {
  position: relative;
}

@media (max-width: 700px) {
  .drawer {
    width: 100vw;
  }

  .quick-actions {
    grid-template-columns: 1fr;
  }

  .file-picker-row {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
