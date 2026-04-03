<template>
  <main class="main">
    <button class="sidebar-toggle panel" @click="$emit('toggle-sidebar')">
      ☰
    </button>

    <transition name="view-switch" mode="out-in">
      <div v-if="!hasConversation" key="hero" class="hero-shell">
        <div class="hero-copy">
          <div class="hero-kicker">Multi-model academic demo</div>
          <transition name="fade-slide" mode="out-in">
            <h1 :key="selectedDomain" class="hero-title">
              {{ heroTitle }}
            </h1>
          </transition>
          <transition name="fade-slide" mode="out-in">
            <p :key="selectedDomain + '-sub'" class="hero-subtitle">
              {{ heroSubtitle }}
            </p>
          </transition>
        </div>

        <div class="domain-switcher">
          <button
            v-for="item in domainOptions"
            :key="item.key"
            class="domain-chip"
            :class="[item.colorClass, { active: selectedDomain === item.key }]"
            @click="$emit('select-domain', item.key)"
          >
            {{ item.label }}
          </button>
        </div>

        <div class="hero-model-block">
          <span class="hero-model-label">Current model</span>
          <div class="model-select">
            <button class="model-select-trigger" @click.stop="toggleHeroModelMenu">
              <span>{{ selectedModelLabel }}</span>
              <span class="caret">⌄</span>
            </button>

            <transition name="menu-fade">
              <div v-if="heroModelMenuOpen" class="model-menu">
                <button
                  v-for="model in domainModels"
                  :key="model.key"
                  class="model-menu-item"
                  :class="{ active: selectedModel === model.key }"
                  @click="pickModel(model.key, 'hero')"
                >
                  {{ model.label }}
                </button>
              </div>
            </transition>
          </div>
        </div>

        <div class="hero-input-zone">
          <section class="composer composer-hero panel animated-panel">
            <textarea
              :value="query"
              @input="$emit('update:query', $event.target.value)"
              @focus="handleHeroFocus"
              @blur="handleHeroBlur"
              placeholder="Ask anything in the selected domain..."
              @keydown.enter.exact.prevent="$emit('send-message')"
              :disabled="loading"
            />

            <div class="composer-toolbar">
              <div class="left-actions">
                <button
                  class="rag-toggle-btn"
                  :class="{ active: useRag }"
                  @click="$emit('update:useRag', !useRag)"
                  :disabled="loading"
                >
                  {{ useRag ? 'RAG ON' : 'RAG OFF' }}
                </button>

                <button class="ghost-btn" @click="$emit('open-settings')">
                  Settings
                </button>
              </div>

              <button class="primary-btn" @click="$emit('send-message')" :disabled="loading">
                {{ loading ? 'Generating...' : 'Start Chat' }}
              </button>
            </div>
          </section>

          <transition name="suggest-reveal">
            <div
              v-if="showExamples"
              class="suggest-popover"
              @mouseenter="exampleHover = true"
              @mouseleave="exampleHover = false"
            >
              <ExampleQuestions
                :selected-domain="selectedDomain"
                @select-example="emit('select-example', $event)"
              />
            </div>
          </transition>
        </div>
      </div>

      <div v-else key="chat" class="conversation-shell">
        <header class="topbar panel animated-panel">
          <div class="topbar-left">
            <div>
              <h1>{{ topbarTitle }}</h1>
              <div class="topbar-model-row">
                <span class="topbar-model-caption">Current model</span>
                <div class="model-select compact">
                  <button class="model-select-trigger compact" @click.stop="toggleChatModelMenu">
                    <span>{{ selectedModelLabel }}</span>
                    <span class="caret">⌄</span>
                  </button>

                  <transition name="menu-fade">
                    <div v-if="chatModelMenuOpen" class="model-menu compact">
                      <button
                        v-for="model in domainModels"
                        :key="model.key"
                        class="model-menu-item"
                        :class="{ active: selectedModel === model.key }"
                        @click="pickModel(model.key, 'chat')"
                      >
                        {{ model.label }}
                      </button>
                    </div>
                  </transition>
                </div>
              </div>
            </div>
          </div>

          <div class="topbar-actions">
            <button class="ghost-btn" @click="$emit('new-chat')" :disabled="loading">New Chat</button>
            <button class="ghost-btn" @click="$emit('open-settings')" :disabled="loading">Settings</button>
            <button class="ghost-btn" @click="$emit('clear-chat')" :disabled="loading">Clear</button>
          </div>
        </header>

        <transition name="panel-smooth">
          <section class="status-panel panel animated-panel" v-if="loading || statusItems.length">
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
        </transition>

        <section class="chat-box panel animated-panel">
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

        <section class="composer panel animated-panel">
          <textarea
            :value="query"
            @input="$emit('update:query', $event.target.value)"
            placeholder="Continue the conversation..."
            @keydown.enter.exact.prevent="$emit('send-message')"
            :disabled="loading"
          />

          <div class="composer-toolbar">
            <div class="left-actions">
              <button
                class="rag-toggle-btn"
                :class="{ active: useRag }"
                @click="$emit('update:useRag', !useRag)"
                :disabled="loading"
              >
                {{ useRag ? 'RAG ON' : 'RAG OFF' }}
              </button>
            </div>

            <button class="primary-btn" @click="$emit('send-message')" :disabled="loading">
              {{ loading ? 'Generating response...' : 'Send' }}
            </button>
          </div>
        </section>
      </div>
    </transition>
  </main>
</template>

<script setup>
import { computed, ref, onMounted, onBeforeUnmount } from 'vue'
import ExampleQuestions from './ExampleQuestions.vue'

const props = defineProps({
  messages: Array,
  statusItems: Array,
  loading: Boolean,
  query: String,
  useRag: Boolean,
  streamingMessageId: Number,
  formatMessage: Function,
  selectedDomain: String,
  domainOptions: Array,
  hasConversation: Boolean,
  selectedModelLabel: String,
  domainModels: Array,
  selectedModel: String,
})

const emit = defineEmits([
  'toggle-sidebar',
  'open-settings',
  'new-chat',
  'clear-chat',
  'send-message',
  'select-domain',
  'select-model',
  'select-example',
  'update:query',
  'update:useRag',
])

const heroModelMenuOpen = ref(false)
const chatModelMenuOpen = ref(false)

const heroFocused = ref(false)
const exampleHover = ref(false)
let blurTimer = null

const showExamples = computed(() => heroFocused.value || exampleHover.value)

const domainTitleMap = {
  medical: 'Medical Chatbot',
  finance: 'Finance Chatbot',
  legal: 'Legal Chatbot',
  general: 'General Knowledge Chatbot',
  multidomain: 'Multi-domain Chatbot',
}

const heroTitle = computed(() =>
  props.selectedDomain
    ? `Here's the ${domainTitleMap[props.selectedDomain] || 'Chatbot'}`
    : `What's on your mind today?`
)

const heroSubtitle = computed(() =>
  props.selectedDomain
    ? `Select a model, ask a question, and compare domain-aware behavior in a clean demo flow.`
    : `What can I help you with today?`
)

const topbarTitle = computed(() =>
  domainTitleMap[props.selectedDomain] || 'Chatbot'
)

function handleHeroFocus() {
  if (blurTimer) {
    clearTimeout(blurTimer)
    blurTimer = null
  }
  heroFocused.value = true
}

function handleHeroBlur() {
  blurTimer = setTimeout(() => {
    if (!exampleHover.value) {
      heroFocused.value = false
    }
  }, 120)
}

function toggleHeroModelMenu() {
  heroModelMenuOpen.value = !heroModelMenuOpen.value
  chatModelMenuOpen.value = false
}

function toggleChatModelMenu() {
  chatModelMenuOpen.value = !chatModelMenuOpen.value
  heroModelMenuOpen.value = false
}

function pickModel(modelKey, source) {
  emit('select-model', modelKey)
  if (source === 'hero') heroModelMenuOpen.value = false
  if (source === 'chat') chatModelMenuOpen.value = false
}

function handleOutsideClick(event) {
  const target = event.target
  if (!target.closest('.model-select')) {
    heroModelMenuOpen.value = false
    chatModelMenuOpen.value = false
  }
}

onMounted(() => {
  window.addEventListener('click', handleOutsideClick)
})

onBeforeUnmount(() => {
  window.removeEventListener('click', handleOutsideClick)
  if (blurTimer) clearTimeout(blurTimer)
})
</script>

<style scoped>
.main {
  min-height: 100vh;
  padding: 22px 28px 24px 92px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  position: relative;
}

.sidebar-toggle {
  position: fixed;
  top: 22px;
  left: 24px;
  width: 48px;
  height: 48px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(20, 23, 30, 0.9);
  color: white;
  cursor: pointer;
  z-index: 30;
}

.hero-shell,
.conversation-shell {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.hero-shell {
  min-height: calc(100vh - 48px);
  align-items: center;
  justify-content: center;
  gap: 22px;
}

.hero-copy {
  text-align: center;
  max-width: 920px;
}

.hero-kicker {
  color: #8b93a7;
  font-size: 13px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 14px;
}

.hero-title {
  margin: 0;
  font-size: clamp(34px, 5vw, 62px);
  font-weight: 700;
  line-height: 1.05;
  color: #f7f8fb;
}

.hero-subtitle {
  margin: 14px auto 0;
  max-width: 760px;
  color: #98a3b6;
  font-size: 17px;
  line-height: 1.7;
}

.domain-switcher {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.domain-chip {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  color: #e8ebf2;
  border-radius: 999px;
  padding: 12px 18px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.domain-chip:hover {
  transform: translateY(-1px);
}

.domain-chip.active {
  box-shadow: 0 0 0 1px rgba(255,255,255,0.04), 0 12px 28px rgba(0,0,0,0.25);
}

.domain-chip.medical.active { background: linear-gradient(135deg, #2a69ff, #5b7cff); }
.domain-chip.finance.active { background: linear-gradient(135deg, #059669, #10b981); }
.domain-chip.legal.active { background: linear-gradient(135deg, #7c3aed, #9f67ff); }
.domain-chip.general.active { background: linear-gradient(135deg, #d97706, #f59e0b); }
.domain-chip.multidomain.active { background: linear-gradient(135deg, #ec4899, #8b5cf6); }

.hero-model-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.hero-model-label,
.topbar-model-caption {
  color: #9aa4b2;
  font-size: 13px;
}

.hero-input-zone {
  width: min(920px, 92%);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-select {
  position: relative;
  min-width: 360px;
  z-index: 40;
}

.model-select.compact {
  min-width: 320px;
  z-index: 40;
}

.model-select-trigger {
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

.model-select-trigger.compact {
  padding: 10px 14px;
  border-radius: 14px;
  font-size: 13px;
}

.caret {
  color: #98a3b6;
  flex-shrink: 0;
}

.model-menu {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  right: 0;
  background: rgba(15, 18, 25, 0.98);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 8px;
  box-shadow: 0 18px 40px rgba(0,0,0,0.35);
  z-index: 60;
  max-height: 260px;
  overflow-y: auto;
}

.model-menu.compact {
  max-height: 220px;
}

.model-menu-item {
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

.model-menu-item:hover {
  background: rgba(255,255,255,0.05);
}

.model-menu-item.active {
  background: linear-gradient(135deg, rgba(91,124,255,0.26), rgba(123,92,255,0.22));
  color: white;
}

.topbar, .status-panel, .chat-box, .composer {
  border-radius: 24px;
  background: rgba(20, 23, 30, 0.9);
  border: 1px solid rgba(255,255,255,0.08);
}

.animated-panel {
  transition: transform 0.24s ease, opacity 0.24s ease, min-height 0.24s ease, padding 0.24s ease;
}

.topbar {
  position: relative;
  z-index: 30;
  overflow: visible;
  padding: 18px 22px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.status-panel {
  position: relative;
  z-index: 10;
  padding: 14px 18px;
}

.chat-box {
  position: relative;
  z-index: 1;
  flex: 1;
  min-height: 440px;
  max-height: calc(100vh - 300px);
  overflow-y: auto;
  padding: 24px;
  scroll-behavior: smooth;
}

.composer {
  position: relative;
  z-index: 1;
  padding: 16px;
}

.composer-hero {
  width: 100%;
}

.suggest-popover {
  width: 100%;
}

.topbar h1 {
  margin: 0;
  font-size: 22px;
}

.topbar-model-row {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.topbar-actions {
  display: flex;
  gap: 10px;
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
  background: linear-gradient(135deg, #5b7cff, #7b5cff);
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
  background: rgba(59, 130, 246, 0.13);
  color: #8fc2ff;
}

.status-item.done {
  background: rgba(16, 185, 129, 0.12);
  color: #97ffd4;
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
  border: 2px solid rgba(91, 124, 255, 0.2);
  border-top-color: #5b7cff;
  animation: spin 0.9s linear infinite;
}

.chat-box::-webkit-scrollbar {
  width: 10px;
}

.chat-box::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.28);
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
  background: linear-gradient(135deg, #1e293b, #334155);
  color: #dbeafe;
}

.user-avatar {
  background: linear-gradient(135deg, #5b7cff, #7b5cff);
  color: white;
}

.bubble {
  max-width: 78%;
  padding: 14px 16px;
  border-radius: 20px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-wrap: break-word;
  position: relative;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
}

.assistant-bubble {
  background: #161a22;
  border: 1px solid rgba(255,255,255,0.08);
  border-bottom-left-radius: 8px;
  color: #f4f7fb;
}

.user-bubble {
  background: linear-gradient(135deg, #5b7cff, #7b5cff);
  color: white;
  border-bottom-right-radius: 8px;
}

textarea {
  width: 100%;
  min-height: 110px;
  resize: none;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.08);
  padding: 14px 16px;
  font-size: 14px;
  background: rgba(255,255,255,0.03);
  color: white;
  outline: none;
}

textarea::placeholder {
  color: #788295;
}

.composer-toolbar {
  margin-top: 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.left-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.rag-toggle-btn {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  color: #c9d2e3;
  border-radius: 14px;
  padding: 12px 16px;
  cursor: pointer;
  font-weight: 700;
  transition: all 0.18s ease;
}

.rag-toggle-btn.active {
  background: linear-gradient(135deg, #0ea5e9, #2563eb);
  color: white;
  border-color: transparent;
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
}

.primary-btn, .ghost-btn {
  border: none;
  border-radius: 14px;
  padding: 12px 16px;
  cursor: pointer;
  font-weight: 700;
  transition: transform 0.16s ease, opacity 0.16s ease, background 0.16s ease;
}

.primary-btn {
  background: linear-gradient(135deg, #5b7cff, #7b5cff);
  color: white;
}

.ghost-btn {
  background: rgba(255,255,255,0.05);
  color: white;
  border: 1px solid rgba(255,255,255,0.08);
}

.primary-btn:hover,
.ghost-btn:hover,
.domain-chip:hover,
.rag-toggle-btn:hover {
  opacity: 0.95;
}

.typing-cursor {
  animation: pulse 0.9s infinite;
  margin-left: 2px;
}

.fade-slide-enter-active,
.fade-slide-leave-active,
.menu-fade-enter-active,
.menu-fade-leave-active,
.view-switch-enter-active,
.view-switch-leave-active,
.suggest-reveal-enter-active,
.suggest-reveal-leave-active,
.panel-smooth-enter-active,
.panel-smooth-leave-active {
  transition: all 0.22s ease;
}

.fade-slide-enter-from,
.fade-slide-leave-to,
.menu-fade-enter-from,
.menu-fade-leave-to,
.view-switch-enter-from,
.view-switch-leave-to,
.suggest-reveal-enter-from,
.suggest-reveal-leave-to,
.panel-smooth-enter-from,
.panel-smooth-leave-to {
  opacity: 0;
  transform: translateY(8px);
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes pulse {
  0% { opacity: 0.55; }
  50% { opacity: 1; }
  100% { opacity: 0.55; }
}

@media (max-width: 900px) {
  .main {
    padding: 18px 16px 18px 74px;
  }

  .topbar {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }

  .topbar-actions {
    width: 100%;
    flex-wrap: wrap;
  }

  .bubble {
    max-width: 88%;
  }

  .composer-toolbar {
    flex-direction: column;
    align-items: stretch;
  }

  .left-actions {
    justify-content: space-between;
  }

  .model-select,
  .model-select.compact {
    min-width: 100%;
  }

  .hero-input-zone {
    width: 100%;
  }
}
</style>
