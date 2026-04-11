<template>
  <main class="main enter-up delay-2">
    <header class="topbar panel">
      <div>
        <h1>Medical Chatbot Demo</h1>
        <p>Vue + SSE Python backend + optional RAG</p>
      </div>
      <button class="ghost-btn" @click="$emit('clear-chat')" :disabled="loading">Clear Chat</button>
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
        :value="query"
        @input="$emit('update:query', $event.target.value)"
        placeholder="Enter your medical question..."
        @keydown.enter.exact.prevent="$emit('send-message')"
        :disabled="loading"
      />

      <div class="composer-toolbar">
        <label class="composer-toggle">
          <input
            type="checkbox"
            :checked="useRag"
            @change="$emit('update:useRag', $event.target.checked)"
            :disabled="loading"
          />
          <span>Use RAG</span>
        </label>

        <div class="actions">
          <button class="primary-btn" @click="$emit('send-message')" :disabled="loading">
            {{ loading ? 'Generating response...' : 'Generate Response' }}
          </button>
        </div>
      </div>
    </section>
  </main>
</template>

<script setup>
defineProps({
  messages: Array,
  statusItems: Array,
  loading: Boolean,
  query: String,
  useRag: Boolean,
  streamingMessageId: Number,
  formatMessage: Function,
})

defineEmits(['clear-chat', 'send-message', 'update:query', 'update:useRag'])
</script>

<style scoped>
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
  will-change: transform, opacity;
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

textarea {
  width: 100%;
  min-height: 110px;
  resize: vertical;
  border-radius: 16px;
  border: 1px solid #d8dee9;
  padding: 12px 14px;
  font-size: 14px;
  background: rgba(255,255,255,0.96);
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

.bubble-enter-active, .bubble-leave-active {
  transition: opacity 0.38s ease, transform 0.38s cubic-bezier(0.22, 1, 0.36, 1);
}
.bubble-enter-from {
  opacity: 0;
  transform: translateY(14px) scale(0.96);
}
.bubble-enter-to {
  opacity: 1;
  transform: translateY(0) scale(1);
}
.bubble-leave-to {
  opacity: 0;
  transform: translateY(-10px) scale(0.98);
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
</style>
