<script setup lang="ts">
import { useMessage } from 'naive-ui'
import { storeToRefs } from 'pinia'
import { computed, nextTick, reactive, ref, watch } from 'vue'
import ChatInput from '@/components/chat/ChatInput.vue'
import { downloadChatReference } from '@/services/chatApi'
import { renderAssistantMessage, type RenderedAssistantMessage } from '@/services/chatMarkdown'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'

const chatStore = useChatStore()
const authStore = useAuthStore()
const messageApi = useMessage()
const { hasMessages, isStreaming, messages, streamingMessageId } = storeToRefs(chatStore)
const userInitial = computed(() => authStore.currentUser?.fullName?.charAt(0).toUpperCase() || 'U')
const downloadingReference = ref('')
const assistantRenderState = reactive<Record<string, { content: string; view: RenderedAssistantMessage }>>({})

const quickActions = [
  { text: 'Thông tin cơ sở đào tạo', icon: 'school', colorClass: 'action-amber' },
  { text: 'Phương thức & điều kiện xét tuyển', icon: 'fact_check', colorClass: 'action-blue' },
  { text: 'Thời gian nộp hồ sơ', icon: 'event_note', colorClass: 'action-green' },
  { text: 'Điểm chuẩn ngành Công nghệ thông tin', icon: 'bar_chart', colorClass: 'action-purple' },
]

async function sendQuickMessage(text: string) {
  await chatStore.sendMessage(text)
  await nextTick()
}

function renderAssistantView(messageId: string, content: string) {
  const current = assistantRenderState[messageId]
  if (current?.content === content) {
    return
  }
  assistantRenderState[messageId] = {
    content,
    view: content ? renderAssistantMessage(content) : { html: '', references: [] },
  }
}

const assistantMessages = computed(() =>
  messages.value
    .filter((message) => message.role === 'assistant')
    .map((message) => ({
      id: message.id,
      content: message.content,
      isStreamingMessage: isStreaming.value && streamingMessageId.value === message.id,
    })),
)

watch(
  assistantMessages,
  (items) => {
    const activeAssistantIds = new Set<string>()

    for (const message of items) {
      activeAssistantIds.add(message.id)
      if (!message.isStreamingMessage) {
        renderAssistantView(message.id, message.content)
      }
    }

    for (const messageId of Object.keys(assistantRenderState)) {
      if (activeAssistantIds.has(messageId)) {
        continue
      }
      delete assistantRenderState[messageId]
    }
  },
  {
    immediate: true,
  },
)

const renderedMessages = computed(() =>
  messages.value.map((message) => {
    const isStreamingMessage =
      message.role === 'assistant' && isStreaming.value && streamingMessageId.value === message.id

    return {
      ...message,
      isStreamingMessage,
      assistantView: message.role === 'assistant' ? assistantRenderState[message.id]?.view ?? null : null,
    }
  }),
)

async function handleReferenceDownload(referenceName: string) {
  const normalizedName = referenceName.trim()
  if (!normalizedName || downloadingReference.value) {
    return
  }

  const hasSession = await authStore.ensureSession()
  if (!hasSession || !authStore.accessToken.trim()) {
    messageApi.error('Phiên đăng nhập không hợp lệ. Vui lòng đăng nhập lại.')
    return
  }

  downloadingReference.value = normalizedName
  try {
    await downloadChatReference(authStore.accessToken.trim(), normalizedName)
  } catch (error) {
    const message =
      error instanceof Error && error.message.trim()
        ? error.message.trim()
        : `Không thể tải file tham chiếu "${normalizedName}".`
    messageApi.error(message)
  } finally {
    downloadingReference.value = ''
  }
}
</script>

<template>
  <div class="chat-view">
    <div class="chat-area">
      <div v-if="!hasMessages" class="welcome-wrap">
        <h1 class="welcome-title">Chào mừng bạn đến với Trợ lý tư vấn Tuyển sinh UTEHY</h1>
        <p class="welcome-sub">Hãy hỏi bất cứ điều gì về chương trình đào tạo, học phí và tuyển sinh.</p>
        <div class="action-grid">
          <button
            v-for="action in quickActions"
            :key="action.text"
            class="action-card"
            type="button"
            @click="sendQuickMessage(action.text)"
          >
            <div class="action-card-left">
              <div class="action-icon" :class="action.colorClass">
                <span class="material-icons-outlined">{{ action.icon }}</span>
              </div>
              <span class="action-label">{{ action.text }}</span>
            </div>
            <span class="action-close">
              <span class="material-icons-outlined">check</span>
            </span>
          </button>
        </div>
      </div>

      <div v-else class="messages">
        <div v-for="message in renderedMessages" :key="message.id" class="msg" :class="{ user: message.role === 'user' }">
          <div class="msg-av" :class="message.role === 'user' ? 'usr' : 'ai'">
            <span v-if="message.role === 'user'">{{ userInitial }}</span>
            <span v-else class="material-icons-outlined msg-av-icon" aria-hidden="true">smart_toy</span>
          </div>
          <div class="msg-bubble">
            <div v-if="message.role === 'user'" class="msg-plain">
              {{ message.content || '...' }}
            </div>
            <template v-else>
              <div v-if="message.isStreamingMessage" class="msg-plain msg-streaming-content">
                {{ message.content || '...' }}
              </div>
              <div
                v-else-if="message.assistantView?.html"
                class="msg-markdown"
                v-html="message.assistantView.html"
              ></div>
              <div v-else class="msg-plain">
                {{ message.content || '...' }}
              </div>
              <div v-if="!message.isStreamingMessage && message.assistantView?.references.length" class="msg-references">
                <p class="msg-references-title">Thông tin tham chiếu</p>
                <button
                  v-for="reference in message.assistantView.references"
                  :key="`${message.id}-${reference.index}-${reference.fileName}`"
                  class="msg-reference-link"
                  type="button"
                  :disabled="downloadingReference === reference.fileName"
                  @click="handleReferenceDownload(reference.fileName)"
                >
                  <span class="msg-reference-index">[{{ reference.index }}].</span>
                  <span class="msg-reference-name">{{ reference.fileName }}</span>
                </button>
              </div>
            </template>
          </div>
        </div>
      </div>

      <div v-if="isStreaming" class="typing-row">
        <div class="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>

    <ChatInput />
  </div>
</template>
