import { defineStore } from 'pinia'
import {
  listChatSessionMessages,
  streamChatAnswer,
  type BackendChatMessage,
} from '@/services/chatApi'
import { useAuthStore } from '@/stores/auth'
import { useHistoryStore } from '@/stores/history'

export type ChatRole = 'user' | 'assistant'

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  createdAt: string
}

function makeId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 100_000)}`
}

function toChatMessage(message: BackendChatMessage): ChatMessage | null {
  if (message.role !== 'user' && message.role !== 'assistant') {
    return null
  }
  return {
    id: message.id,
    role: message.role,
    content: message.content || '',
    createdAt: message.created_at || new Date().toISOString(),
  }
}

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [] as ChatMessage[],
    draft: '',
    isStreaming: false,
    activeSessionId: '' as string,
    streamError: '',
    streamAbortController: null as AbortController | null,
    streamingMessageId: '' as string,
    pendingAssistantTokens: '',
    streamFlushTimer: null as number | null,
  }),
  getters: {
    hasMessages: (state) => state.messages.length > 0,
  },
  actions: {
    setDraft(value: string) {
      this.draft = value
    },
    clearAssistantStreamState() {
      if (this.streamFlushTimer !== null) {
        window.cancelAnimationFrame(this.streamFlushTimer)
        this.streamFlushTimer = null
      }
      this.pendingAssistantTokens = ''
      this.streamingMessageId = ''
    },
    resetConversation() {
      this.streamAbortController?.abort()
      this.streamAbortController = null
      this.clearAssistantStreamState()
      this.messages = []
      this.draft = ''
      this.isStreaming = false
      this.activeSessionId = ''
      this.streamError = ''
    },
    pushUserMessage(content: string) {
      this.messages.push({
        id: makeId('user'),
        role: 'user',
        content,
        createdAt: new Date().toISOString(),
      })
    },
    startAssistantMessage() {
      const id = makeId('assistant')
      this.messages.push({
        id,
        role: 'assistant',
        content: '',
        createdAt: new Date().toISOString(),
      })
      this.clearAssistantStreamState()
      this.streamingMessageId = id
      this.isStreaming = true
      return id
    },
    appendAssistantToken(messageId: string, token: string) {
      const target = this.messages.find((item) => item.id === messageId)
      if (!target) {
        return
      }
      target.content += token
    },
    flushAssistantTokens() {
      if (this.streamFlushTimer !== null) {
        window.cancelAnimationFrame(this.streamFlushTimer)
        this.streamFlushTimer = null
      }
      if (!this.streamingMessageId || !this.pendingAssistantTokens) {
        return
      }
      const chunk = this.pendingAssistantTokens
      this.pendingAssistantTokens = ''
      this.appendAssistantToken(this.streamingMessageId, chunk)
    },
    scheduleAssistantFlush() {
      if (this.streamFlushTimer !== null) {
        return
      }
      this.streamFlushTimer = window.requestAnimationFrame(() => {
        this.flushAssistantTokens()
      })
    },
    queueAssistantToken(token: string) {
      if (!token) {
        return
      }
      this.pendingAssistantTokens += token
      this.scheduleAssistantFlush()
    },
    setAssistantMessageContent(messageId: string, content: string) {
      if (this.streamingMessageId === messageId) {
        this.pendingAssistantTokens = ''
        if (this.streamFlushTimer !== null) {
          window.cancelAnimationFrame(this.streamFlushTimer)
          this.streamFlushTimer = null
        }
      }
      const target = this.messages.find((item) => item.id === messageId)
      if (!target) {
        return
      }
      target.content = content
    },
    finishAssistantMessage() {
      if (this.pendingAssistantTokens && this.streamingMessageId) {
        this.appendAssistantToken(this.streamingMessageId, this.pendingAssistantTokens)
      }
      this.clearAssistantStreamState()
      this.isStreaming = false
    },
    async openSession(accessToken: string, sessionId: string) {
      const token = accessToken.trim()
      const targetSessionId = sessionId.trim()
      if (!token || !targetSessionId) {
        return
      }

      this.streamAbortController?.abort()
      this.streamAbortController = null
      this.clearAssistantStreamState()
      this.streamError = ''
      this.isStreaming = false

      const backendMessages = await listChatSessionMessages(token, targetSessionId)
      this.messages = backendMessages
        .map((message) => toChatMessage(message))
        .filter((item): item is ChatMessage => item !== null)

      this.activeSessionId = targetSessionId
      const historyStore = useHistoryStore()
      historyStore.setActiveItem(targetSessionId)
    },
    async sendMessage(rawText?: string) {
      const text = (rawText ?? this.draft).trim()
      if (!text || this.isStreaming) {
        return
      }

      const authStore = useAuthStore()
      const historyStore = useHistoryStore()
      this.pushUserMessage(text)
      this.draft = ''

      const assistantId = this.startAssistantMessage()
      this.streamError = ''

      try {
        const hasSession = await authStore.ensureSession()
        if (!hasSession || !authStore.accessToken.trim()) {
          throw new Error('Authentication session is invalid. Please sign in again.')
        }

        const accessToken = authStore.accessToken.trim()
        let sessionId = this.activeSessionId.trim()
        let hasTouchedHistory = false
        if (sessionId) {
          historyStore.touchSession(sessionId, text)
          hasTouchedHistory = true
        }
        const controller = new AbortController()
        this.streamAbortController = controller

        await streamChatAnswer(
          accessToken,
          {
            question: text,
            ...(sessionId ? { session_id: sessionId } : {}),
          },
          {
            onMeta: (data) => {
              if (!data || typeof data !== 'object') {
                return
              }
              const payload = data as Record<string, unknown>
              const streamSessionId = payload.session_id
              if (typeof streamSessionId === 'string' && streamSessionId.trim()) {
                sessionId = streamSessionId.trim()
                this.activeSessionId = sessionId
                historyStore.setActiveItem(sessionId)
                if (!hasTouchedHistory) {
                  historyStore.touchSession(sessionId, text)
                  hasTouchedHistory = true
                }
              }
            },
            onToken: (token) => {
              if (this.streamingMessageId !== assistantId) {
                this.streamingMessageId = assistantId
              }
              this.queueAssistantToken(token)
            },
            onError: (error) => {
              this.streamError = error || 'Unknown stream error'
              controller.abort()
            },
          },
          controller.signal,
        )
      } catch (error) {
        const message = error instanceof Error ? error.message.trim() : ''
        const fallbackMessage = 'Xin lỗi, chưa thể tạo câu trả lời lúc này. Vui lòng thử lại.'
        const safeMessage = this.streamError || message || fallbackMessage
        this.setAssistantMessageContent(assistantId, safeMessage)
      } finally {
        this.streamAbortController = null
        this.finishAssistantMessage()
        if (!this.messages.find((item) => item.id === assistantId)?.content.trim()) {
          this.setAssistantMessageContent(assistantId, 'Xin lỗi, chưa có dữ liệu phản hồi từ hệ thống.')
        }
      }
    },
  },
})
