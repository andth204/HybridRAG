import { defineStore } from 'pinia'
import { createChatSession, streamChatAnswer } from '@/services/chatApi'
import { useAuthStore } from '@/stores/auth'
import { useUiStore } from '@/stores/ui'

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

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [] as ChatMessage[],
    draft: '',
    isStreaming: false,
    activeSessionId: '' as string,
    streamError: '',
    streamAbortController: null as AbortController | null,
  }),
  getters: {
    hasMessages: (state) => state.messages.length > 0,
  },
  actions: {
    setDraft(value: string) {
      this.draft = value
    },
    resetConversation() {
      this.streamAbortController?.abort()
      this.streamAbortController = null
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
    setAssistantMessageContent(messageId: string, content: string) {
      const target = this.messages.find((item) => item.id === messageId)
      if (!target) {
        return
      }
      target.content = content
    },
    finishAssistantMessage() {
      this.isStreaming = false
    },
    async ensureActiveSession(accessToken: string): Promise<string> {
      if (this.activeSessionId) {
        return this.activeSessionId
      }
      const sessionId = await createChatSession(accessToken)
      this.activeSessionId = sessionId
      return sessionId
    },
    async sendMessage(rawText?: string) {
      const text = (rawText ?? this.draft).trim()
      if (!text || this.isStreaming) {
        return
      }

      const authStore = useAuthStore()
      const uiStore = useUiStore()
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
        const sessionId = await this.ensureActiveSession(accessToken)
        const controller = new AbortController()
        this.streamAbortController = controller

        await streamChatAnswer(
          accessToken,
          {
            session_id: sessionId,
            question: text,
            search_mode: uiStore.searchMode,
          },
          {
            onMeta: (data) => {
              if (!data || typeof data !== 'object') {
                return
              }
              const payload = data as Record<string, unknown>
              const streamSessionId = payload.session_id
              if (typeof streamSessionId === 'string' && streamSessionId.trim()) {
                this.activeSessionId = streamSessionId
              }
            },
            onToken: (token) => {
              this.appendAssistantToken(assistantId, token)
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
        const fallbackMessage = 'Xin loi, chua the tao cau tra loi luc nay. Vui long thu lai.'
        const safeMessage = this.streamError || message || fallbackMessage
        this.setAssistantMessageContent(assistantId, safeMessage)
      } finally {
        this.streamAbortController = null
        if (!this.messages.find((item) => item.id === assistantId)?.content.trim()) {
          this.setAssistantMessageContent(assistantId, 'Xin loi, chua co du lieu phan hoi tu he thong.')
        }
        this.finishAssistantMessage()
      }
    },
  },
})
