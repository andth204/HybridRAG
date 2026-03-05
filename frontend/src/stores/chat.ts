import { defineStore } from 'pinia'

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

const mockResponses = [
  'I received your question. Please share more scope so the answer can be more precise.',
  'Clear. Start by splitting the request into 3 parts: input, retrieval, and output.',
  'Good idea. I can summarize this as implementation steps for your frontend backlog.',
  'I can also help you define the API contract between frontend and backend.',
]

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [] as ChatMessage[],
    draft: '',
    isStreaming: false,
  }),
  getters: {
    hasMessages: (state) => state.messages.length > 0,
  },
  actions: {
    setDraft(value: string) {
      this.draft = value
    },
    resetConversation() {
      this.messages = []
      this.draft = ''
      this.isStreaming = false
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
    finishAssistantMessage() {
      this.isStreaming = false
    },
    async sendMessage(rawText?: string) {
      const text = (rawText ?? this.draft).trim()
      if (!text || this.isStreaming) {
        return
      }

      this.pushUserMessage(text)
      this.draft = ''

      const assistantId = this.startAssistantMessage()
      const reply =
        mockResponses[Math.floor(Math.random() * mockResponses.length)] ??
        'I received your request and will reply after processing.'

      for (const char of reply) {
        await new Promise((resolve) => setTimeout(resolve, 12))
        this.appendAssistantToken(assistantId, char)
      }

      this.finishAssistantMessage()
    },
  },
})
