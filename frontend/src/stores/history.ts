import { defineStore } from 'pinia'
import { deleteChatSession, listChatSessions, type BackendChatSession } from '@/services/chatApi'

const UNTITLED_SESSION_TITLE = 'Untitled conversation'

export interface HistoryItem {
  id: string
  group: string
  title: string
  preview: string
  time: string
  createdAt: string
  updatedAt: string
}

function normalizeTitle(rawTitle: string | null | undefined): string {
  const value = (rawTitle ?? '').trim()
  return value || UNTITLED_SESSION_TITLE
}

function shorten(value: string, maxLength = 96): string {
  const text = value.trim()
  if (!text) {
    return ''
  }
  if (text.length <= maxLength) {
    return text
  }
  return `${text.slice(0, maxLength - 1)}...`
}

function parseHistoryDate(rawValue: string): Date | null {
  const value = rawValue.trim()
  if (!value) {
    return null
  }

  const hasTimeZone = /(?:[zZ]|[+-]\d{2}:\d{2})$/.test(value)
  const normalized = hasTimeZone ? value : `${value}Z`
  const parsed = new Date(normalized)
  if (Number.isNaN(parsed.getTime())) {
    return null
  }
  return parsed
}

function formatTime(isoValue: string): string {
  const date = parseHistoryDate(isoValue)
  if (!date) {
    return '--:--'
  }
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function formatGroup(isoValue: string): string {
  const date = parseHistoryDate(isoValue)
  if (!date) {
    return 'Unknown'
  }

  const now = new Date()
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const startTarget = new Date(date.getFullYear(), date.getMonth(), date.getDate())
  const diffDays = Math.floor((startToday.getTime() - startTarget.getTime()) / 86_400_000)

  if (diffDays <= 0) {
    return 'Today'
  }
  if (diffDays === 1) {
    return 'Yesterday'
  }
  if (diffDays <= 7) {
    return 'Previous 7 days'
  }
  return date.toLocaleDateString([], { month: 'short', year: 'numeric' })
}

function toHistoryItem(session: BackendChatSession): HistoryItem {
  const createdAt = session.created_at || new Date().toISOString()
  const updatedAt = session.updated_at || createdAt
  const title = normalizeTitle(session.title)
  return {
    id: session.id,
    group: formatGroup(updatedAt),
    title,
    preview: shorten(title),
    time: formatTime(updatedAt),
    createdAt,
    updatedAt,
  }
}

function sortHistoryItems(items: HistoryItem[]): HistoryItem[] {
  return [...items].sort((a, b) => {
    const leftTime = parseHistoryDate(a.updatedAt)?.getTime() ?? 0
    const rightTime = parseHistoryDate(b.updatedAt)?.getTime() ?? 0
    return rightTime - leftTime
  })
}

function makeTitleFromQuestion(question: string): string {
  const compact = question.trim().replace(/\s+/g, ' ')
  if (!compact) {
    return UNTITLED_SESSION_TITLE
  }
  return shorten(compact, 72)
}

export const useHistoryStore = defineStore('history', {
  state: () => ({
    searchTerm: '',
    items: [] as HistoryItem[],
    activeItemId: null as string | null,
    hasLoaded: false,
    isLoading: false,
    loadRequestId: 0,
  }),
  getters: {
    filteredItems: (state) => {
      const term = state.searchTerm.trim().toLowerCase()
      if (!term) {
        return state.items
      }
      return state.items.filter((item) => {
        return item.title.toLowerCase().includes(term) || item.preview.toLowerCase().includes(term)
      })
    },
    groupedItems(): Record<string, HistoryItem[]> {
      return this.filteredItems.reduce<Record<string, HistoryItem[]>>((result, item) => {
        const groupItems = result[item.group] ?? []
        groupItems.push(item)
        result[item.group] = groupItems
        return result
      }, {})
    },
    recentItems(): HistoryItem[] {
      return this.items.slice(0, 7)
    },
  },
  actions: {
    reset() {
      this.loadRequestId += 1
      this.searchTerm = ''
      this.items = []
      this.activeItemId = null
      this.hasLoaded = false
      this.isLoading = false
    },
    setSearchTerm(value: string) {
      this.searchTerm = value
    },
    setActiveItem(id: string | null) {
      this.activeItemId = id
    },
    async loadSessions(accessToken: string) {
      const token = accessToken.trim()
      if (!token) {
        this.reset()
        return
      }

      const requestId = this.loadRequestId + 1
      this.loadRequestId = requestId
      this.isLoading = true
      try {
        const sessions = await listChatSessions(token)
        if (this.loadRequestId !== requestId) {
          return
        }
        const mapped = sessions.map((session) => toHistoryItem(session))
        this.items = sortHistoryItems(mapped)
        if (this.activeItemId && !this.items.some((item) => item.id === this.activeItemId)) {
          this.activeItemId = null
        }
        this.hasLoaded = true
      } finally {
        if (this.loadRequestId === requestId) {
          this.isLoading = false
        }
      }
    },
    upsertSession(session: Partial<BackendChatSession> & { id: string }, titleHint?: string) {
      const nowIso = new Date().toISOString()
      const createdAt = session.created_at || nowIso
      const updatedAt = session.updated_at || nowIso
      const hintedTitle = (titleHint ?? '').trim()
      const baseTitle = session.title ?? (hintedTitle || UNTITLED_SESSION_TITLE)
      const item: HistoryItem = {
        id: session.id,
        createdAt,
        updatedAt,
        title: normalizeTitle(baseTitle),
        preview: shorten(normalizeTitle(baseTitle)),
        time: formatTime(updatedAt),
        group: formatGroup(updatedAt),
      }

      const index = this.items.findIndex((value) => value.id === session.id)
      if (index < 0) {
        this.items = sortHistoryItems([item, ...this.items])
        return
      }

      const next = [...this.items]
      const previous = next[index]
      if (!previous) {
        this.items = sortHistoryItems([item, ...this.items])
        return
      }

      const resolvedTitle =
        previous.title !== UNTITLED_SESSION_TITLE ? previous.title : item.title
      next[index] = {
        ...previous,
        ...item,
        title: resolvedTitle,
        preview: shorten(resolvedTitle),
      }
      this.items = sortHistoryItems(next)
    },
    touchSession(sessionId: string, question: string) {
      const title = makeTitleFromQuestion(question)
      this.upsertSession({ id: sessionId, updated_at: new Date().toISOString(), title }, title)
    },
    async deleteItem(accessToken: string, id: string) {
      await deleteChatSession(accessToken, id)
      this.items = this.items.filter((item) => item.id !== id)
      if (this.activeItemId === id) {
        this.activeItemId = null
      }
    },
  },
})
