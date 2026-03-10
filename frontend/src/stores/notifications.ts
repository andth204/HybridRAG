import { defineStore } from 'pinia'

export type NotificationSource = 'documents' | 'users'
export type NotificationLevel = 'info' | 'success' | 'warning' | 'error'

export interface NotificationItem {
  id: string
  source: NotificationSource
  level: NotificationLevel
  message: string
  createdAt: string
  isRead: boolean
}

const STORAGE_KEY_PREFIX = 'hybridrag.notifications'
const MAX_NOTIFICATIONS = 100

function makeId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 100_000)}`
}

function buildStorageKey(userId: string | null | undefined): string {
  const suffix = userId?.trim() || 'guest'
  return `${STORAGE_KEY_PREFIX}.${suffix}`
}

function parseStoredNotifications(raw: string | null): NotificationItem[] {
  if (!raw) {
    return []
  }

  try {
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed
      .filter((item): item is NotificationItem => {
        return (
          item &&
          typeof item.id === 'string' &&
          typeof item.message === 'string' &&
          typeof item.createdAt === 'string' &&
          typeof item.isRead === 'boolean' &&
          (item.source === 'documents' || item.source === 'users') &&
          (item.level === 'info' ||
            item.level === 'success' ||
            item.level === 'warning' ||
            item.level === 'error')
        )
      })
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, MAX_NOTIFICATIONS)
  } catch {
    return []
  }
}

export const useNotificationsStore = defineStore('notifications', {
  state: () => ({
    items: [] as NotificationItem[],
    currentStorageKey: buildStorageKey(null),
  }),
  getters: {
    unreadCount(state): number {
      return state.items.filter((item) => !item.isRead).length
    },
  },
  actions: {
    hydrateForUser(userId: string | null | undefined) {
      this.currentStorageKey = buildStorageKey(userId)
      this.items = parseStoredNotifications(localStorage.getItem(this.currentStorageKey))
    },
    persist() {
      localStorage.setItem(this.currentStorageKey, JSON.stringify(this.items.slice(0, MAX_NOTIFICATIONS)))
    },
    pushNotification(payload: {
      source: NotificationSource
      level: NotificationLevel
      message: string
      createdAt?: string
    }) {
      const message = payload.message.trim()
      if (!message) {
        return
      }
      this.items.unshift({
        id: makeId('notice'),
        source: payload.source,
        level: payload.level,
        message,
        createdAt: payload.createdAt || new Date().toISOString(),
        isRead: false,
      })
      if (this.items.length > MAX_NOTIFICATIONS) {
        this.items = this.items.slice(0, MAX_NOTIFICATIONS)
      }
      this.persist()
    },
    markAsRead(notificationId: string) {
      const target = this.items.find((item) => item.id === notificationId)
      if (!target || target.isRead) {
        return
      }
      target.isRead = true
      this.persist()
    },
    markAllAsRead() {
      let changed = false
      for (const item of this.items) {
        if (!item.isRead) {
          item.isRead = true
          changed = true
        }
      }
      if (changed) {
        this.persist()
      }
    },
    remove(notificationId: string) {
      const nextItems = this.items.filter((item) => item.id !== notificationId)
      if (nextItems.length === this.items.length) {
        return
      }
      this.items = nextItems
      this.persist()
    },
    clearAll() {
      this.items = []
      this.persist()
    },
  },
})
