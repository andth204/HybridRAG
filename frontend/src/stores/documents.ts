import { defineStore } from 'pinia'

export type DocumentStatus = 'indexed' | 'indexing' | 'deleting' | 'failed'

export interface DocumentItem {
  id: string
  key: string
  fileId: string | null
  status: DocumentStatus
  progress: number
  sizeBytes: number
  updatedAt: string
}

export type KafkaEventType = 'file_indexed' | 'file_deleted' | 'file_failed'

export interface KafkaStatusEvent {
  eventType: KafkaEventType
  key: string
  fileId?: string
  message?: string
}

export interface UiNotification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  message: string
}

function makeId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 100_000)}`
}

function nowIso(): string {
  return new Date().toISOString()
}

function shouldSimulateFailure(rate = 0.22): boolean {
  return Math.random() < rate
}

const indexingTimerMap = new Map<string, number>()

function stopIndexingTimer(docId: string) {
  const timerId = indexingTimerMap.get(docId)
  if (timerId !== undefined) {
    window.clearInterval(timerId)
    indexingTimerMap.delete(docId)
  }
}

function makeKeyUnique(rawKey: string, existing: DocumentItem[]): string {
  const base = rawKey.trim() || 'untitled-file'
  if (!existing.some((item) => item.key === base)) {
    return base
  }

  let attempt = 2
  while (existing.some((item) => item.key === `${base} (${attempt})`)) {
    attempt += 1
  }
  return `${base} (${attempt})`
}

export const useDocumentsStore = defineStore('documents', {
  state: () => ({
    searchTerm: '',
    isUploading: false,
    documents: [
      {
        id: makeId('doc'),
        key: 'de-an-tuyen-sinh-2026.pdf',
        fileId: 'f_001x8',
        status: 'indexed' as DocumentStatus,
        progress: 100,
        sizeBytes: 1_448_200,
        updatedAt: nowIso(),
      },
      {
        id: makeId('doc'),
        key: 'hoc-phi-cac-nganh.xlsx',
        fileId: 'f_003q4',
        status: 'indexed' as DocumentStatus,
        progress: 100,
        sizeBytes: 324_800,
        updatedAt: nowIso(),
      },
      {
        id: makeId('doc'),
        key: 'chuong-trinh-dao-tao-cntt.docx',
        fileId: null,
        status: 'indexing' as DocumentStatus,
        progress: 42,
        sizeBytes: 992_000,
        updatedAt: nowIso(),
      },
      {
        id: makeId('doc'),
        key: 'bao-cao-chat-luong-co-so-dao-tao.pdf',
        fileId: null,
        status: 'failed' as DocumentStatus,
        progress: 71,
        sizeBytes: 2_104_500,
        updatedAt: nowIso(),
      },
    ] as DocumentItem[],
    notifications: [] as UiNotification[],
  }),
  getters: {
    filteredDocuments(state): DocumentItem[] {
      const term = state.searchTerm.trim().toLowerCase()
      const docs = [...state.documents].sort((a, b) => {
        return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
      })

      if (!term) {
        return docs
      }

      return docs.filter((item) => item.key.toLowerCase().includes(term))
    },
    indexingCount(state): number {
      return state.documents.filter((item) => item.status === 'indexing').length
    },
    indexedCount(state): number {
      return state.documents.filter((item) => item.status === 'indexed').length
    },
    failedCount(state): number {
      return state.documents.filter((item) => item.status === 'failed').length
    },
  },
  actions: {
    setSearchTerm(value: string) {
      this.searchTerm = value
    },
    pushNotification(type: UiNotification['type'], message: string) {
      this.notifications.push({
        id: makeId('notice'),
        type,
        message,
      })
    },
    consumeNotifications(): UiNotification[] {
      const notices = [...this.notifications]
      this.notifications = []
      return notices
    },
    handleKafkaStatusEvent(event: KafkaStatusEvent) {
      const target = this.documents.find((item) => item.key === event.key)
      if (!target && event.eventType !== 'file_deleted') {
        return
      }

      if (event.eventType === 'file_indexed') {
        if (!target) {
          return
        }
        target.status = 'indexed'
        target.progress = 100
        target.fileId = event.fileId ?? target.fileId ?? `f_${Math.floor(Math.random() * 100_000)}`
        target.updatedAt = nowIso()
        stopIndexingTimer(target.id)
        this.pushNotification('success', event.message ?? `Indexed "${target.key}" successfully.`)
        return
      }

      if (event.eventType === 'file_failed') {
        if (!target) {
          return
        }
        target.status = 'failed'
        target.progress = Math.max(1, Math.min(99, target.progress))
        target.updatedAt = nowIso()
        stopIndexingTimer(target.id)
        this.pushNotification('error', event.message ?? `Indexing failed for "${target.key}".`)
        return
      }

      if (target) {
        stopIndexingTimer(target.id)
      }
      this.documents = this.documents.filter((item) => item.key !== event.key)
      this.pushNotification('success', event.message ?? `Deleted "${event.key}" successfully.`)
    },
    startMockIndexing(
      target: DocumentItem,
      options?: {
        failRate?: number
        onSuccess?: string
        onFail?: string
      },
    ) {
      const failRate = options?.failRate ?? 0.22
      const durationMs = Math.min(18_000, Math.max(4_000, 2_600 + Math.round(target.sizeBytes / 1200)))
      const startedAt = Date.now()

      stopIndexingTimer(target.id)
      target.status = 'indexing'
      target.progress = Math.max(0, Math.min(97, target.progress))
      target.updatedAt = nowIso()

      const tickId = window.setInterval(() => {
        const active = this.documents.find((item) => item.id === target.id)
        if (!active || active.status !== 'indexing') {
          stopIndexingTimer(target.id)
          return
        }

        const elapsed = Date.now() - startedAt
        const ratio = Math.min(0.98, elapsed / durationMs)
        const baseProgress = Math.floor(ratio * 100)
        const nextProgress = Math.max(active.progress, Math.min(98, baseProgress))
        active.progress = nextProgress
        active.updatedAt = nowIso()

        if (elapsed < durationMs) {
          return
        }

        stopIndexingTimer(target.id)
        if (shouldSimulateFailure(failRate)) {
          this.handleKafkaStatusEvent({
            eventType: 'file_failed',
            key: active.key,
            message: options?.onFail ?? `Indexing failed for "${active.key}".`,
          })
          return
        }

        this.handleKafkaStatusEvent({
          eventType: 'file_indexed',
          key: active.key,
          fileId: active.fileId ?? `f_${Math.floor(Math.random() * 100_000)}`,
          message: options?.onSuccess ?? `File "${active.key}" indexed and stored successfully.`,
        })
      }, 320)

      indexingTimerMap.set(target.id, tickId)
    },
    resumePendingIndexing() {
      for (const item of this.documents) {
        if (item.status !== 'indexing' || indexingTimerMap.has(item.id)) {
          continue
        }
        this.startMockIndexing(item, {
          onSuccess: `Indexing completed for "${item.key}".`,
          onFail: `Indexing failed for "${item.key}".`,
        })
      }
    },
    async uploadFiles(files: File[]) {
      if (!files.length) {
        return
      }

      this.isUploading = true
      try {
        for (const file of files) {
          const uniqueKey = makeKeyUnique(file.name, this.documents)
          const item: DocumentItem = {
            id: makeId('doc'),
            key: uniqueKey,
            fileId: null,
            status: 'indexing',
            progress: 0,
            sizeBytes: file.size,
            updatedAt: nowIso(),
          }
          this.documents.unshift(item)

          this.pushNotification(
            'info',
            `Uploaded "${item.key}". Waiting for indexing status from backend event stream...`,
          )

          // UI-only simulation for now. Replace this with real backend Kafka progress/status events.
          this.startMockIndexing(item)
        }
      } finally {
        this.isUploading = false
      }
    },
    requestReindex(docId: string) {
      const target = this.documents.find((item) => item.id === docId)
      if (!target) {
        return
      }

      target.progress = 0
      this.pushNotification('info', `Re-index request sent for "${target.key}".`)

      this.startMockIndexing(target, {
        onSuccess: `Re-index completed for "${target.key}".`,
        onFail: `Re-index failed for "${target.key}".`,
      })
    },
    requestDelete(docId: string) {
      const target = this.documents.find((item) => item.id === docId)
      if (!target) {
        return
      }

      stopIndexingTimer(target.id)
      target.status = 'deleting'
      target.updatedAt = nowIso()
      this.pushNotification('warning', `Delete request accepted for "${target.key}".`)

      const delayMs = 900 + Math.floor(Math.random() * 1300)
      window.setTimeout(() => {
        this.handleKafkaStatusEvent({
          eventType: 'file_deleted',
          key: target.key,
          message: `Delete event processed for "${target.key}".`,
        })
      }, delayMs)
    },
  },
})
