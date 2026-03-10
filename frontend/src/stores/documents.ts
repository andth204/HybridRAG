import axios from 'axios'
import { defineStore } from 'pinia'
import { useAuthStore } from '@/stores/auth'
import { useNotificationsStore } from '@/stores/notifications'
import {
  deleteFileByKey,
  listFiles,
  requestFileReindex,
  streamFileStatusEvents,
  uploadFileForIndex,
  type BackendFileListItem,
  type BackendFileStatusEvent,
} from '@/services/filesApi'

export type DocumentStatus = 'indexed' | 'indexing' | 'deleting' | 'failed'

export interface DocumentItem {
  id: string
  key: string
  scopedKey: string
  fileId: string | null
  status: DocumentStatus
  progress: number
  sizeBytes: number
  updatedAt: string
  etag: string | null
  versionId: string | null
}

export type UiNotificationType = 'info' | 'success' | 'warning' | 'error'

const pseudoProgressTimerMap = new Map<string, number>()
const completionProgressTimerMap = new Map<string, number>()
const streamReconnectDelayMs = 2_500
const refreshDelayMs = 350
const liveRefreshIntervalMs = 1_500
let statusStreamAbortController: AbortController | null = null
let statusStreamReconnectTimerId: number | null = null
let documentsRefreshTimerId: number | null = null
let liveDocumentsRefreshTimerId: number | null = null

function extractApiErrorMessage(error: unknown, fallback: string): string {
  if (!axios.isAxiosError(error)) {
    return fallback
  }

  const detail = error.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) {
    return detail
  }

  const message = error.message?.trim()
  return message || fallback
}

function nowIso(): string {
  return new Date().toISOString()
}

function normalizeDocumentKey(raw: string): string {
  return raw.trim().replace(/^\/+/, '')
}

function scopeDocumentKey(userId: string, key: string): string {
  return `${userId}/${normalizeDocumentKey(key)}`
}

function toIsoFromEventTs(value: number | null | undefined): string {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return new Date(value * 1000).toISOString()
  }
  return nowIso()
}

function sortDocuments(items: DocumentItem[]): DocumentItem[] {
  return [...items].sort((a, b) => {
    const updatedDiff = new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    if (updatedDiff !== 0) {
      return updatedDiff
    }
    return a.key.localeCompare(b.key)
  })
}

function mapBackendStatus(status: string): DocumentStatus {
  if (status === 'indexed') {
    return 'indexed'
  }
  if (status === 'deleting') {
    return 'deleting'
  }
  return 'indexing'
}

function mapBackendItem(item: BackendFileListItem): DocumentItem {
  const status = mapBackendStatus(item.status)
  return {
    id: item.scoped_key,
    key: item.key,
    scopedKey: item.scoped_key,
    fileId: item.file_id ?? null,
    status,
    progress: status === 'indexed' ? 100 : 0,
    sizeBytes: Number(item.size_bytes || 0),
    updatedAt: item.updated_at || nowIso(),
    etag: item.etag ?? null,
    versionId: item.version_id ?? null,
  }
}

function buildLocalDocument(params: {
  userId: string
  key: string
  sizeBytes: number
  status: DocumentStatus
  updatedAt?: string
  progress?: number
  fileId?: string | null
  etag?: string | null
  versionId?: string | null
}): DocumentItem {
  const normalizedKey = normalizeDocumentKey(params.key)
  const status = params.status
  return {
    id: scopeDocumentKey(params.userId, normalizedKey),
    key: normalizedKey,
    scopedKey: scopeDocumentKey(params.userId, normalizedKey),
    fileId: params.fileId ?? null,
    status,
    progress: params.progress ?? (status === 'indexed' ? 100 : 0),
    sizeBytes: Math.max(0, params.sizeBytes),
    updatedAt: params.updatedAt || nowIso(),
    etag: params.etag ?? null,
    versionId: params.versionId ?? null,
  }
}

function clearReconnectTimer() {
  if (statusStreamReconnectTimerId !== null) {
    window.clearTimeout(statusStreamReconnectTimerId)
    statusStreamReconnectTimerId = null
  }
}

function clearRefreshTimer() {
  if (documentsRefreshTimerId !== null) {
    window.clearTimeout(documentsRefreshTimerId)
    documentsRefreshTimerId = null
  }
}

function clearLiveRefreshTimer() {
  if (liveDocumentsRefreshTimerId !== null) {
    window.clearInterval(liveDocumentsRefreshTimerId)
    liveDocumentsRefreshTimerId = null
  }
}

export const useDocumentsStore = defineStore('documents', {
  state: () => ({
    searchTerm: '',
    isUploading: false,
    isLoading: false,
    documents: [] as DocumentItem[],
    streamUserId: null as string | null,
  }),
  getters: {
    filteredDocuments(state): DocumentItem[] {
      const term = state.searchTerm.trim().toLowerCase()
      const docs = sortDocuments(state.documents)
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
    pushNotification(type: UiNotificationType, message: string, createdAt?: string) {
      const notificationsStore = useNotificationsStore()
      notificationsStore.pushNotification({
        source: 'documents',
        level: type,
        message,
        createdAt,
      })
    },
    stopPseudoProgress(docId: string) {
      const timerId = pseudoProgressTimerMap.get(docId)
      if (timerId !== undefined) {
        window.clearInterval(timerId)
        pseudoProgressTimerMap.delete(docId)
      }
    },
    stopCompletionProgress(docId: string) {
      const timerId = completionProgressTimerMap.get(docId)
      if (timerId !== undefined) {
        window.clearInterval(timerId)
        completionProgressTimerMap.delete(docId)
      }
    },
    startPseudoProgress(docId: string) {
      this.stopPseudoProgress(docId)
      this.stopCompletionProgress(docId)
      const target = this.documents.find((item) => item.id === docId)
      if (!target || target.status !== 'indexing') {
        return
      }
      target.progress = Math.max(0, Math.min(12, target.progress))
      const timerId = window.setInterval(() => {
        const active = this.documents.find((item) => item.id === docId)
        if (!active || active.status !== 'indexing') {
          this.stopPseudoProgress(docId)
          return
        }
        const increment = active.progress < 35 ? 4 : active.progress < 70 ? 3 : 1
        active.progress = Math.min(92, active.progress + increment)
      }, 700)
      pseudoProgressTimerMap.set(docId, timerId)
    },
    ensureLiveRefresh() {
      if (liveDocumentsRefreshTimerId !== null) {
        return
      }
      liveDocumentsRefreshTimerId = window.setInterval(() => {
        void this.refreshDocumentsFromSession()
      }, liveRefreshIntervalMs)
    },
    syncLiveRefresh(nextItems: DocumentItem[]) {
      const hasPending = nextItems.some((item) => item.status === 'indexing' || item.status === 'deleting')
      if (hasPending) {
        this.ensureLiveRefresh()
        return
      }
      clearLiveRefreshTimer()
    },
    completeProgressThen(docId: string, onDone: () => void) {
      this.stopPseudoProgress(docId)
      this.stopCompletionProgress(docId)
      const target = this.documents.find((item) => item.id === docId)
      if (!target) {
        onDone()
        return
      }
      if (target.progress >= 100) {
        onDone()
        return
      }
      target.progress = Math.max(target.progress, 92)
      const timerId = window.setInterval(() => {
        const active = this.documents.find((item) => item.id === docId)
        if (!active) {
          this.stopCompletionProgress(docId)
          return
        }
        const remaining = 100 - active.progress
        active.progress = Math.min(100, active.progress + Math.max(remaining, 2))
        if (active.progress >= 100) {
          this.stopCompletionProgress(docId)
          onDone()
        }
      }, 45)
      completionProgressTimerMap.set(docId, timerId)
    },
    applyDocuments(items: DocumentItem[]) {
      const nextItems = sortDocuments(items)
      const nextIds = new Set(nextItems.map((item) => item.id))
      for (const current of this.documents) {
        if (!nextIds.has(current.id)) {
          this.stopPseudoProgress(current.id)
          this.stopCompletionProgress(current.id)
        }
      }
      for (const item of nextItems) {
        if (item.status === 'indexing') {
          this.startPseudoProgress(item.id)
          continue
        }
        this.stopPseudoProgress(item.id)
        this.stopCompletionProgress(item.id)
        if (item.status === 'indexed') {
          item.progress = 100
        }
      }
      this.documents = nextItems
      this.syncLiveRefresh(nextItems)
    },
    mergeServerDocuments(serverItems: DocumentItem[]): DocumentItem[] {
      const currentByKey = new Map(this.documents.map((item) => [item.key, item]))
      const merged = serverItems.map((serverItem) => {
        const localItem = currentByKey.get(serverItem.key)
        if (!localItem) {
          return serverItem
        }
        if (serverItem.status === 'indexing' && localItem.status === 'indexing') {
          return {
            ...serverItem,
            progress: Math.max(localItem.progress, serverItem.progress),
          }
        }
        if (serverItem.status === 'deleting' && localItem.status === 'deleting') {
          return {
            ...serverItem,
            progress: localItem.progress,
          }
        }
        return serverItem
      })

      const mergedKeys = new Set(merged.map((item) => item.key))
      for (const localItem of this.documents) {
        if (mergedKeys.has(localItem.key)) {
          continue
        }
        if (localItem.status === 'indexing' || localItem.status === 'deleting' || localItem.status === 'failed') {
          merged.push(localItem)
        }
      }
      return merged
    },
    upsertDocument(nextItem: DocumentItem) {
      const next = [...this.documents]
      const index = next.findIndex((item) => item.key === nextItem.key)
      if (index >= 0) {
        next[index] = nextItem
      } else {
        next.unshift(nextItem)
      }
      this.applyDocuments(next)
    },
    removeDocumentByKey(key: string) {
      const normalizedKey = normalizeDocumentKey(key)
      const target = this.documents.find((item) => item.key === normalizedKey)
      if (target) {
        this.stopPseudoProgress(target.id)
      }
      this.applyDocuments(this.documents.filter((item) => item.key !== normalizedKey))
    },
    scheduleDocumentsRefresh() {
      clearRefreshTimer()
      documentsRefreshTimerId = window.setTimeout(() => {
        documentsRefreshTimerId = null
        void this.refreshDocumentsFromSession()
      }, refreshDelayMs)
    },
    async refreshDocumentsFromSession() {
      const authStore = useAuthStore()
      if (!authStore.isAuthenticated || authStore.currentUser?.role !== 'manager') {
        return
      }
      const ok = await authStore.ensureSession()
      if (!ok || !authStore.accessToken.trim()) {
        return
      }
      await this.loadDocuments(authStore.accessToken.trim())
    },
    async loadDocuments(accessToken: string) {
      this.isLoading = true
      try {
        const items = await listFiles(accessToken)
        const merged = this.mergeServerDocuments(items.map(mapBackendItem))
        this.applyDocuments(merged)
      } finally {
        this.isLoading = false
      }
    },
    disconnectStatusStream() {
      clearReconnectTimer()
      if (statusStreamAbortController) {
        statusStreamAbortController.abort()
        statusStreamAbortController = null
      }
      this.streamUserId = null
    },
    scheduleStatusStreamReconnect() {
      clearReconnectTimer()
      statusStreamReconnectTimerId = window.setTimeout(() => {
        statusStreamReconnectTimerId = null
        void this.connectStatusStream(true)
      }, streamReconnectDelayMs)
    },
    async connectStatusStream(force = false) {
      const authStore = useAuthStore()
      const currentUser = authStore.currentUser
      if (!authStore.isAuthenticated || !currentUser || currentUser.role !== 'manager') {
        this.disconnectStatusStream()
        return
      }

      const sessionOk = await authStore.ensureSession()
      if (!sessionOk || !authStore.accessToken.trim()) {
        this.disconnectStatusStream()
        return
      }

      if (!force && statusStreamAbortController && this.streamUserId === currentUser.id) {
        return
      }

      this.disconnectStatusStream()
      const controller = new AbortController()
      statusStreamAbortController = controller
      this.streamUserId = currentUser.id

      void streamFileStatusEvents(
        authStore.accessToken.trim(),
        {
          onStatus: (event) => {
            this.handleStatusEvent(event)
          },
        },
        controller.signal,
      )
        .then(() => {
          if (!controller.signal.aborted) {
            this.scheduleStatusStreamReconnect()
          }
        })
        .catch(() => {
          if (!controller.signal.aborted) {
            this.scheduleStatusStreamReconnect()
            this.scheduleDocumentsRefresh()
          }
        })
    },
    handleStatusEvent(event: BackendFileStatusEvent) {
      const eventTime = toIsoFromEventTs(event.ts)
      const target = this.documents.find((item) => item.key === event.key)

      if (event.result === 'deleted') {
        this.removeDocumentByKey(event.key)
        this.pushNotification('success', event.message, eventTime)
        return
      }

      if (event.result === 'duplicated') {
        if (target) {
          const docId = target.id
          this.completeProgressThen(docId, () => {
            const current = this.documents.find((item) => item.id === docId)
            if (!current) {
              return
            }
            current.status = 'indexed'
            current.progress = 100
            current.fileId = event.file_id ?? current.fileId
            current.etag = event.etag ?? current.etag
            current.versionId = event.version_id ?? current.versionId
            current.updatedAt = eventTime
            this.applyDocuments([...this.documents])
          })
        } else {
          this.scheduleDocumentsRefresh()
        }
        this.pushNotification('warning', event.message, eventTime)
        return
      }

      if (event.result === 'failed' || event.result === 'skipped') {
        if (target) {
          target.status = 'failed'
          target.progress = Math.max(target.progress, 8)
          target.fileId = event.file_id ?? target.fileId
          target.etag = event.etag ?? target.etag
          target.versionId = event.version_id ?? target.versionId
          target.updatedAt = eventTime
          this.stopPseudoProgress(target.id)
          this.applyDocuments([...this.documents])
        } else {
          this.scheduleDocumentsRefresh()
        }
        this.pushNotification(event.result === 'failed' ? 'error' : 'warning', event.message, eventTime)
        return
      }

      if (event.result === 'success') {
        if (target) {
          const docId = target.id
          this.completeProgressThen(docId, () => {
            const current = this.documents.find((item) => item.id === docId)
            if (!current) {
              return
            }
            current.status = 'indexed'
            current.progress = 100
            current.fileId = event.file_id ?? current.fileId
            current.etag = event.etag ?? current.etag
            current.versionId = event.version_id ?? current.versionId
            current.updatedAt = eventTime
            this.applyDocuments([...this.documents])
          })
        } else {
          const nextItem = buildLocalDocument({
            userId: this.streamUserId || 'unknown',
            key: event.key,
            sizeBytes: 0,
            status: 'indexed',
            updatedAt: eventTime,
            progress: 100,
            fileId: event.file_id ?? null,
            etag: event.etag ?? null,
            versionId: event.version_id ?? null,
          })
          this.upsertDocument(nextItem)
          this.scheduleDocumentsRefresh()
        }
        this.pushNotification('success', event.message, eventTime)
      }
    },
    reset() {
      this.disconnectStatusStream()
      clearRefreshTimer()
      clearLiveRefreshTimer()
      for (const item of this.documents) {
        this.stopPseudoProgress(item.id)
        this.stopCompletionProgress(item.id)
      }
      this.documents = []
      this.isUploading = false
      this.isLoading = false
    },
    async syncWithSession() {
      const authStore = useAuthStore()
      const currentUser = authStore.currentUser
      if (!authStore.isAuthenticated || !currentUser || currentUser.role !== 'manager') {
        this.reset()
        return
      }

      const sessionOk = await authStore.ensureSession()
      if (!sessionOk || !authStore.accessToken.trim()) {
        this.reset()
        return
      }

      if (this.streamUserId && this.streamUserId !== currentUser.id) {
        this.reset()
      }

      await this.loadDocuments(authStore.accessToken.trim())
      await this.connectStatusStream()
    },
    async uploadFiles(files: File[]) {
      if (!files.length) {
        return
      }

      const authStore = useAuthStore()
      const currentUser = authStore.currentUser
      if (!authStore.isAuthenticated || !currentUser || currentUser.role !== 'manager') {
        return
      }

      const sessionOk = await authStore.ensureSession()
      if (!sessionOk || !authStore.accessToken.trim()) {
        return
      }

      this.isUploading = true
      try {
        for (const file of files) {
          const normalizedKey = normalizeDocumentKey(file.name || 'upload.bin')
          const previous = this.documents.find((item) => item.key === normalizedKey)
          const pendingItem = buildLocalDocument({
            userId: currentUser.id,
            key: normalizedKey,
            sizeBytes: file.size,
            status: 'indexing',
            updatedAt: nowIso(),
            progress: 6,
            fileId: previous?.fileId ?? null,
            etag: previous?.etag ?? null,
            versionId: previous?.versionId ?? null,
          })

          this.upsertDocument(pendingItem)
          this.startPseudoProgress(pendingItem.id)

          try {
            const accepted = await uploadFileForIndex(authStore.accessToken.trim(), file)
            const target = this.documents.find((item) => item.key === accepted.key)
            if (target) {
              target.id = accepted.scoped_key
              target.scopedKey = accepted.scoped_key
              target.updatedAt = nowIso()
              this.applyDocuments([...this.documents])
            }
          } catch (error) {
            if (previous) {
              this.upsertDocument(previous)
            } else {
              this.removeDocumentByKey(normalizedKey)
            }
            this.pushNotification('error', extractApiErrorMessage(error, `Upload failed for "${normalizedKey}".`))
          }
        }
      } finally {
        this.isUploading = false
      }
    },
    async requestReindex(docId: string) {
      const target = this.documents.find((item) => item.id === docId)
      if (!target) {
        return
      }

      const authStore = useAuthStore()
      if (!authStore.isAuthenticated || authStore.currentUser?.role !== 'manager') {
        return
      }

      const previous = { ...target }
      target.status = 'indexing'
      target.progress = 6
      target.updatedAt = nowIso()
      this.startPseudoProgress(target.id)
      this.applyDocuments([...this.documents])

      try {
        const sessionOk = await authStore.ensureSession()
        if (!sessionOk || !authStore.accessToken.trim()) {
          throw new Error('Session expired.')
        }
        await requestFileReindex(authStore.accessToken.trim(), target.key)
      } catch (error) {
        this.upsertDocument(previous)
        this.pushNotification('error', extractApiErrorMessage(error, `Re-index failed for "${target.key}".`))
      }
    },
    async requestDelete(docId: string) {
      const target = this.documents.find((item) => item.id === docId)
      if (!target) {
        return
      }

      const authStore = useAuthStore()
      if (!authStore.isAuthenticated || authStore.currentUser?.role !== 'manager') {
        return
      }

      const previous = { ...target }
      target.status = 'deleting'
      target.updatedAt = nowIso()
      this.stopPseudoProgress(target.id)
      this.applyDocuments([...this.documents])

      try {
        const sessionOk = await authStore.ensureSession()
        if (!sessionOk || !authStore.accessToken.trim()) {
          throw new Error('Session expired.')
        }
        await deleteFileByKey(authStore.accessToken.trim(), target.key)
      } catch (error) {
        this.upsertDocument(previous)
        this.pushNotification('error', extractApiErrorMessage(error, `Delete failed for "${target.key}".`))
      }
    },
  },
})
