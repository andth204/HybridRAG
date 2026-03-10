import axios from 'axios'
import { fetchEventSource, type EventSourceMessage } from '@microsoft/fetch-event-source'

export interface BackendFileAcceptedResponse {
  bucket: string
  key: string
  scoped_key: string
  action: string
  message: string
  status_endpoint: string
}

export interface BackendFileListItem {
  bucket: string
  key: string
  scoped_key: string
  status: string
  size_bytes: number
  file_id?: string | null
  etag?: string | null
  version_id?: string | null
  updated_at?: string | null
}

export interface BackendFilesListResponse {
  items: BackendFileListItem[]
}

export interface BackendFileStatusEvent {
  bucket: string
  key: string
  scoped_key: string
  action: string
  result: string
  message: string
  chunks?: number | null
  etag?: string | null
  version_id?: string | null
  file_id?: string | null
  reason?: string | null
  ts?: number | null
}

export interface FileStatusStreamHandlers {
  onReady?: (data: unknown) => void
  onStatus?: (event: BackendFileStatusEvent) => void
  onPing?: (data: unknown) => void
  onError?: (message: string) => void
}

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, '')
}

const apiBaseUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL?.trim() || '/api/v1')

const filesApiClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 30_000,
})

function authHeaders(accessToken: string) {
  return {
    Authorization: `Bearer ${accessToken}`,
  }
}

function buildEndpoint(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${apiBaseUrl}${normalizedPath}`
}

function safeParseJson(value: string): unknown {
  try {
    return JSON.parse(value)
  } catch {
    return value
  }
}

function isFileStatusEvent(value: unknown): value is BackendFileStatusEvent {
  if (!value || typeof value !== 'object') {
    return false
  }
  const payload = value as Record<string, unknown>
  return (
    typeof payload.key === 'string' &&
    typeof payload.scoped_key === 'string' &&
    typeof payload.action === 'string' &&
    typeof payload.result === 'string' &&
    typeof payload.message === 'string'
  )
}

function handleStreamMessage(message: EventSourceMessage, handlers: FileStatusStreamHandlers) {
  if (message.event === 'ready') {
    handlers.onReady?.(safeParseJson(message.data))
    return
  }

  if (message.event === 'ping') {
    handlers.onPing?.(safeParseJson(message.data))
    return
  }

  if (message.event === 'status') {
    const parsed = safeParseJson(message.data)
    if (isFileStatusEvent(parsed)) {
      handlers.onStatus?.(parsed)
    }
    return
  }

  if (message.event === 'error') {
    const parsed = safeParseJson(message.data)
    if (typeof parsed === 'string') {
      handlers.onError?.(parsed || 'Unknown file stream error')
      return
    }
    if (parsed && typeof parsed === 'object') {
      const payload = parsed as Record<string, unknown>
      if (typeof payload.message === 'string') {
        handlers.onError?.(payload.message || 'Unknown file stream error')
        return
      }
    }
    handlers.onError?.('Unknown file stream error')
  }
}

export async function listFiles(accessToken: string): Promise<BackendFileListItem[]> {
  const { data } = await filesApiClient.get<BackendFilesListResponse>('/files', {
    headers: authHeaders(accessToken),
  })
  return data.items ?? []
}

export async function uploadFileForIndex(
  accessToken: string,
  file: File,
): Promise<BackendFileAcceptedResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const { data } = await filesApiClient.post<BackendFileAcceptedResponse>('/files/index', formData, {
    headers: authHeaders(accessToken),
    timeout: 120_000,
  })
  return data
}

export async function requestFileReindex(
  accessToken: string,
  key: string,
): Promise<BackendFileAcceptedResponse> {
  const formData = new FormData()
  formData.append('key', key)

  const { data } = await filesApiClient.post<BackendFileAcceptedResponse>('/files/index', formData, {
    headers: authHeaders(accessToken),
    timeout: 30_000,
  })
  return data
}

export async function deleteFileByKey(accessToken: string, key: string): Promise<BackendFileAcceptedResponse> {
  const encodedKey = key
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/')

  const { data } = await filesApiClient.delete<BackendFileAcceptedResponse>(`/files/${encodedKey}`, {
    headers: authHeaders(accessToken),
    timeout: 30_000,
  })
  return data
}

export async function streamFileStatusEvents(
  accessToken: string,
  handlers: FileStatusStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  await fetchEventSource(buildEndpoint('/files/events/stream'), {
    method: 'GET',
    signal,
    openWhenHidden: true,
    headers: {
      Accept: 'text/event-stream',
      ...authHeaders(accessToken),
    },
    async onopen(response) {
      if (response.ok) {
        return
      }
      let errorMessage = `File status stream failed with status ${response.status}`
      try {
        const text = (await response.text()).trim()
        if (text) {
          errorMessage = text
        }
      } catch {
        // Keep fallback message.
      }
      throw new Error(errorMessage)
    },
    onmessage(message) {
      handleStreamMessage(message, handlers)
    },
    onerror(error) {
      handlers.onError?.(error instanceof Error ? error.message : 'Unknown file stream error')
      throw error
    },
  })
}
