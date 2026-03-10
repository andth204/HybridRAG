import axios from 'axios'
import { streamChat, type ChatStreamHandlers, type ChatStreamPayload } from '@/services/sseClient'

export interface BackendChatSession {
  id: string
  title: string | null
  created_at: string
  updated_at: string
}

export interface BackendChatSessionResponse {
  items: BackendChatSession[]
}

export interface BackendChatMessage {
  id: string
  session_id: string
  role: string
  content: string
  metadata?: Record<string, unknown> | null
  created_at: string
}

export interface BackendChatMessagesResponse {
  items: BackendChatMessage[]
}

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, '')
}

const apiBaseUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL?.trim() || '/api/v1')

const chatApiClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 15_000,
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

function parseDownloadFilename(contentDisposition: string | undefined, fallback: string): string {
  const rawHeader = contentDisposition?.trim()
  if (!rawHeader) {
    return fallback
  }

  const utf8Match = rawHeader.match(/filename\*\s*=\s*UTF-8''([^;]+)/i)
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1]).trim() || fallback
    } catch {
      return fallback
    }
  }

  const asciiMatch = rawHeader.match(/filename\s*=\s*"?([^";]+)"?/i)
  if (asciiMatch?.[1]) {
    return asciiMatch[1].trim() || fallback
  }

  return fallback
}

export async function createChatSession(accessToken: string, title?: string): Promise<string> {
  const payload = title?.trim() ? { title: title.trim() } : {}
  const { data } = await chatApiClient.post<BackendChatSession>('/chat/sessions', payload, {
    headers: authHeaders(accessToken),
  })
  return data.id
}

export async function createChatSessionWithData(
  accessToken: string,
  title?: string,
): Promise<BackendChatSession> {
  const payload = title?.trim() ? { title: title.trim() } : {}
  const { data } = await chatApiClient.post<BackendChatSession>('/chat/sessions', payload, {
    headers: authHeaders(accessToken),
  })
  return data
}

export async function listChatSessions(accessToken: string): Promise<BackendChatSession[]> {
  const { data } = await chatApiClient.get<BackendChatSessionResponse>('/chat/sessions', {
    headers: authHeaders(accessToken),
  })
  return data.items ?? []
}

export async function listChatSessionMessages(
  accessToken: string,
  sessionId: string,
): Promise<BackendChatMessage[]> {
  const encodedSessionId = encodeURIComponent(sessionId)
  const { data } = await chatApiClient.get<BackendChatMessagesResponse>(
    `/chat/sessions/${encodedSessionId}/messages`,
    {
      headers: authHeaders(accessToken),
    },
  )
  return data.items ?? []
}

export async function deleteChatSession(accessToken: string, sessionId: string): Promise<void> {
  const encodedSessionId = encodeURIComponent(sessionId)
  await chatApiClient.delete(`/chat/sessions/${encodedSessionId}`, {
    headers: authHeaders(accessToken),
  })
}

export async function downloadChatReference(accessToken: string, referenceName: string): Promise<void> {
  const normalizedName = referenceName.trim()
  if (!normalizedName) {
    throw new Error('Reference file name is empty.')
  }

  const encodedReferenceName = encodeURIComponent(normalizedName)
  const response = await chatApiClient.get(`/files/by-name/${encodedReferenceName}/download`, {
    headers: authHeaders(accessToken),
    responseType: 'blob',
  })

  const blob = response.data instanceof Blob ? response.data : new Blob([response.data])
  const downloadName = parseDownloadFilename(response.headers['content-disposition'], normalizedName)
  const objectUrl = window.URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = objectUrl
  anchor.download = downloadName
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  window.URL.revokeObjectURL(objectUrl)
}

export async function streamChatAnswer(
  accessToken: string,
  payload: ChatStreamPayload,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  await streamChat(buildEndpoint('/chat/answer/stream'), payload, handlers, signal, accessToken)
}
