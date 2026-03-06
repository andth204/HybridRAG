import axios from 'axios'
import { streamChat, type ChatStreamHandlers, type ChatStreamPayload } from '@/services/sseClient'

interface BackendChatSessionResponse {
  id: string
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

export async function createChatSession(accessToken: string, title?: string): Promise<string> {
  const payload = title?.trim() ? { title: title.trim() } : {}
  const { data } = await chatApiClient.post<BackendChatSessionResponse>('/chat/sessions', payload, {
    headers: authHeaders(accessToken),
  })
  return data.id
}

export async function streamChatAnswer(
  accessToken: string,
  payload: ChatStreamPayload,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  await streamChat(buildEndpoint('/chat/answer/stream'), payload, handlers, signal, accessToken)
}
