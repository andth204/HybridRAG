import { fetchEventSource, type EventSourceMessage } from '@microsoft/fetch-event-source'

export interface ChatStreamPayload {
  message: string
  sessionId?: string
}

export interface ChatStreamHandlers {
  onMeta?: (data: unknown) => void
  onToken?: (token: string) => void
  onDone?: () => void
  onError?: (error: string) => void
}

function safeParseJson(value: string): unknown {
  try {
    return JSON.parse(value)
  } catch {
    return value
  }
}

function handleMessage(event: EventSourceMessage, handlers: ChatStreamHandlers) {
  if (event.event === 'meta') {
    handlers.onMeta?.(safeParseJson(event.data))
    return
  }

  if (event.event === 'token') {
    handlers.onToken?.(event.data)
    return
  }

  if (event.event === 'done') {
    handlers.onDone?.()
    return
  }

  if (event.event === 'error') {
    handlers.onError?.(event.data || 'Unknown stream error')
  }
}

export async function streamChat(
  endpoint: string,
  payload: ChatStreamPayload,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
  accessToken?: string,
) {
  await fetchEventSource(endpoint, {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    },
    body: JSON.stringify(payload),
    onmessage(message) {
      handleMessage(message, handlers)
    },
    onerror(error) {
      handlers.onError?.(error instanceof Error ? error.message : 'Unknown fetch error')
      throw error
    },
  })
}
