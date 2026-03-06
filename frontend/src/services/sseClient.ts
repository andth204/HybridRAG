import { fetchEventSource, type EventSourceMessage } from '@microsoft/fetch-event-source'

export interface ChatStreamPayload {
  question: string
  session_id: string
  search_mode?: string
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
  if (event.event === 'meta' || event.event === 'start') {
    handlers.onMeta?.(safeParseJson(event.data))
    return
  }

  if (event.event === 'token') {
    handlers.onToken?.(event.data)
    return
  }

  if (event.event === 'chunk') {
    const parsed = safeParseJson(event.data)
    if (typeof parsed === 'string') {
      handlers.onToken?.(parsed)
      return
    }
    if (typeof parsed === 'object' && parsed) {
      const payload = parsed as Record<string, unknown>
      if (typeof payload.content === 'string') {
        handlers.onToken?.(payload.content)
        return
      }
    }
    return
  }

  if (event.event === 'done') {
    handlers.onDone?.()
    return
  }

  if (event.event === 'error') {
    const parsed = safeParseJson(event.data)
    if (typeof parsed === 'string') {
      handlers.onError?.(parsed || 'Unknown stream error')
      return
    }
    if (typeof parsed === 'object' && parsed) {
      const payload = parsed as Record<string, unknown>
      if (typeof payload.message === 'string') {
        handlers.onError?.(payload.message || 'Unknown stream error')
        return
      }
    }
    handlers.onError?.('Unknown stream error')
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
