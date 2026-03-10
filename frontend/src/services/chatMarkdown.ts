import DOMPurify from 'dompurify'
import hljs from 'highlight.js'
import MarkdownIt from 'markdown-it'

export interface ChatReferenceItem {
  index: number
  fileName: string
}

export interface RenderedAssistantMessage {
  html: string
  references: ChatReferenceItem[]
}

function escapeHtml(source: string): string {
  return source
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

const md: MarkdownIt = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  highlight(code: string, language: string): string {
    if (language && hljs.getLanguage(language)) {
      return `<pre class="msg-code-block"><code class="hljs language-${language}">${hljs.highlight(code, {
        language,
        ignoreIllegals: true,
      }).value}</code></pre>`
    }

    return `<pre class="msg-code-block"><code class="hljs">${escapeHtml(code)}</code></pre>`
  },
})

const defaultLinkOpen =
  md.renderer.rules.link_open ??
  ((tokens: unknown[], idx: number, options: unknown, env: unknown, self: { renderToken: (tokens: unknown[], idx: number, options: unknown) => string }) =>
    self.renderToken(tokens, idx, options))

md.renderer.rules.link_open = (
  tokens: unknown[],
  idx: number,
  options: unknown,
  env: unknown,
  self: { renderToken: (tokens: unknown[], idx: number, options: unknown) => string },
) => {
  const token = tokens[idx] as { attrSet?: (name: string, value: string) => void } | undefined
  if (!token?.attrSet) {
    return defaultLinkOpen(tokens, idx, options, env, self)
  }
  token.attrSet('target', '_blank')
  token.attrSet('rel', 'noopener noreferrer')
  return defaultLinkOpen(tokens, idx, options, env, self)
}

md.renderer.rules.table_open = () => '<div class="msg-table-wrap"><table class="msg-table">'
md.renderer.rules.table_close = () => '</table></div>'

function normalizeLeadingMarkers(source: string): string {
  const lines = source.split(/\r?\n/)
  let inFence = false

  return lines
    .map((line) => {
      const trimmed = line.trimStart()
      if (trimmed.startsWith('```') || trimmed.startsWith('~~~')) {
        inFence = !inFence
        return line
      }

      if (inFence) {
        return line
      }

      if (/^\s*[+-]\s+\S/.test(line)) {
        return line.replace(/^(\s*)[+-](\s+)/, '$1•$2')
      }

      return line
    })
    .join('\n')
}

function normalizeReferenceLabel(value: string): string {
  return value
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .trim()
    .toLowerCase()
}

function splitReferenceSection(source: string): { body: string; references: ChatReferenceItem[] } {
  const normalized = source.trim()
  if (!normalized) {
    return { body: '', references: [] }
  }

  const lines = normalized.split(/\r?\n/)
  const headingIndex = lines.findIndex((line) => {
    const trimmed = line.trim()
    if (!trimmed.startsWith('[') || !trimmed.endsWith(']')) {
      return false
    }
    return normalizeReferenceLabel(trimmed.slice(1, -1)) === 'thong tin tham chieu'
  })

  if (headingIndex < 0) {
    return { body: normalized, references: [] }
  }

  const references = lines
    .slice(headingIndex + 1)
    .map((line) => line.match(/^\s*\[(\d+)\]\.\s+(.+?)\s*$/))
    .filter((match): match is RegExpMatchArray => Boolean(match))
    .map((match) => {
      const rawIndex = match[1] ?? ''
      const rawFileName = match[2] ?? ''
      return {
        index: Number(rawIndex),
        fileName: rawFileName.trim(),
      }
    })
    .filter((item) => Number.isFinite(item.index) && item.index > 0 && item.fileName.length > 0)

  if (!references.length) {
    return { body: normalized, references: [] }
  }

  return {
    body: lines.slice(0, headingIndex).join('\n').trim(),
    references,
  }
}

export function renderChatMarkdown(source: string): string {
  const normalized = source.trim()
  if (!normalized) {
    return '<p>...</p>'
  }

  const rendered = md.render(normalizeLeadingMarkers(normalized))
  return DOMPurify.sanitize(rendered, {
    ADD_ATTR: ['target', 'rel', 'class'],
  })
}

export function renderAssistantMessage(source: string): RenderedAssistantMessage {
  const { body, references } = splitReferenceSection(source)
  return {
    html: body ? renderChatMarkdown(body) : '',
    references,
  }
}
