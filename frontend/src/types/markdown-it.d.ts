declare module 'markdown-it' {
  export interface MarkdownItOptions {
    html?: boolean
    linkify?: boolean
    breaks?: boolean
    highlight?: (code: string, language: string) => string
  }

  export interface MarkdownItToken {
    attrSet(name: string, value: string): void
  }

  export interface MarkdownItRendererSelf {
    renderToken(tokens: unknown[], idx: number, options: unknown): string
  }

  export type MarkdownItRenderRule = (
    tokens: unknown[],
    idx: number,
    options: unknown,
    env: unknown,
    self: MarkdownItRendererSelf,
  ) => string

  export interface MarkdownItRenderer {
    rules: Record<string, MarkdownItRenderRule | undefined>
  }

  export interface MarkdownItUtils {
    escapeHtml(source: string): string
  }

  export default class MarkdownIt {
    constructor(options?: MarkdownItOptions)
    render(source: string): string
    renderer: MarkdownItRenderer
    utils: MarkdownItUtils
  }
}
