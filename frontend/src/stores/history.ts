import { defineStore } from 'pinia'

export interface HistoryItem {
  id: number
  group: string
  title: string
  preview: string
  time: string
  active?: boolean
}

const initialHistory: HistoryItem[] = [
  {
    id: 1,
    group: 'Today',
    title: 'Summarize AI updates for this week',
    preview: 'Give me a short bullet list of major AI updates this week.',
    time: '2:41 PM',
    active: true,
  },
  {
    id: 2,
    group: 'Today',
    title: 'Write product description for RAG app',
    preview: 'Write a product description for an enterprise RAG assistant.',
    time: '11:07 AM',
  },
  {
    id: 3,
    group: 'Yesterday',
    title: 'Debug Flask /api/users endpoint',
    preview: 'Help debug a 500 error on the endpoint that returns users.',
    time: '6:55 PM',
  },
  {
    id: 4,
    group: 'Previous 7 days',
    title: 'Build content plan for product launch',
    preview: 'Plan a 7-day content roadmap for a product launch campaign.',
    time: 'Mon',
  },
]

export const useHistoryStore = defineStore('history', {
  state: () => ({
    searchTerm: '',
    deletedIds: new Set<number>(),
    items: initialHistory as HistoryItem[],
    activeItemId: initialHistory.find((item) => item.active)?.id ?? null,
  }),
  getters: {
    filteredItems: (state) => {
      return state.items.filter((item) => {
        if (state.deletedIds.has(item.id)) {
          return false
        }

        if (!state.searchTerm.trim()) {
          return true
        }

        const term = state.searchTerm.toLowerCase()
        return item.title.toLowerCase().includes(term) || item.preview.toLowerCase().includes(term)
      })
    },
    groupedItems(): Record<string, HistoryItem[]> {
      return this.filteredItems.reduce<Record<string, HistoryItem[]>>((result, item) => {
        const group = result[item.group] ?? []
        group.push(item)
        result[item.group] = group
        return result
      }, {})
    },
  },
  actions: {
    setSearchTerm(value: string) {
      this.searchTerm = value
    },
    setActiveItem(id: number | null) {
      this.activeItemId = id
    },
    deleteItem(id: number) {
      this.deletedIds.add(id)
      if (this.activeItemId === id) {
        const nextActive = this.items.find((item) => !this.deletedIds.has(item.id))
        this.activeItemId = nextActive?.id ?? null
      }
    },
  },
})
