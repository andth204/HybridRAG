import { defineStore } from 'pinia'

export type ThemeMode = 'light' | 'dark'
export type MainView = 'chat' | 'history' | 'documents' | 'users' | 'statistics'

const THEME_STORAGE_KEY = 'hybridrag.theme'
const SIDEBAR_STORAGE_KEY = 'hybridrag.sidebar-collapsed'
const RIGHT_PANEL_STORAGE_KEY = 'hybridrag.right-panel-collapsed'

export const useUiStore = defineStore('ui', {
  state: () => ({
    theme: 'light' as ThemeMode,
    mainView: 'chat' as MainView,
    isSidebarCollapsed: false,
    isRightPanelCollapsed: false,
  }),
  getters: {
    isDark: (state) => state.theme === 'dark',
  },
  actions: {
    initTheme() {
      const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode | null
      this.setTheme(savedTheme === 'dark' ? 'dark' : 'light')
      this.isSidebarCollapsed = localStorage.getItem(SIDEBAR_STORAGE_KEY) === '1'
      this.isRightPanelCollapsed = localStorage.getItem(RIGHT_PANEL_STORAGE_KEY) === '1'
    },
    setTheme(theme: ThemeMode) {
      this.theme = theme
      localStorage.setItem(THEME_STORAGE_KEY, theme)
      document.documentElement.classList.toggle('dark', theme === 'dark')
    },
    setSidebarCollapsed(value: boolean) {
      this.isSidebarCollapsed = value
      localStorage.setItem(SIDEBAR_STORAGE_KEY, value ? '1' : '0')
    },
    toggleSidebar() {
      this.setSidebarCollapsed(!this.isSidebarCollapsed)
    },
    setRightPanelCollapsed(value: boolean) {
      this.isRightPanelCollapsed = value
      localStorage.setItem(RIGHT_PANEL_STORAGE_KEY, value ? '1' : '0')
    },
    toggleRightPanel() {
      this.setRightPanelCollapsed(!this.isRightPanelCollapsed)
    },
    switchMainView(view: MainView) {
      this.mainView = view
    },
  },
})
