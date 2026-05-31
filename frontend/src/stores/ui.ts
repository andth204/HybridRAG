import { defineStore } from 'pinia'

export type ThemeMode = 'light' | 'dark'
export type MainView = 'chat' | 'history' | 'documents' | 'users' | 'statistics'
export type AppLanguage = string
export type SpeechLanguage = string
export type SettingsTab = 'general' | 'account' | 'logout'

const THEME_STORAGE_KEY = 'hybridrag.theme'
const SIDEBAR_STORAGE_KEY = 'hybridrag.sidebar-collapsed'
const RIGHT_PANEL_STORAGE_KEY = 'hybridrag.right-panel-collapsed'
const LANGUAGE_STORAGE_KEY = 'hybridrag.language'
const SPEECH_LANGUAGE_STORAGE_KEY = 'hybridrag.speech-language'
const VOICE_STORAGE_KEY = 'hybridrag.voice'

export const useUiStore = defineStore('ui', {
  state: () => ({
    theme: 'light' as ThemeMode,
    mainView: 'chat' as MainView,
    isSidebarCollapsed: false,
    isRightPanelCollapsed: true,
    language: 'auto' as AppLanguage,
    speechLanguage: 'auto' as SpeechLanguage,
    preferredVoice: '',
    isSettingsOpen: false,
    settingsTab: 'general' as SettingsTab,
  }),
  getters: {
    isDark: (state) => state.theme === 'dark',
  },
  actions: {
    initTheme() {
      const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode | null
      this.setTheme(savedTheme === 'dark' ? 'dark' : 'light')
      this.isSidebarCollapsed = localStorage.getItem(SIDEBAR_STORAGE_KEY) === '1'
      // Right panel (chat history) always starts collapsed on app open;
      // it only expands when the user clicks the history toggle.
      this.isRightPanelCollapsed = true

      const savedLanguage = localStorage.getItem(LANGUAGE_STORAGE_KEY)
      this.language = savedLanguage?.trim() ? savedLanguage : 'auto'

      const savedSpeechLanguage = localStorage.getItem(SPEECH_LANGUAGE_STORAGE_KEY)
      this.speechLanguage = savedSpeechLanguage?.trim() ? savedSpeechLanguage : 'auto'

      this.preferredVoice = localStorage.getItem(VOICE_STORAGE_KEY) ?? ''
    },
    setTheme(theme: ThemeMode) {
      this.theme = theme
      localStorage.setItem(THEME_STORAGE_KEY, theme)
      document.documentElement.classList.toggle('dark', theme === 'dark')
    },
    setLanguage(language: AppLanguage) {
      this.language = language
      localStorage.setItem(LANGUAGE_STORAGE_KEY, language)
    },
    setSpeechLanguage(language: SpeechLanguage) {
      this.speechLanguage = language
      localStorage.setItem(SPEECH_LANGUAGE_STORAGE_KEY, language)
    },
    setPreferredVoice(voiceName: string) {
      this.preferredVoice = voiceName
      localStorage.setItem(VOICE_STORAGE_KEY, voiceName)
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
    openSettings(tab: SettingsTab = 'general') {
      this.settingsTab = tab
      this.isSettingsOpen = true
    },
    setSettingsTab(tab: SettingsTab) {
      this.settingsTab = tab
    },
    closeSettings() {
      this.isSettingsOpen = false
    },
  },
})
