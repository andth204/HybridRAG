<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import {
  useUiStore,
  type AppLanguage,
  type SearchMode,
  type SettingsTab,
  type SpeechLanguage,
  type ThemeMode,
} from '@/stores/ui'

type DropdownMenu = 'theme' | 'language' | 'search-mode' | 'speech-language' | 'voice'

const uiStore = useUiStore()
const authStore = useAuthStore()
const router = useRouter()

const settingsPanelRef = ref<HTMLElement | null>(null)
const activeTriggerRef = ref<HTMLElement | null>(null)
const availableVoices = ref<SpeechSynthesisVoice[]>([])
const isPreviewPlaying = ref(false)
const openMenu = ref<DropdownMenu | null>(null)
const dropdownMaxHeight = ref(260)

const activeTab = computed(() => uiStore.settingsTab)

const themeOptions: Array<{ value: ThemeMode; label: string }> = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
]

const languageOptions: Array<{ value: AppLanguage; label: string }> = [
  { value: 'auto', label: 'Auto detect' },
  { value: 'en', label: 'English' },
  { value: 'vi', label: 'Vietnamese' },
  { value: 'ar', label: 'Arabic' },
  { value: 'bs', label: 'Bosnian' },
  { value: 'bg', label: 'Bulgarian' },
  { value: 'ca', label: 'Catalan' },
  { value: 'zh', label: 'Chinese' },
  { value: 'cs', label: 'Czech' },
  { value: 'da', label: 'Danish' },
  { value: 'nl', label: 'Dutch' },
  { value: 'fi', label: 'Finnish' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'hi', label: 'Hindi' },
  { value: 'id', label: 'Indonesian' },
  { value: 'it', label: 'Italian' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },
  { value: 'pl', label: 'Polish' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ru', label: 'Russian' },
  { value: 'es', label: 'Spanish' },
  { value: 'th', label: 'Thai' },
  { value: 'tr', label: 'Turkish' },
]

const speechLanguageOptions: Array<{ value: SpeechLanguage; label: string }> = [
  { value: 'auto', label: 'Auto detect' },
  { value: 'en-US', label: 'English (US)' },
  { value: 'en-GB', label: 'English (UK)' },
  { value: 'vi-VN', label: 'Vietnamese' },
  { value: 'ar-SA', label: 'Arabic' },
  { value: 'de-DE', label: 'German' },
  { value: 'es-ES', label: 'Spanish' },
  { value: 'fr-FR', label: 'French' },
  { value: 'it-IT', label: 'Italian' },
  { value: 'ja-JP', label: 'Japanese' },
  { value: 'ko-KR', label: 'Korean' },
  { value: 'pt-BR', label: 'Portuguese (Brazil)' },
  { value: 'ru-RU', label: 'Russian' },
  { value: 'zh-CN', label: 'Chinese (Mainland)' },
]

const searchModeOptions: Array<{ value: SearchMode; label: string }> = [
  { value: 'keyword', label: 'Keyword' },
  { value: 'semantic', label: 'Semantic' },
  { value: 'hybrid', label: 'Hybrid' },
]

const activeThemeLabel = computed(() => {
  return themeOptions.find((item) => item.value === uiStore.theme)?.label ?? 'Light'
})

const activeLanguageLabel = computed(() => {
  return languageOptions.find((item) => item.value === uiStore.language)?.label ?? 'Auto detect'
})

const activeSpeechLanguageLabel = computed(() => {
  return speechLanguageOptions.find((item) => item.value === uiStore.speechLanguage)?.label ?? 'Auto detect'
})

const activeSearchModeLabel = computed(() => {
  return searchModeOptions.find((item) => item.value === uiStore.searchMode)?.label ?? 'Hybrid'
})

const speechSupported = computed(() => {
  return typeof window !== 'undefined' && 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window
})

const voicesByLanguage = computed(() => {
  if (uiStore.speechLanguage === 'auto') {
    return availableVoices.value
  }

  const languagePrefix = uiStore.speechLanguage.split('-')[0]?.toLowerCase() ?? ''
  const filtered = availableVoices.value.filter((voice) => {
    return voice.lang.toLowerCase().startsWith(languagePrefix)
  })
  return filtered.length ? filtered : availableVoices.value
})

const activeVoiceLabel = computed(() => {
  if (uiStore.preferredVoice) {
    return uiStore.preferredVoice
  }
  return voicesByLanguage.value[0]?.name ?? 'System'
})

const accountRows = computed(() => {
  const user = authStore.currentUser
  return [
    { label: 'Name', value: user?.fullName || 'Not set' },
    { label: 'Email', value: user?.email || 'Not set' },
  ]
})

function loadVoices() {
  if (!speechSupported.value) {
    return
  }

  const voices = window.speechSynthesis.getVoices()
  if (!voices.length) {
    return
  }
  availableVoices.value = voices
}

function closePanel() {
  uiStore.closeSettings()
  closeMenu()
}

function switchTab(tab: SettingsTab) {
  uiStore.setSettingsTab(tab)
  closeMenu()
}

function closeMenu() {
  openMenu.value = null
  activeTriggerRef.value = null
}

function updateDropdownBounds() {
  if (!openMenu.value || !settingsPanelRef.value || !activeTriggerRef.value) {
    return
  }

  const panelRect = settingsPanelRef.value.getBoundingClientRect()
  const triggerRect = activeTriggerRef.value.getBoundingClientRect()
  const menuGap = 7
  const bottomSpacing = 10
  const availableHeight = Math.floor(panelRect.bottom - triggerRect.bottom - menuGap - bottomSpacing)
  dropdownMaxHeight.value = Math.max(0, availableHeight)
}

async function toggleMenu(menu: DropdownMenu, event: MouseEvent) {
  if (openMenu.value === menu) {
    closeMenu()
    return
  }

  activeTriggerRef.value = event.currentTarget as HTMLElement
  openMenu.value = menu
  await nextTick()
  updateDropdownBounds()
}

function selectTheme(value: ThemeMode) {
  uiStore.setTheme(value)
  closeMenu()
}

function selectLanguage(value: AppLanguage) {
  uiStore.setLanguage(value)
  closeMenu()
}

function selectSearchMode(value: SearchMode) {
  uiStore.setSearchMode(value)
  closeMenu()
}

function selectSpeechLanguage(value: SpeechLanguage) {
  uiStore.setSpeechLanguage(value)
  closeMenu()
}

function selectVoice(name: string) {
  uiStore.setPreferredVoice(name)
  closeMenu()
}

function playVoicePreview() {
  if (!speechSupported.value) {
    return
  }

  window.speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance('Hello, this is your HybridRAG voice preview.')
  utterance.lang = uiStore.speechLanguage === 'auto' ? navigator.language : uiStore.speechLanguage

  const selectedVoice =
    availableVoices.value.find((voice) => voice.name === uiStore.preferredVoice) ?? voicesByLanguage.value[0]

  if (selectedVoice) {
    utterance.voice = selectedVoice
  }

  isPreviewPlaying.value = true
  utterance.onend = () => {
    isPreviewPlaying.value = false
  }
  utterance.onerror = () => {
    isPreviewPlaying.value = false
  }
  window.speechSynthesis.speak(utterance)
}

async function logoutFromSystem() {
  await authStore.logout()
  uiStore.closeSettings()
  closeMenu()
  await router.push('/login')
}

function handleEscape(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    closePanel()
  }
}

function handleDocumentClick(event: MouseEvent) {
  const target = event.target as HTMLElement | null
  if (!target?.closest('.settings-dropdown-wrap')) {
    closeMenu()
  }
}

function handleViewportChange() {
  if (openMenu.value) {
    updateDropdownBounds()
  }
}

onMounted(() => {
  loadVoices()
  if (speechSupported.value) {
    window.speechSynthesis.addEventListener('voiceschanged', loadVoices)
  }
  window.addEventListener('keydown', handleEscape)
  window.addEventListener('mousedown', handleDocumentClick)
  window.addEventListener('resize', handleViewportChange)
  window.addEventListener('scroll', handleViewportChange, true)
})

onBeforeUnmount(() => {
  if (speechSupported.value) {
    window.speechSynthesis.removeEventListener('voiceschanged', loadVoices)
    window.speechSynthesis.cancel()
  }
  window.removeEventListener('keydown', handleEscape)
  window.removeEventListener('mousedown', handleDocumentClick)
  window.removeEventListener('resize', handleViewportChange)
  window.removeEventListener('scroll', handleViewportChange, true)
})
</script>

<template>
  <div class="settings-overlay" @click.self="closePanel">
    <section ref="settingsPanelRef" class="settings-panel" role="dialog" aria-modal="true" aria-label="System settings">
      <aside class="settings-sidebar">
        <div class="settings-sidebar-head">
          <span class="material-icons-outlined">tune</span>
          <span>System Settings</span>
        </div>

        <button
          class="settings-nav-item"
          :class="{ active: activeTab === 'general' }"
          type="button"
          @click="switchTab('general')"
        >
          <span class="material-icons-outlined">dashboard_customize</span>
          General
        </button>

        <button
          class="settings-nav-item"
          :class="{ active: activeTab === 'account' }"
          type="button"
          @click="switchTab('account')"
        >
          <span class="material-icons-outlined">account_circle</span>
          Account
        </button>

        <button
          class="settings-nav-item danger"
          :class="{ active: activeTab === 'logout' }"
          type="button"
          @click="switchTab('logout')"
        >
          <span class="material-icons-outlined">logout</span>
          Logout
        </button>
      </aside>

      <div class="settings-content">
        <header class="settings-content-head">
          <h3>{{ activeTab === 'general' ? 'General' : activeTab === 'account' ? 'Account' : 'Logout' }}</h3>
          <button class="settings-close-btn" type="button" aria-label="Close settings" @click="closePanel">
            <span class="material-icons-outlined">close</span>
          </button>
        </header>

        <div v-if="activeTab === 'general'" class="settings-pane">
          <div class="settings-list-card">
            <div class="settings-list-row">
              <span class="settings-row-label">Theme</span>
              <div class="settings-row-right">
                <span class="settings-row-value">{{ activeThemeLabel }}</span>
                <div class="settings-dropdown-wrap">
                  <button
                    class="settings-row-trigger"
                    type="button"
                    aria-label="Open theme options"
                    @click.stop="toggleMenu('theme', $event)"
                  >
                    <span class="material-icons-outlined">expand_more</span>
                  </button>
                  <div
                    v-if="openMenu === 'theme'"
                    class="settings-popup-menu"
                    role="listbox"
                    aria-label="Theme options"
                    :style="{ maxHeight: `${dropdownMaxHeight}px` }"
                  >
                    <button
                      v-for="option in themeOptions"
                      :key="option.value"
                      class="settings-popup-item"
                      :class="{ selected: uiStore.theme === option.value }"
                      type="button"
                      @click="selectTheme(option.value)"
                    >
                      <span>{{ option.label }}</span>
                      <span v-if="uiStore.theme === option.value" class="material-icons-outlined">check</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div class="settings-list-row">
              <span class="settings-row-label">Language</span>
              <div class="settings-row-right">
                <span class="settings-row-value">{{ activeLanguageLabel }}</span>
                <div class="settings-dropdown-wrap">
                  <button
                    class="settings-row-trigger"
                    type="button"
                    aria-label="Open language options"
                    @click.stop="toggleMenu('language', $event)"
                  >
                    <span class="material-icons-outlined">expand_more</span>
                  </button>
                  <div
                    v-if="openMenu === 'language'"
                    class="settings-popup-menu"
                    role="listbox"
                    aria-label="Language options"
                    :style="{ maxHeight: `${dropdownMaxHeight}px` }"
                  >
                    <button
                      v-for="option in languageOptions"
                      :key="option.value"
                      class="settings-popup-item"
                      :class="{ selected: uiStore.language === option.value }"
                      type="button"
                      @click="selectLanguage(option.value)"
                    >
                      <span>{{ option.label }}</span>
                      <span v-if="uiStore.language === option.value" class="material-icons-outlined">check</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div class="settings-list-row">
              <span class="settings-row-label">Speech language</span>
              <div class="settings-row-right">
                <span class="settings-row-value">{{ activeSpeechLanguageLabel }}</span>
                <div class="settings-dropdown-wrap">
                  <button
                    class="settings-row-trigger"
                    type="button"
                    aria-label="Open speech language options"
                    @click.stop="toggleMenu('speech-language', $event)"
                  >
                    <span class="material-icons-outlined">expand_more</span>
                  </button>
                  <div
                    v-if="openMenu === 'speech-language'"
                    class="settings-popup-menu"
                    role="listbox"
                    aria-label="Speech language options"
                    :style="{ maxHeight: `${dropdownMaxHeight}px` }"
                  >
                    <button
                      v-for="option in speechLanguageOptions"
                      :key="option.value"
                      class="settings-popup-item"
                      :class="{ selected: uiStore.speechLanguage === option.value }"
                      type="button"
                      @click="selectSpeechLanguage(option.value)"
                    >
                      <span>{{ option.label }}</span>
                      <span v-if="uiStore.speechLanguage === option.value" class="material-icons-outlined">check</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div class="settings-list-row">
              <span class="settings-row-label">Search type</span>
              <div class="settings-row-right">
                <span class="settings-row-value">{{ activeSearchModeLabel }}</span>
                <div class="settings-dropdown-wrap">
                  <button
                    class="settings-row-trigger"
                    type="button"
                    aria-label="Open search type options"
                    @click.stop="toggleMenu('search-mode', $event)"
                  >
                    <span class="material-icons-outlined">expand_more</span>
                  </button>
                  <div
                    v-if="openMenu === 'search-mode'"
                    class="settings-popup-menu"
                    role="listbox"
                    aria-label="Search type options"
                    :style="{ maxHeight: `${dropdownMaxHeight}px` }"
                  >
                    <button
                      v-for="option in searchModeOptions"
                      :key="option.value"
                      class="settings-popup-item"
                      :class="{ selected: uiStore.searchMode === option.value }"
                      type="button"
                      @click="selectSearchMode(option.value)"
                    >
                      <span>{{ option.label }}</span>
                      <span v-if="uiStore.searchMode === option.value" class="material-icons-outlined">check</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div class="settings-list-row">
              <span class="settings-row-label">Voice</span>
              <div class="settings-row-right settings-voice-right">
                <button
                  class="settings-play-btn"
                  type="button"
                  :disabled="!speechSupported || isPreviewPlaying"
                  @click="playVoicePreview"
                >
                  <span class="material-icons-outlined">play_arrow</span>
                  <span>{{ isPreviewPlaying ? 'Playing...' : 'Play' }}</span>
                </button>
                <span class="settings-row-divider"></span>
                <span class="settings-row-value">{{ activeVoiceLabel }}</span>
                <div class="settings-dropdown-wrap">
                  <button
                    class="settings-row-trigger"
                    type="button"
                    aria-label="Open voice options"
                    @click.stop="toggleMenu('voice', $event)"
                  >
                    <span class="material-icons-outlined">expand_more</span>
                  </button>
                  <div
                    v-if="openMenu === 'voice'"
                    class="settings-popup-menu"
                    role="listbox"
                    aria-label="Voice options"
                    :style="{ maxHeight: `${dropdownMaxHeight}px` }"
                  >
                    <button
                      class="settings-popup-item"
                      :class="{ selected: !uiStore.preferredVoice }"
                      type="button"
                      @click="selectVoice('')"
                    >
                      <span>System</span>
                      <span v-if="!uiStore.preferredVoice" class="material-icons-outlined">check</span>
                    </button>
                    <button
                      v-for="voice in voicesByLanguage"
                      :key="voice.voiceURI"
                      class="settings-popup-item"
                      :class="{ selected: uiStore.preferredVoice === voice.name }"
                      type="button"
                      @click="selectVoice(voice.name)"
                    >
                      <span>{{ voice.name }}</span>
                      <span v-if="uiStore.preferredVoice === voice.name" class="material-icons-outlined">check</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <p v-if="!speechSupported" class="settings-note">Speech preview is not supported in this browser.</p>
        </div>

        <div v-else-if="activeTab === 'account'" class="settings-pane">
          <div class="settings-list-card">
            <div v-for="row in accountRows" :key="row.label" class="settings-list-row">
              <span class="settings-row-label">{{ row.label }}</span>
              <span class="settings-row-value">{{ row.value }}</span>
            </div>
          </div>
        </div>

        <div v-else class="settings-pane settings-logout-pane">
          <p>You are about to sign out from this workspace.</p>
          <button class="settings-danger-btn" type="button" @click="logoutFromSystem">
            <span class="material-icons-outlined">logout</span>
            Logout now
          </button>
        </div>
      </div>
    </section>
  </div>
</template>
