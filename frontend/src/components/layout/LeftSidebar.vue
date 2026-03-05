<script setup lang="ts">
import { computed } from 'vue'
import { useUiStore } from '@/stores/ui'

type NavKey = 'chat' | 'documents' | 'users' | 'statistics' | 'history' | 'settings' | 'help'

interface NavItem {
  key: NavKey
  label: string
  icon: string
}

const uiStore = useUiStore()

const navItems: NavItem[] = [
  { key: 'chat', label: 'AI Chat', icon: 'chat_bubble_outline' },
  { key: 'documents', label: 'Documents', icon: 'description' },
  { key: 'users', label: 'Users', icon: 'groups' },
  { key: 'statistics', label: 'Statistics', icon: 'analytics' },
  { key: 'history', label: 'History', icon: 'history' },
  { key: 'settings', label: 'Settings', icon: 'settings' },
  { key: 'help', label: 'Help', icon: 'help_outline' },
]

const activeMainView = computed(() => uiStore.mainView)
const isCollapsed = computed(() => uiStore.isSidebarCollapsed)
const mainNavItems = computed(() => navItems.slice(0, 5))
const auxNavItems = computed(() => navItems.slice(5))

function isNavActive(key: NavKey) {
  return key === activeMainView.value
}

function handleNavClick(key: NavKey) {
  if (key === 'history') {
    uiStore.switchMainView('history')
    return
  }
  if (key === 'documents') {
    uiStore.switchMainView('documents')
    return
  }
  if (key === 'users') {
    uiStore.switchMainView('users')
    return
  }
  if (key === 'statistics') {
    uiStore.switchMainView('statistics')
    return
  }

  uiStore.switchMainView('chat')
}
</script>

<template>
  <aside class="sidebar" :class="{ collapsed: isCollapsed }">
    <div class="sidebar-logo">
      <div v-if="!isCollapsed" class="logo-icon">
        <span class="material-icons-outlined">blur_on</span>
      </div>
      <span v-if="!isCollapsed" class="logo-name">Script</span>
      <button class="btn-sq" type="button" @click="uiStore.toggleSidebar()">
        <span class="material-icons-outlined">{{ isCollapsed ? 'menu_open' : 'menu' }}</span>
      </button>
    </div>

    <div class="sidebar-search" :class="{ compact: isCollapsed }">
      <button v-if="isCollapsed" class="collapsed-search-btn" type="button" aria-label="Search">
        <span class="material-icons-outlined">search</span>
      </button>
      <div v-else class="search-wrap">
        <span class="material-icons-outlined">search</span>
        <input type="text" placeholder="Search" />
      </div>
    </div>

    <nav class="sidebar-nav">
      <button
        v-for="item in mainNavItems"
        :key="item.key"
        class="nav-item"
        :class="{
          active: isNavActive(item.key),
          compact: isCollapsed,
        }"
        type="button"
        :aria-label="item.label"
        @click="handleNavClick(item.key)"
      >
        <span class="material-icons-outlined">{{ item.icon }}</span>
        <span v-if="!isCollapsed">{{ item.label }}</span>
      </button>

      <div v-if="!isCollapsed" class="nav-section">Settings &amp; Help</div>

      <button
        v-for="item in auxNavItems"
        :key="item.key"
        class="nav-item"
        :class="{ compact: isCollapsed }"
        type="button"
        :aria-label="item.label"
        @click="handleNavClick(item.key)"
      >
        <span class="material-icons-outlined">{{ item.icon }}</span>
        <span v-if="!isCollapsed">{{ item.label }}</span>
      </button>
    </nav>

    <div class="sidebar-bottom">
      <div class="theme-toggle" :class="{ compact: isCollapsed }">
        <button
          class="theme-btn"
          :class="{ active: !uiStore.isDark, compact: isCollapsed }"
          type="button"
          @click="uiStore.setTheme('light')"
        >
          <span class="material-icons-outlined">light_mode</span>
          <span v-if="!isCollapsed">Light</span>
        </button>
        <button
          class="theme-btn"
          :class="{ active: uiStore.isDark, compact: isCollapsed }"
          type="button"
          @click="uiStore.setTheme('dark')"
        >
          <span class="material-icons-outlined">dark_mode</span>
          <span v-if="!isCollapsed">Dark</span>
        </button>
      </div>

      <div class="user-card">
        <div class="user-av-ph">E</div>
        <div v-if="!isCollapsed" class="user-info">
          <div class="user-name">Emilie Catlin</div>
          <div class="user-email">hey@emiliecatlin.com</div>
        </div>
      </div>
    </div>
  </aside>
</template>
