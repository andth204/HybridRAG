<script setup lang="ts">
import { computed } from 'vue'
import { useUiStore } from '@/stores/ui'
import { useAuthStore } from '@/stores/auth'

type NavKey = 'chat' | 'documents' | 'users' | 'statistics' | 'history' | 'settings'

interface NavItem {
  key: NavKey
  label: string
  icon: string
}

const uiStore = useUiStore()
const authStore = useAuthStore()

const primaryNavItems: NavItem[] = [
  { key: 'chat', label: 'AI Chat', icon: 'chat_bubble_outline' },
  { key: 'documents', label: 'Documents', icon: 'description' },
  { key: 'users', label: 'Users', icon: 'groups' },
  { key: 'statistics', label: 'Statistics', icon: 'analytics' },
  { key: 'history', label: 'History', icon: 'history' },
]

const secondaryNavItems: NavItem[] = [
  { key: 'settings', label: 'Settings', icon: 'settings' },
]

const activeMainView = computed(() => uiStore.mainView)
const isCollapsed = computed(() => uiStore.isSidebarCollapsed)
const isManager = computed(() => authStore.currentUser?.role === 'manager')

function canAccessView(key: NavKey): boolean {
  if (key === 'documents' || key === 'users' || key === 'statistics') {
    return isManager.value
  }
  return true
}

const mainNavItems = computed(() => primaryNavItems.filter((item) => canAccessView(item.key)))
const auxNavItems = computed(() => secondaryNavItems)

function isNavActive(key: NavKey) {
  if (!canAccessView(key)) {
    return false
  }
  if (key === 'settings') {
    return uiStore.isSettingsOpen
  }
  return key === activeMainView.value
}

function handleNavClick(key: NavKey) {
  if (!canAccessView(key)) {
    uiStore.switchMainView('chat')
    return
  }

  if (key === 'settings') {
    uiStore.openSettings('general')
    return
  }

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

const displayName = computed(() => authStore.currentUser?.fullName ?? 'Guest User')
const displayEmail = computed(() => authStore.currentUser?.email ?? 'guest@hybridrag.local')
const displayInitial = computed(() => displayName.value.charAt(0).toUpperCase() || 'G')
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

      <button
        v-for="item in auxNavItems"
        :key="item.key"
        class="nav-item"
        :class="{ active: isNavActive(item.key), compact: isCollapsed }"
        type="button"
        :aria-label="item.label"
        @click="handleNavClick(item.key)"
      >
        <span class="material-icons-outlined">{{ item.icon }}</span>
        <span v-if="!isCollapsed">{{ item.label }}</span>
      </button>
    </nav>

    <div class="sidebar-bottom">
      <div class="user-card">
        <div class="user-av-ph">{{ displayInitial }}</div>
        <div v-if="!isCollapsed" class="user-info">
          <div class="user-name">{{ displayName }}</div>
          <div class="user-email">{{ displayEmail }}</div>
        </div>
      </div>
    </div>
  </aside>
</template>
