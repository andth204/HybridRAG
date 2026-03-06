<script setup lang="ts">
import { computed } from 'vue'
import { useUiStore } from '@/stores/ui'

const uiStore = useUiStore()

const title = computed(() => {
  if (uiStore.mainView === 'history') {
    return 'Chat History'
  }
  if (uiStore.mainView === 'documents') {
    return 'Documents'
  }
  if (uiStore.mainView === 'users') {
    return 'User Management'
  }
  if (uiStore.mainView === 'statistics') {
    return 'Statistics'
  }
  return 'AI Chat'
})

function openHistory() {
  uiStore.switchMainView('history')
}

function openSettings() {
  uiStore.openSettings('general')
}

function openAccount() {
  uiStore.openSettings('account')
}
</script>

<template>
  <header class="main-header">
    <span class="main-header-title">{{ title }}</span>
    <div class="header-actions">
      <button class="btn-icon-r" type="button" aria-label="Open history" @click="openHistory">
        <span class="material-icons-outlined">history</span>
      </button>
      <button class="btn-icon-r" type="button" aria-label="Open settings" @click="openSettings">
        <span class="material-icons-outlined">settings</span>
      </button>
      <button class="btn-icon-r account-btn" type="button" aria-label="Open account settings" @click="openAccount">
        <span class="material-icons-outlined">account_circle</span>
        <span class="badge"></span>
      </button>
      <button class="btn-upgrade" type="button">
        <span class="material-icons-outlined">bolt</span>
        Upgrade
      </button>
    </div>
  </header>
</template>
