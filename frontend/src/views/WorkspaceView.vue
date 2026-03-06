<script setup lang="ts">
import { computed, watchEffect } from 'vue'
import ChatView from '@/components/chat/ChatView.vue'
import DocumentsView from '@/components/documents/DocumentsView.vue'
import HistoryView from '@/components/chat/HistoryView.vue'
import LeftSidebar from '@/components/layout/LeftSidebar.vue'
import MainHeader from '@/components/layout/MainHeader.vue'
import RightPanel from '@/components/layout/RightPanel.vue'
import SettingsPanel from '@/components/layout/SettingsPanel.vue'
import StatisticsView from '@/components/statistics/StatisticsView.vue'
import UsersView from '@/components/users/UsersView.vue'
import { useAuthStore } from '@/stores/auth'
import { useUiStore } from '@/stores/ui'

const uiStore = useUiStore()
const authStore = useAuthStore()

const isManager = computed(() => authStore.currentUser?.role === 'manager')
const activeMainView = computed(() => {
  const view = uiStore.mainView
  if (isManager.value) {
    return view
  }
  if (view === 'documents' || view === 'users' || view === 'statistics') {
    return 'chat'
  }
  return view
})

watchEffect(() => {
  if (!isManager.value && activeMainView.value !== uiStore.mainView) {
    uiStore.switchMainView(activeMainView.value)
  }
})
</script>

<template>
  <div
    class="app-shell"
    :class="{
      'without-right-panel': activeMainView !== 'chat',
      'sidebar-collapsed': uiStore.isSidebarCollapsed,
      'right-panel-collapsed': activeMainView === 'chat' && uiStore.isRightPanelCollapsed,
    }"
  >
    <LeftSidebar />

    <main class="main">
      <MainHeader />
      <ChatView v-if="activeMainView === 'chat'" />
      <HistoryView v-else-if="activeMainView === 'history'" />
      <DocumentsView v-else-if="activeMainView === 'documents'" />
      <UsersView v-else-if="activeMainView === 'users'" />
      <StatisticsView v-else-if="activeMainView === 'statistics'" />
      <ChatView v-else />
    </main>

    <RightPanel v-if="activeMainView === 'chat'" />

    <SettingsPanel v-if="uiStore.isSettingsOpen" />
  </div>
</template>
