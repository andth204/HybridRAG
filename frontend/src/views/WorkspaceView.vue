<script setup lang="ts">
import ChatView from '@/components/chat/ChatView.vue'
import DocumentsView from '@/components/documents/DocumentsView.vue'
import HistoryView from '@/components/chat/HistoryView.vue'
import LeftSidebar from '@/components/layout/LeftSidebar.vue'
import MainHeader from '@/components/layout/MainHeader.vue'
import RightPanel from '@/components/layout/RightPanel.vue'
import StatisticsView from '@/components/statistics/StatisticsView.vue'
import UsersView from '@/components/users/UsersView.vue'
import { useUiStore } from '@/stores/ui'

const uiStore = useUiStore()
</script>

<template>
  <div
    class="app-shell"
    :class="{
      'without-right-panel': uiStore.mainView !== 'chat',
      'sidebar-collapsed': uiStore.isSidebarCollapsed,
      'right-panel-collapsed': uiStore.mainView === 'chat' && uiStore.isRightPanelCollapsed,
    }"
  >
    <LeftSidebar />

    <main class="main">
      <MainHeader />
      <ChatView v-if="uiStore.mainView === 'chat'" />
      <HistoryView v-else-if="uiStore.mainView === 'history'" />
      <DocumentsView v-else-if="uiStore.mainView === 'documents'" />
      <UsersView v-else-if="uiStore.mainView === 'users'" />
      <StatisticsView v-else-if="uiStore.mainView === 'statistics'" />
      <ChatView v-else />
    </main>

    <RightPanel v-if="uiStore.mainView === 'chat'" />
  </div>
</template>
