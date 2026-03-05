<script setup lang="ts">
import { computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useChatStore } from '@/stores/chat'
import { useHistoryStore, type HistoryItem } from '@/stores/history'
import { useUiStore } from '@/stores/ui'

const uiStore = useUiStore()
const chatStore = useChatStore()
const historyStore = useHistoryStore()
const { activeItemId } = storeToRefs(historyStore)

const recentChats = computed(() => {
  return historyStore.items.filter((item) => !historyStore.deletedIds.has(item.id)).slice(0, 7)
})

const chatCount = computed(() => recentChats.value.length)
const isCollapsed = computed(() => uiStore.isRightPanelCollapsed)

function startNewChat() {
  chatStore.resetConversation()
  historyStore.setActiveItem(null)
  uiStore.switchMainView('chat')
}

function getPrimaryQuestion(item: HistoryItem) {
  return item.title.trim() || item.preview.trim()
}

function openChat(item: HistoryItem) {
  historyStore.setActiveItem(item.id)
  uiStore.switchMainView('chat')
  void chatStore.sendMessage(getPrimaryQuestion(item))
}
</script>

<template>
  <aside class="right-panel" :class="{ collapsed: isCollapsed }">
    <div class="panel-header">
      <span v-if="!isCollapsed" class="panel-title">
        Chat History <span class="panel-count">{{ chatCount }}</span>
      </span>
      <button
        class="panel-menu"
        type="button"
        :aria-label="isCollapsed ? 'Expand chat history panel' : 'Collapse chat history panel'"
        @click="uiStore.toggleRightPanel()"
      >
        <span class="material-icons-outlined">history</span>
      </button>
    </div>

    <div class="project-list">
      <button class="new-project-btn" :class="{ compact: isCollapsed }" type="button" title="New Chat" @click="startNewChat">
        <div class="new-project-icon">
          <span class="material-icons-outlined">add</span>
        </div>
        <div v-if="!isCollapsed">
          <div class="new-project-label">New Chat</div>
          <div class="new-project-sub">Start a new conversation</div>
        </div>
      </button>

      <button
        v-for="item in recentChats"
        :key="item.id"
        class="project-card"
        :class="{ active: activeItemId === item.id, compact: isCollapsed }"
        type="button"
        :title="getPrimaryQuestion(item)"
        @click="openChat(item)"
      >
        <span v-if="isCollapsed" class="project-mini-icon material-icons-outlined">chat_bubble_outline</span>
        <span v-else class="project-name">{{ getPrimaryQuestion(item) }}</span>
      </button>
    </div>
  </aside>
</template>
