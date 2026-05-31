<script setup lang="ts">
import { computed, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useChatStore } from '@/stores/chat'
import { useHistoryStore, type HistoryItem } from '@/stores/history'
import { useAuthStore } from '@/stores/auth'
import { useUiStore } from '@/stores/ui'

const uiStore = useUiStore()
const authStore = useAuthStore()
const chatStore = useChatStore()
const historyStore = useHistoryStore()
const { activeItemId, recentItems } = storeToRefs(historyStore)
const recentChats = computed(() => recentItems.value)

const chatCount = computed(() => recentChats.value.length)
const isCollapsed = computed(() => uiStore.isRightPanelCollapsed)
const deletingChatId = ref<string | null>(null)

function startNewChat() {
  chatStore.resetConversation()
  historyStore.setActiveItem(null)
  uiStore.switchMainView('chat')
}

function getPrimaryQuestion(item: HistoryItem) {
  return item.title.trim() || item.preview.trim()
}

async function openChat(item: HistoryItem) {
  const hasSession = await authStore.ensureSession()
  if (!hasSession || !authStore.accessToken.trim()) {
    return
  }

  historyStore.setActiveItem(item.id)
  uiStore.switchMainView('chat')
  await chatStore.openSession(authStore.accessToken.trim(), item.id)
}

async function deleteRecentChat(sessionId: string) {
  const targetSessionId = sessionId.trim()
  if (!targetSessionId || deletingChatId.value === targetSessionId) {
    return
  }

  const hasSession = await authStore.ensureSession()
  if (!hasSession || !authStore.accessToken.trim()) {
    return
  }

  deletingChatId.value = targetSessionId
  try {
    await historyStore.deleteItem(authStore.accessToken.trim(), targetSessionId)
    if (chatStore.activeSessionId === targetSessionId) {
      chatStore.resetConversation()
      historyStore.setActiveItem(null)
    }
  } finally {
    deletingChatId.value = null
  }
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

      <div
        v-for="item in recentChats"
        :key="item.id"
        class="project-card-row"
        :class="{ active: activeItemId === item.id, compact: isCollapsed }"
      >
        <button
          class="project-card"
          :class="{ active: activeItemId === item.id, compact: isCollapsed }"
          type="button"
          :title="getPrimaryQuestion(item)"
          @click="openChat(item)"
        >
          <span v-if="isCollapsed" class="project-mini-icon material-icons-outlined">chat_bubble_outline</span>
          <template v-else>
            <span class="project-card-icon"><span class="material-icons-outlined">chat_bubble_outline</span></span>
            <span class="project-name">{{ getPrimaryQuestion(item) }}</span>
          </template>
        </button>

        <div v-if="!isCollapsed" class="project-card-actions" @click.stop>
          <button
            class="project-card-action-btn del"
            type="button"
            aria-label="Delete chat history"
            title="Delete chat"
            :disabled="deletingChatId === item.id"
            @click="deleteRecentChat(item.id)"
          >
            <span class="material-icons-outlined">delete_outline</span>
          </button>
        </div>
      </div>
    </div>
  </aside>
</template>
