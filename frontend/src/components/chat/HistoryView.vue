<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { useChatStore } from '@/stores/chat'
import { useHistoryStore, type HistoryItem } from '@/stores/history'
import { useAuthStore } from '@/stores/auth'
import { useUiStore } from '@/stores/ui'

const authStore = useAuthStore()
const historyStore = useHistoryStore()
const uiStore = useUiStore()
const chatStore = useChatStore()
const { activeItemId, groupedItems, filteredItems, searchTerm } = storeToRefs(historyStore)

async function openFromHistory(item: HistoryItem) {
  const hasSession = await authStore.ensureSession()
  if (!hasSession || !authStore.accessToken.trim()) {
    return
  }
  historyStore.setActiveItem(item.id)
  await chatStore.openSession(authStore.accessToken.trim(), item.id)
  uiStore.switchMainView('chat')
}

async function deleteItem(id: string) {
  const hasSession = await authStore.ensureSession()
  if (!hasSession || !authStore.accessToken.trim()) {
    return
  }
  await historyStore.deleteItem(authStore.accessToken.trim(), id)
}

function startNewChat() {
  chatStore.resetConversation()
  historyStore.setActiveItem(null)
  uiStore.switchMainView('chat')
}
</script>

<template>
  <div class="history-view visible">
    <div class="hist-toolbar">
      <div class="hist-search-wrap">
        <span class="material-icons-outlined">search</span>
        <input
          v-model="searchTerm"
          type="text"
          placeholder="Search conversations..."
          @input="historyStore.setSearchTerm(searchTerm)"
        />
      </div>
      <button class="hist-new-btn" type="button" @click="startNewChat">
        <span class="material-icons-outlined">add</span>
        New Chat
      </button>
    </div>

    <div class="hist-body">
      <div v-if="filteredItems.length === 0" class="hist-empty">
        <span class="material-icons-outlined">search_off</span>
        <div class="hist-empty-title">No results found</div>
        <div class="hist-empty-sub">Try a different keyword or clear the search.</div>
      </div>

      <template v-else>
        <template v-for="(items, groupName) in groupedItems" :key="groupName">
          <div class="hist-group-label">{{ groupName }}</div>
          <div
            v-for="item in items"
            :key="item.id"
            class="hist-item"
            :class="{ 'active-chat': activeItemId === item.id }"
            @click="openFromHistory(item)"
          >
            <div class="hist-icon-wrap">
              <span class="material-icons-outlined">chat_bubble_outline</span>
            </div>
            <div class="hist-info">
              <div class="hist-title">{{ item.title }}</div>
              <div class="hist-preview">{{ item.preview }}</div>
            </div>
            <div class="hist-meta">
              <span class="hist-time">{{ item.time }}</span>
            </div>
            <div class="hist-actions" @click.stop>
              <button class="hist-action-btn del" type="button" @click="deleteItem(item.id)">
                <span class="material-icons-outlined">delete_outline</span>
              </button>
            </div>
          </div>
        </template>
      </template>
    </div>
  </div>
</template>
