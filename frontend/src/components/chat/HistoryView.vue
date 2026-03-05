<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { useChatStore } from '@/stores/chat'
import { useHistoryStore, type HistoryItem } from '@/stores/history'
import { useUiStore } from '@/stores/ui'

const historyStore = useHistoryStore()
const uiStore = useUiStore()
const chatStore = useChatStore()
const { activeItemId, groupedItems, filteredItems, searchTerm } = storeToRefs(historyStore)

function openFromHistory(item: HistoryItem) {
  historyStore.setActiveItem(item.id)
  uiStore.switchMainView('chat')
  void chatStore.sendMessage(item.title)
}

function deleteItem(id: number) {
  historyStore.deleteItem(id)
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
      <button class="hist-new-btn" type="button" @click="uiStore.switchMainView('chat')">
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
              <button class="hist-action-btn" type="button">
                <span class="material-icons-outlined">push_pin</span>
              </button>
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
