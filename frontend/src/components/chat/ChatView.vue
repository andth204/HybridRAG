<script setup lang="ts">
import { computed, nextTick } from 'vue'
import { storeToRefs } from 'pinia'
import ChatInput from '@/components/chat/ChatInput.vue'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'

const chatStore = useChatStore()
const authStore = useAuthStore()
const { hasMessages, isStreaming, messages } = storeToRefs(chatStore)
const userInitial = computed(() => (authStore.currentUser?.fullName?.charAt(0).toUpperCase() || 'U'))

const quickActions = [
  { text: 'Thông tin cơ sở đào tạo', icon: 'school', colorClass: 'action-amber' },
  { text: 'Phương thức & điều kiện xét tuyển', icon: 'fact_check', colorClass: 'action-blue' },
  { text: 'Thời gian nộp hồ sơ', icon: 'event_note', colorClass: 'action-green' },
  { text: 'Điểm chuẩn ngành Công nghệ thông tin', icon: 'bar_chart', colorClass: 'action-purple' },
]

async function sendQuickMessage(text: string) {
  await chatStore.sendMessage(text)
  await nextTick()
}
</script>

<template>
  <div class="chat-view">
    <div class="chat-area">
      <div v-if="!hasMessages" class="welcome-wrap">
        <h1 class="welcome-title">Chào mừng bạn đến với Trợ lý tư vấn Tuyển sinh UTEHY</h1>
        <p class="welcome-sub">Hãy hỏi bất cứ điều gì về chương trình đào tạo, học phí và tuyển sinh.</p>
        <div class="action-grid">
          <button
            v-for="action in quickActions"
            :key="action.text"
            class="action-card"
            type="button"
            @click="sendQuickMessage(action.text)"
          >
            <div class="action-card-left">
              <div class="action-icon" :class="action.colorClass">
                <span class="material-icons-outlined">{{ action.icon }}</span>
              </div>
              <span class="action-label">{{ action.text }}</span>
            </div>
            <span class="action-close">
              <span class="material-icons-outlined">check</span>
            </span>
          </button>
        </div>
      </div>

      <div v-else class="messages">
        <div v-for="message in messages" :key="message.id" class="msg" :class="{ user: message.role === 'user' }">
          <div class="msg-av" :class="message.role === 'user' ? 'usr' : 'ai'">
            <span v-if="message.role === 'user'">{{ userInitial }}</span>
            <span v-else>S</span>
          </div>
          <div class="msg-bubble">
            {{ message.content || '...' }}
          </div>
        </div>
      </div>

      <div v-if="isStreaming" class="typing-row">
        <div class="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>

    <ChatInput />
  </div>
</template>
