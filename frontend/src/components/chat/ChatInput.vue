<script setup lang="ts">
import { computed } from 'vue'
import { useChatStore } from '@/stores/chat'

const chatStore = useChatStore()

const draft = computed({
  get: () => chatStore.draft,
  set: (value: string) => {
    chatStore.setDraft(value)
  },
})

const charCount = computed(() => `${draft.value.length}/2,000`)

function handleSend() {
  void chatStore.sendMessage()
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    handleSend()
  }
}
</script>

<template>
  <div class="input-area">
    <div class="input-box">
      <div class="input-top">
        <textarea
          v-model="draft"
          rows="1"
          placeholder="Type your question…"
          maxlength="2000"
          @keydown="handleKeydown"
        ></textarea>

        <button class="send-btn" type="button" :disabled="chatStore.isStreaming" @click="handleSend">
          <span class="material-icons-outlined">send</span>
        </button>
      </div>

      <div class="input-bottom">
        <div class="input-tools"></div>
        <span class="char-count">{{ charCount }}</span>
      </div>
    </div>
  </div>
</template>
