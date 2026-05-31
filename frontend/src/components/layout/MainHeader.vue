<script setup lang="ts">
import dayjs from 'dayjs'
import { NPopover } from 'naive-ui'
import { computed, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useAuthStore } from '@/stores/auth'
import { useNotificationsStore } from '@/stores/notifications'
import { useUiStore } from '@/stores/ui'

const uiStore = useUiStore()
const authStore = useAuthStore()
const notificationsStore = useNotificationsStore()
const { items, unreadCount } = storeToRefs(notificationsStore)
const canViewNotifications = computed(() => authStore.currentUser?.role === 'manager')

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

function openSettings() {
  uiStore.openSettings('general')
}

function openAccount() {
  uiStore.openSettings('account')
}

function formatNotificationTime(value: string) {
  return dayjs(value).format('DD/MM/YYYY HH:mm')
}

function markAllNotificationsRead() {
  notificationsStore.markAllAsRead()
}

function clearAllNotifications() {
  notificationsStore.clearAll()
}

function openNotification(notificationId: string) {
  notificationsStore.markAsRead(notificationId)
}

function deleteNotification(notificationId: string) {
  notificationsStore.remove(notificationId)
}

watch(
  () => authStore.currentUser?.id ?? null,
  (userId) => {
    notificationsStore.hydrateForUser(userId)
  },
  { immediate: true },
)
</script>

<template>
  <header class="main-header">
    <span class="main-header-title">{{ title }}</span>
    <div class="header-actions">
      <NPopover
        v-if="canViewNotifications"
        trigger="click"
        placement="bottom-end"
        raw
        :show-arrow="false"
        class="notifications-popover"
      >
        <template #trigger>
          <button class="btn-icon-r notification-btn" type="button" aria-label="Open notifications">
            <span class="material-icons-outlined">notifications_none</span>
            <span v-if="unreadCount" class="notification-count">{{ unreadCount }}</span>
          </button>
        </template>

        <div class="notifications-panel">
          <div class="notifications-head">
            <strong>Thông báo</strong>
            <div class="notifications-head-actions">
              <button class="notifications-head-btn" type="button" @click="markAllNotificationsRead">
                Đã đọc
              </button>
              <button class="notifications-head-btn" type="button" @click="clearAllNotifications">
                Xóa
              </button>
            </div>
          </div>

          <div v-if="!items.length" class="notifications-empty">
            <span class="material-icons-outlined">notifications_off</span>
            <p>Chưa có thông báo nào.</p>
          </div>

          <div v-else class="notifications-list">
            <article
              v-for="item in items"
              :key="item.id"
              class="notification-item"
              :class="{ unread: !item.isRead }"
              @click="openNotification(item.id)"
            >
              <p class="notification-message">{{ item.message }}</p>
              <div class="notification-meta">
                <span>{{ formatNotificationTime(item.createdAt) }}</span>
                <button class="notification-delete-btn" type="button" @click.stop="deleteNotification(item.id)">
                  Xóa
                </button>
              </div>
            </article>
          </div>
        </div>
      </NPopover>
      <button class="btn-icon-r account-btn" type="button" aria-label="Open account settings" @click="openAccount">
        <span class="material-icons-outlined">account_circle</span>
        <span class="badge"></span>
      </button>
      <button class="btn-icon-r" type="button" aria-label="Open settings" @click="openSettings">
        <span class="material-icons-outlined">settings</span>
      </button>
    </div>
  </header>
</template>
