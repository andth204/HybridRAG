<script setup lang="ts">
import dayjs from 'dayjs'
import { useMessage } from 'naive-ui'
import { storeToRefs } from 'pinia'
import { computed, onMounted, ref, watch } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useUsersStore, type ManagedUser, type UserRole } from '@/stores/users'

const usersStore = useUsersStore()
const authStore = useAuthStore()
const message = useMessage()

const { errorMessage, filteredUsers, isLoading, managerUsers, newUsers7d, roleFilter, searchTerm, standardUsers, totalUsers } =
  storeToRefs(usersStore)

const isManager = computed(() => authStore.currentUser?.role === 'manager')
const activeUser = computed(() => {
  return filteredUsers.value.find((user) => user.id === usersStore.selectedUserId) ?? filteredUsers.value[0] ?? null
})

const pendingRole = ref<UserRole>('user')

watch(
  () => activeUser.value?.id ?? null,
  (nextId) => {
    if (nextId && usersStore.selectedUserId !== nextId) {
      usersStore.selectUser(nextId)
      return
    }
    if (!nextId && usersStore.selectedUserId !== null) {
      usersStore.selectUser(null)
    }
  },
  { immediate: true },
)

function formatDate(value: string) {
  return dayjs(value).format('DD/MM/YYYY HH:mm')
}

function initials(fullName: string): string {
  const parts = fullName.trim().split(/\s+/).filter(Boolean)
  if (parts.length === 0) {
    return 'U'
  }
  if (parts.length === 1) {
    const firstPart = parts[0] ?? ''
    return firstPart.slice(0, 2).toUpperCase()
  }
  const firstInitial = parts[0]?.[0] ?? ''
  const secondInitial = parts[1]?.[0] ?? ''
  return `${firstInitial}${secondInitial}`.toUpperCase()
}

function roleClass(role: UserRole) {
  return `users-role-${role}`
}

function roleLabel(role: UserRole) {
  if (role === 'manager') {
    return 'Manager'
  }
  return 'User'
}

function accessClass(isBlocked: boolean) {
  return isBlocked ? 'users-status-blocked' : 'users-status-active'
}

function accessLabel(isBlocked: boolean) {
  return isBlocked ? 'Blocked' : 'Allowed'
}

const currentUserId = computed(() => authStore.currentUser?.id ?? '')

async function loadUsers() {
  if (!isManager.value) {
    return
  }
  const ok = await usersStore.fetchUsers(authStore.accessToken)
  if (!ok && usersStore.errorMessage) {
    message.error(usersStore.errorMessage)
  }
}

async function onRoleChange(user: ManagedUser, event: Event) {
  pendingRole.value = (event.target as HTMLSelectElement).value as UserRole
}

async function confirmRoleUpdate(user: ManagedUser) {
  if (pendingRole.value === user.role) {
    message.info('Role is unchanged.')
    return
  }

  const result = await usersStore.updateRole(authStore.accessToken, user.id, pendingRole.value)
  if (!result.ok) {
    message.error(result.error || 'Update role failed.')
    return
  }
  message.success('Role updated successfully.')
}

async function toggleLoginAccess(user: ManagedUser) {
  if (user.id === currentUserId.value && !user.isBlocked) {
    message.warning('You cannot block your own account.')
    return
  }

  const nextBlocked = !user.isBlocked
  const confirmed = window.confirm(
    nextBlocked
      ? `Block "${user.fullName}" from signing in?`
      : `Allow "${user.fullName}" to sign in again?`,
  )
  if (!confirmed) {
    return
  }

  const result = await usersStore.updateLoginAccess(authStore.accessToken, user.id, nextBlocked)
  if (!result.ok) {
    message.error(result.error || 'Update login access failed.')
    return
  }
  message.success(nextBlocked ? 'User has been blocked.' : 'User can sign in again.')
}

async function removeUser(user: ManagedUser) {
  const approved = window.confirm(`Delete user "${user.fullName}"?`)
  if (!approved) {
    return
  }

  const result = await usersStore.removeUser(authStore.accessToken, user.id)
  if (!result.ok) {
    message.error(result.error || 'Delete user failed.')
    return
  }
  message.warning(`User "${user.fullName}" has been removed.`)
}

onMounted(() => {
  void loadUsers()
})

watch(
  () => activeUser.value?.role ?? 'user',
  (role) => {
    pendingRole.value = role as UserRole
  },
  { immediate: true },
)
</script>

<template>
  <div class="users-view">
    <div v-if="!isManager" class="users-permission-note">
      <span class="material-icons-outlined">shield_lock</span>
      <p>Only manager accounts can access user management.</p>
    </div>

    <template v-else>
      <div class="users-toolbar">
        <div class="users-search-wrap">
          <span class="material-icons-outlined">search</span>
          <input v-model="searchTerm" type="text" placeholder="Search by name or email..." />
        </div>

        <select v-model="roleFilter" class="users-filter-select">
          <option value="all">All roles</option>
          <option value="manager">Manager</option>
          <option value="user">User</option>
        </select>
      </div>

      <div class="users-stats-row">
        <span class="users-stat-pill">Total {{ totalUsers }}</span>
        <span class="users-stat-pill status-active">{{ managerUsers }} managers</span>
        <span class="users-stat-pill status-invited">{{ standardUsers }} users</span>
        <span class="users-stat-pill">{{ newUsers7d }} new (7d)</span>
      </div>

      <div v-if="errorMessage" class="users-error-banner">{{ errorMessage }}</div>

      <div class="users-layout">
        <section class="users-list-panel">
          <div class="users-list-head">Users</div>
          <div class="users-list-body">
            <div v-if="isLoading" class="users-empty">
              <span class="material-icons-outlined">hourglass_top</span>
              <p>Loading users...</p>
            </div>

            <div v-else-if="filteredUsers.length === 0" class="users-empty">
              <span class="material-icons-outlined">group_off</span>
              <p>No users found with current filter.</p>
            </div>

            <button
              v-for="user in filteredUsers"
              v-else
              :key="user.id"
              class="users-row"
              :class="{ active: activeUser?.id === user.id }"
              type="button"
              @click="usersStore.selectUser(user.id)"
            >
              <div class="users-row-left">
                <div class="users-avatar">{{ initials(user.fullName) }}</div>
                <div class="users-row-main">
                  <p class="users-name">{{ user.fullName }}</p>
                  <p class="users-email">{{ user.email }}</p>
                </div>
              </div>
              <div class="users-row-right">
                <span class="users-badge" :class="roleClass(user.role)">{{ roleLabel(user.role) }}</span>
                <span class="users-badge" :class="accessClass(user.isBlocked)">{{ accessLabel(user.isBlocked) }}</span>
              </div>
            </button>
          </div>
        </section>

        <aside class="users-detail-panel">
          <template v-if="activeUser">
            <div class="users-detail-top">
              <div class="users-avatar lg">{{ initials(activeUser.fullName) }}</div>
              <div class="users-detail-main">
                <h3>{{ activeUser.fullName }}</h3>
                <p>{{ activeUser.email }}</p>
                <div class="users-detail-tags">
                  <span class="users-badge" :class="roleClass(activeUser.role)">{{ roleLabel(activeUser.role) }}</span>
                  <span class="users-badge" :class="accessClass(activeUser.isBlocked)">
                    {{ accessLabel(activeUser.isBlocked) }}
                  </span>
                </div>
              </div>
            </div>

            <div class="users-detail-grid">
              <div class="users-detail-item">
                <span>Created</span>
                <strong>{{ formatDate(activeUser.createdAt) }}</strong>
              </div>
              <div class="users-detail-item">
                <span>Updated</span>
                <strong>{{ formatDate(activeUser.updatedAt) }}</strong>
              </div>
              <div class="users-detail-item">
                <span>Total Sessions</span>
                <strong>{{ activeUser.sessionCount }}</strong>
              </div>
              <div class="users-detail-item">
                <span>Total Messages</span>
                <strong>{{ activeUser.messageCount }}</strong>
              </div>
            </div>

            <div class="users-detail-field">
              <label>Role</label>
              <select :value="pendingRole" @change="onRoleChange(activeUser, $event)">
                <option value="manager">Manager</option>
                <option value="user">User</option>
              </select>
            </div>

            <div class="users-detail-actions">
              <button class="users-action-btn" type="button" @click="confirmRoleUpdate(activeUser)">
                <span class="material-icons-outlined">check_circle</span>
                Confirm Role
              </button>
              <button
                class="users-action-btn"
                :class="{ danger: !activeUser.isBlocked }"
                type="button"
                @click="toggleLoginAccess(activeUser)"
              >
                <span class="material-icons-outlined">{{ activeUser.isBlocked ? 'lock_open' : 'block' }}</span>
                {{ activeUser.isBlocked ? 'Allow Login' : 'Block Login' }}
              </button>
              <button class="users-action-btn danger" type="button" @click="removeUser(activeUser)">
                <span class="material-icons-outlined">delete_outline</span>
                Delete User
              </button>
            </div>
          </template>

          <div v-else class="users-empty detail">
            <span class="material-icons-outlined">person_search</span>
            <p>Select a user to view details.</p>
          </div>
        </aside>
      </div>
    </template>
  </div>
</template>
