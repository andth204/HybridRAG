<script setup lang="ts">
import dayjs from 'dayjs'
import { NPopselect, NSelect } from 'naive-ui'
import { storeToRefs } from 'pinia'
import { computed, onMounted, ref, watch } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useNotificationsStore } from '@/stores/notifications'
import { useUsersStore, type ManagedUser, type UserRole } from '@/stores/users'

const usersStore = useUsersStore()
const authStore = useAuthStore()
const notificationsStore = useNotificationsStore()

const { errorMessage, filteredUsers, isLoading, managerUsers, newUsers7d, roleFilter, searchTerm, standardUsers, totalUsers } =
  storeToRefs(usersStore)

const isManager = computed(() => authStore.currentUser?.role === 'manager')
const activeUser = computed(() => {
  return filteredUsers.value.find((user) => user.id === usersStore.selectedUserId) ?? filteredUsers.value[0] ?? null
})

const pendingRole = ref<UserRole>('user')
const roleOptions = [
  { label: 'Manager', value: 'manager' },
  { label: 'User', value: 'user' },
]
const loginAccessOptions = [
  { label: 'Allowed', value: 'allowed' },
  { label: 'Blocked', value: 'blocked' },
]
const deleteActionOptions = [
  { label: 'Delete user permanently', value: 'delete' },
  { label: 'Cancel', value: 'cancel' },
]

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
    notificationsStore.pushNotification({
      source: 'users',
      level: 'error',
      message: usersStore.errorMessage,
    })
  }
}

function accessValue(user: ManagedUser): 'allowed' | 'blocked' {
  return user.isBlocked ? 'blocked' : 'allowed'
}

async function changeRole(user: ManagedUser, nextRole: string) {
  const role = nextRole === 'manager' ? 'manager' : 'user'
  pendingRole.value = role
  if (role === user.role) {
    return
  }

  const result = await usersStore.updateRole(authStore.accessToken, user.id, role)
  if (!result.ok) {
    const errorText = result.error || 'Update role failed.'
    pendingRole.value = user.role
    notificationsStore.pushNotification({
      source: 'users',
      level: 'error',
      message: errorText,
    })
    return
  }
  notificationsStore.pushNotification({
    source: 'users',
    level: 'success',
    message: `Updated role for "${user.fullName}" to ${roleLabel(role)}.`,
  })
}

async function handleRoleSelection(nextRole: string) {
  if (!activeUser.value) {
    return
  }
  await changeRole(activeUser.value, nextRole)
}

async function changeLoginAccess(user: ManagedUser, nextAccess: string) {
  const nextBlocked = nextAccess === 'blocked'
  if (nextBlocked === user.isBlocked) {
    return
  }

  if (user.id === currentUserId.value && nextBlocked) {
    notificationsStore.pushNotification({
      source: 'users',
      level: 'warning',
      message: 'You cannot block your own account.',
    })
    return
  }

  const result = await usersStore.updateLoginAccess(authStore.accessToken, user.id, nextBlocked)
  if (!result.ok) {
    const errorText = result.error || 'Update login access failed.'
    notificationsStore.pushNotification({
      source: 'users',
      level: 'error',
      message: errorText,
    })
    return
  }
  notificationsStore.pushNotification({
    source: 'users',
    level: nextBlocked ? 'warning' : 'success',
    message: nextBlocked
      ? `Blocked "${user.fullName}" from signing in.`
      : `Allowed "${user.fullName}" to sign in again.`,
  })
}

async function handleLoginAccessSelection(nextAccess: string) {
  if (!activeUser.value) {
    return
  }
  await changeLoginAccess(activeUser.value, nextAccess)
}

async function handleDeleteSelection(user: ManagedUser, action: string) {
  if (action !== 'delete') {
    return
  }

  const result = await usersStore.removeUser(authStore.accessToken, user.id)
  if (!result.ok) {
    const errorText = result.error || 'Delete user failed.'
    notificationsStore.pushNotification({
      source: 'users',
      level: 'error',
      message: errorText,
    })
    return
  }
  notificationsStore.pushNotification({
    source: 'users',
    level: 'warning',
    message: `Deleted user "${user.fullName}".`,
  })
}

async function handleDeleteAction(action: string) {
  if (!activeUser.value) {
    return
  }
  await handleDeleteSelection(activeUser.value, action)
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

            <div class="users-detail-controls">
              <div class="users-detail-field">
                <label>Role</label>
                <NSelect
                  class="users-combo"
                  :value="pendingRole"
                  :options="roleOptions"
                  :consistent-menu-width="false"
                  @update:value="(value) => handleRoleSelection(String(value ?? 'user'))"
                />
              </div>

              <div class="users-detail-field">
                <label>Login Access</label>
                <NSelect
                  class="users-combo"
                  :value="accessValue(activeUser)"
                  :options="loginAccessOptions"
                  :consistent-menu-width="false"
                  :disabled="activeUser.id === currentUserId && !activeUser.isBlocked"
                  @update:value="(value) => handleLoginAccessSelection(String(value ?? 'allowed'))"
                />
              </div>
            </div>

            <div class="users-detail-actions">
              <NPopselect
                :options="deleteActionOptions"
                :consistent-menu-width="false"
                @update:value="(value) => handleDeleteAction(String(value ?? 'cancel'))"
              >
                <button class="users-action-btn danger" type="button">
                  <span class="material-icons-outlined">delete_outline</span>
                  Delete User
                </button>
              </NPopselect>
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

