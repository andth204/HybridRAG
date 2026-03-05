<script setup lang="ts">
import dayjs from 'dayjs'
import { useMessage } from 'naive-ui'
import { storeToRefs } from 'pinia'
import { computed, ref, watch } from 'vue'
import { useUsersStore, type ManagedUser, type UserRole, type UserStatus } from '@/stores/users'

const usersStore = useUsersStore()
const message = useMessage()

const { activeUsers, filteredUsers, newUsers7d, roleFilter, searchTerm, statusFilter, suspendedUsers, totalUsers } =
  storeToRefs(usersStore)

const activeUser = computed(() => {
  return filteredUsers.value.find((user) => user.id === usersStore.selectedUserId) ?? filteredUsers.value[0] ?? null
})

const isCreateModalOpen = ref(false)
const draftName = ref('')
const draftEmail = ref('')
const draftRole = ref<UserRole>('user')

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

function statusClass(status: UserStatus) {
  return `users-status-${status}`
}

function roleClass(role: UserRole) {
  return `users-role-${role}`
}

function statusLabel(status: UserStatus) {
  if (status === 'active') {
    return 'Active'
  }
  if (status === 'invited') {
    return 'Invited'
  }
  return 'Suspended'
}

function roleLabel(role: UserRole) {
  if (role === 'admin') {
    return 'Admin'
  }
  if (role === 'staff') {
    return 'Staff'
  }
  return 'User'
}

function openCreateModal() {
  draftName.value = ''
  draftEmail.value = ''
  draftRole.value = 'user'
  isCreateModalOpen.value = true
}

function closeCreateModal() {
  isCreateModalOpen.value = false
}

function createUser() {
  const fullName = draftName.value.trim()
  const email = draftEmail.value.trim()
  if (!fullName || !email) {
    message.warning('Please enter full name and email.')
    return
  }
  if (!/.+@.+\..+/.test(email)) {
    message.error('Email format is invalid.')
    return
  }

  const created = usersStore.createUser({
    fullName,
    email,
    role: draftRole.value,
  })
  message.success(`User "${created.fullName}" has been created.`)
  closeCreateModal()
}

function onRoleChange(user: ManagedUser, event: Event) {
  const nextRole = (event.target as HTMLSelectElement).value as UserRole
  if (usersStore.updateRole(user.id, nextRole)) {
    message.success(`Role updated for "${user.fullName}".`)
  }
}

function toggleSuspension(user: ManagedUser) {
  const isSuspended = user.status === 'suspended'
  if (usersStore.toggleUserSuspension(user.id)) {
    message.success(isSuspended ? `"${user.fullName}" is active again.` : `"${user.fullName}" has been suspended.`)
  }
}

function removeUser(user: ManagedUser) {
  const approved = window.confirm(`Delete user "${user.fullName}"?`)
  if (!approved) {
    return
  }
  if (usersStore.removeUser(user.id)) {
    message.warning(`User "${user.fullName}" has been removed.`)
  }
}
</script>

<template>
  <div class="users-view">
    <div class="users-toolbar">
      <div class="users-search-wrap">
        <span class="material-icons-outlined">search</span>
        <input v-model="searchTerm" type="text" placeholder="Search by name or email..." />
      </div>

      <select v-model="roleFilter" class="users-filter-select">
        <option value="all">All roles</option>
        <option value="admin">Admin</option>
        <option value="staff">Staff</option>
        <option value="user">User</option>
      </select>

      <select v-model="statusFilter" class="users-filter-select">
        <option value="all">All statuses</option>
        <option value="active">Active</option>
        <option value="invited">Invited</option>
        <option value="suspended">Suspended</option>
      </select>

      <button class="users-primary-btn" type="button" @click="openCreateModal">
        <span class="material-icons-outlined">person_add</span>
        Add User
      </button>
    </div>

    <div class="users-stats-row">
      <span class="users-stat-pill">Total {{ totalUsers }}</span>
      <span class="users-stat-pill status-active">{{ activeUsers }} active</span>
      <span class="users-stat-pill status-invited">{{ newUsers7d }} new (7d)</span>
      <span class="users-stat-pill status-suspended">{{ suspendedUsers }} suspended</span>
    </div>

    <div class="users-layout">
      <section class="users-list-panel">
        <div class="users-list-head">Users</div>
        <div class="users-list-body">
          <div v-if="filteredUsers.length === 0" class="users-empty">
            <span class="material-icons-outlined">group_off</span>
            <p>No users found with current filter.</p>
          </div>

          <button
            v-for="user in filteredUsers"
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
              <span class="users-badge" :class="statusClass(user.status)">{{ statusLabel(user.status) }}</span>
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
                <span class="users-badge" :class="statusClass(activeUser.status)">
                  {{ statusLabel(activeUser.status) }}
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
              <span>Last active</span>
              <strong>{{ formatDate(activeUser.lastActiveAt) }}</strong>
            </div>
            <div class="users-detail-item">
              <span>Sessions</span>
              <strong>{{ activeUser.totalSessions }}</strong>
            </div>
            <div class="users-detail-item">
              <span>Messages</span>
              <strong>{{ activeUser.totalMessages }}</strong>
            </div>
            <div class="users-detail-item">
              <span>Storage used</span>
              <strong>{{ activeUser.storageMb }} MB</strong>
            </div>
          </div>

          <div class="users-detail-field">
            <label>Role</label>
            <select :value="activeUser.role" @change="onRoleChange(activeUser, $event)">
              <option value="admin">Admin</option>
              <option value="staff">Staff</option>
              <option value="user">User</option>
            </select>
          </div>

          <div class="users-detail-actions">
            <button class="users-action-btn" type="button" @click="toggleSuspension(activeUser)">
              <span class="material-icons-outlined">
                {{ activeUser.status === 'suspended' ? 'lock_open' : 'lock' }}
              </span>
              {{ activeUser.status === 'suspended' ? 'Activate User' : 'Suspend User' }}
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

    <div v-if="isCreateModalOpen" class="users-modal-backdrop" @click.self="closeCreateModal">
      <div class="users-modal">
        <h3>Create New User</h3>
        <label>
          Full Name
          <input v-model="draftName" type="text" placeholder="Enter full name" />
        </label>
        <label>
          Email
          <input v-model="draftEmail" type="email" placeholder="name@company.com" />
        </label>
        <label>
          Role
          <select v-model="draftRole">
            <option value="admin">Admin</option>
            <option value="staff">Staff</option>
            <option value="user">User</option>
          </select>
        </label>
        <div class="users-modal-actions">
          <button class="users-action-btn" type="button" @click="closeCreateModal">Cancel</button>
          <button class="users-primary-btn" type="button" @click="createUser">Create</button>
        </div>
      </div>
    </div>
  </div>
</template>
