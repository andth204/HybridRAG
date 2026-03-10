import axios from 'axios'
import { defineStore } from 'pinia'
import {
  deleteManagedUser,
  fetchUsersList,
  patchManagedUserLoginAccess,
  patchManagedUserRole,
  type BackendManagedUser,
} from '@/services/usersApi'

export type UserRole = 'manager' | 'user'
export type RoleFilter = UserRole | 'all'

export interface ManagedUser {
  id: string
  fullName: string
  email: string
  role: UserRole
  isBlocked: boolean
  sessionCount: number
  messageCount: number
  createdAt: string
  updatedAt: string
}

interface UserActionResult {
  ok: boolean
  error?: string
}

interface UpdateRoleResult extends UserActionResult {
  user?: ManagedUser
}

interface UpdateLoginAccessResult extends UserActionResult {
  user?: ManagedUser
}

function normalizeDisplayName(raw: string): string {
  const base = raw.trim()
  if (!base) {
    return 'HybridRAG User'
  }
  return base
    .split(/[._-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function toManagedUser(user: BackendManagedUser): ManagedUser {
  const fallbackName = user.email.split('@')[0] || 'HybridRAG User'
  return {
    id: user.id,
    fullName: normalizeDisplayName(user.username || fallbackName),
    email: user.email.trim().toLowerCase(),
    role: user.role,
    isBlocked: Boolean(user.is_blocked),
    sessionCount: Number(user.session_count || 0),
    messageCount: Number(user.message_count || 0),
    createdAt: user.created_at,
    updatedAt: user.updated_at,
  }
}

function extractApiErrorMessage(error: unknown, fallback: string): string {
  if (!axios.isAxiosError(error)) {
    return fallback
  }

  const detail = error.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) {
    return detail
  }

  const message = error.message?.trim()
  return message || fallback
}

function compareManagedUsers(a: ManagedUser, b: ManagedUser): number {
  if (a.role !== b.role) {
    return a.role === 'manager' ? -1 : 1
  }

  const updatedDiff = new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  if (updatedDiff !== 0) {
    return updatedDiff
  }

  return a.fullName.localeCompare(b.fullName)
}

function getSortedUsers(users: ManagedUser[]): ManagedUser[] {
  return [...users].sort(compareManagedUsers)
}

export const useUsersStore = defineStore('users', {
  state: () => ({
    users: [] as ManagedUser[],
    searchTerm: '',
    roleFilter: 'all' as RoleFilter,
    selectedUserId: null as string | null,
    isLoading: false,
    errorMessage: '',
  }),
  getters: {
    filteredUsers(state): ManagedUser[] {
      const term = state.searchTerm.trim().toLowerCase()
      const filtered = state.users.filter((user) => {
        if (state.roleFilter !== 'all' && user.role !== state.roleFilter) {
          return false
        }
        if (!term) {
          return true
        }
        return user.fullName.toLowerCase().includes(term) || user.email.toLowerCase().includes(term)
      })

      return getSortedUsers(filtered)
    },
    selectedUser(state): ManagedUser | null {
      if (!state.selectedUserId) {
        return null
      }
      return state.users.find((user) => user.id === state.selectedUserId) ?? null
    },
    totalUsers(state): number {
      return state.users.length
    },
    managerUsers(state): number {
      return state.users.filter((user) => user.role === 'manager').length
    },
    standardUsers(state): number {
      return state.users.filter((user) => user.role === 'user').length
    },
    newUsers7d(state): number {
      const sevenDaysAgo = new Date()
      sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7)
      return state.users.filter((user) => new Date(user.createdAt) >= sevenDaysAgo).length
    },
  },
  actions: {
    setSearchTerm(value: string) {
      this.searchTerm = value
    },
    setRoleFilter(value: RoleFilter) {
      this.roleFilter = value
    },
    selectUser(userId: string | null) {
      this.selectedUserId = userId
    },
    setUsersFromBackend(items: BackendManagedUser[]) {
      const mapped = getSortedUsers(items.map(toManagedUser))
      const previousSelectedId = this.selectedUserId
      this.users = mapped
      if (previousSelectedId && mapped.some((item) => item.id === previousSelectedId)) {
        this.selectedUserId = previousSelectedId
        return
      }
      this.selectedUserId = mapped[0]?.id ?? null
    },
    async fetchUsers(accessToken: string): Promise<boolean> {
      this.isLoading = true
      this.errorMessage = ''
      try {
        const items = await fetchUsersList(accessToken)
        this.setUsersFromBackend(items)
        return true
      } catch (error) {
        this.errorMessage = extractApiErrorMessage(error, 'Failed to load users.')
        return false
      } finally {
        this.isLoading = false
      }
    },
    async updateRole(accessToken: string, userId: string, role: UserRole): Promise<UpdateRoleResult> {
      try {
        const updated = await patchManagedUserRole(accessToken, userId, role)
        const mapped = toManagedUser(updated)
        const index = this.users.findIndex((user) => user.id === userId)
        if (index >= 0) {
          this.users.splice(index, 1, mapped)
        } else {
          this.users.unshift(mapped)
        }
        this.users = getSortedUsers(this.users)
        return { ok: true, user: mapped }
      } catch (error) {
        return {
          ok: false,
          error: extractApiErrorMessage(error, 'Update role failed.'),
        }
      }
    },
    async updateLoginAccess(accessToken: string, userId: string, isBlocked: boolean): Promise<UpdateLoginAccessResult> {
      try {
        const updated = await patchManagedUserLoginAccess(accessToken, userId, isBlocked)
        const mapped = toManagedUser(updated)
        const index = this.users.findIndex((user) => user.id === userId)
        if (index >= 0) {
          this.users.splice(index, 1, mapped)
        } else {
          this.users.unshift(mapped)
        }
        this.users = getSortedUsers(this.users)
        return { ok: true, user: mapped }
      } catch (error) {
        return {
          ok: false,
          error: extractApiErrorMessage(error, 'Update login access failed.'),
        }
      }
    },
    async removeUser(accessToken: string, userId: string): Promise<UserActionResult> {
      try {
        await deleteManagedUser(accessToken, userId)
        const index = this.users.findIndex((user) => user.id === userId)
        if (index >= 0) {
          this.users.splice(index, 1)
        }
        if (this.selectedUserId === userId) {
          this.selectedUserId = getSortedUsers(this.users)[0]?.id ?? null
        }
        return { ok: true }
      } catch (error) {
        return {
          ok: false,
          error: extractApiErrorMessage(error, 'Delete user failed.'),
        }
      }
    },
  },
})
