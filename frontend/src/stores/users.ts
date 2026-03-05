import { defineStore } from 'pinia'

export type UserRole = 'admin' | 'staff' | 'user'
export type UserStatus = 'active' | 'invited' | 'suspended'

export type RoleFilter = UserRole | 'all'
export type StatusFilter = UserStatus | 'all'

export interface ManagedUser {
  id: string
  fullName: string
  email: string
  role: UserRole
  status: UserStatus
  createdAt: string
  lastActiveAt: string
  totalSessions: number
  totalMessages: number
  storageMb: number
}

interface CreateUserPayload {
  fullName: string
  email: string
  role: UserRole
}

function makeUserId(): string {
  return `usr_${Date.now()}_${Math.floor(Math.random() * 100_000)}`
}

function nowIso(): string {
  return new Date().toISOString()
}

function daysAgo(days: number): string {
  const date = new Date()
  date.setDate(date.getDate() - days)
  return date.toISOString()
}

const seedUsers: ManagedUser[] = [
  {
    id: makeUserId(),
    fullName: 'Emilie Catlin',
    email: 'emilie@script.ai',
    role: 'admin',
    status: 'active',
    createdAt: daysAgo(320),
    lastActiveAt: nowIso(),
    totalSessions: 188,
    totalMessages: 1221,
    storageMb: 812,
  },
  {
    id: makeUserId(),
    fullName: 'Duc Tran',
    email: 'duc.tran@script.ai',
    role: 'staff',
    status: 'active',
    createdAt: daysAgo(190),
    lastActiveAt: daysAgo(1),
    totalSessions: 102,
    totalMessages: 688,
    storageMb: 235,
  },
  {
    id: makeUserId(),
    fullName: 'Minh Nguyen',
    email: 'minh.nguyen@script.ai',
    role: 'user',
    status: 'active',
    createdAt: daysAgo(82),
    lastActiveAt: daysAgo(0),
    totalSessions: 57,
    totalMessages: 341,
    storageMb: 126,
  },
  {
    id: makeUserId(),
    fullName: 'An Le',
    email: 'an.le@script.ai',
    role: 'user',
    status: 'invited',
    createdAt: daysAgo(6),
    lastActiveAt: daysAgo(6),
    totalSessions: 0,
    totalMessages: 0,
    storageMb: 0,
  },
  {
    id: makeUserId(),
    fullName: 'Trang Bui',
    email: 'trang.bui@script.ai',
    role: 'staff',
    status: 'suspended',
    createdAt: daysAgo(41),
    lastActiveAt: daysAgo(4),
    totalSessions: 24,
    totalMessages: 81,
    storageMb: 34,
  },
  {
    id: makeUserId(),
    fullName: 'Huy Pham',
    email: 'huy.pham@script.ai',
    role: 'user',
    status: 'active',
    createdAt: daysAgo(17),
    lastActiveAt: daysAgo(2),
    totalSessions: 16,
    totalMessages: 66,
    storageMb: 28,
  },
]

export const useUsersStore = defineStore('users', {
  state: () => ({
    users: seedUsers,
    searchTerm: '',
    roleFilter: 'all' as RoleFilter,
    statusFilter: 'all' as StatusFilter,
    selectedUserId: seedUsers[0]?.id ?? null,
  }),
  getters: {
    filteredUsers(state): ManagedUser[] {
      const term = state.searchTerm.trim().toLowerCase()
      const filtered = state.users.filter((user) => {
        if (state.roleFilter !== 'all' && user.role !== state.roleFilter) {
          return false
        }
        if (state.statusFilter !== 'all' && user.status !== state.statusFilter) {
          return false
        }
        if (!term) {
          return true
        }
        return user.fullName.toLowerCase().includes(term) || user.email.toLowerCase().includes(term)
      })

      return filtered.sort((a, b) => {
        return new Date(b.lastActiveAt).getTime() - new Date(a.lastActiveAt).getTime()
      })
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
    activeUsers(state): number {
      return state.users.filter((user) => user.status === 'active').length
    },
    suspendedUsers(state): number {
      return state.users.filter((user) => user.status === 'suspended').length
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
    setStatusFilter(value: StatusFilter) {
      this.statusFilter = value
    },
    selectUser(userId: string | null) {
      this.selectedUserId = userId
    },
    createUser(payload: CreateUserPayload): ManagedUser {
      let email = payload.email.trim().toLowerCase()
      let duplicateIndex = 1
      while (this.users.some((user) => user.email === email)) {
        duplicateIndex += 1
        const [namePart, domainPart] = payload.email.trim().toLowerCase().split('@')
        email = `${namePart}+${duplicateIndex}@${domainPart ?? 'script.ai'}`
      }

      const createdUser: ManagedUser = {
        id: makeUserId(),
        fullName: payload.fullName.trim(),
        email,
        role: payload.role,
        status: 'invited',
        createdAt: nowIso(),
        lastActiveAt: nowIso(),
        totalSessions: 0,
        totalMessages: 0,
        storageMb: 0,
      }

      this.users.unshift(createdUser)
      this.selectedUserId = createdUser.id
      return createdUser
    },
    updateRole(userId: string, role: UserRole): boolean {
      const target = this.users.find((user) => user.id === userId)
      if (!target) {
        return false
      }
      target.role = role
      return true
    },
    toggleUserSuspension(userId: string): boolean {
      const target = this.users.find((user) => user.id === userId)
      if (!target) {
        return false
      }
      target.status = target.status === 'suspended' ? 'active' : 'suspended'
      target.lastActiveAt = nowIso()
      return true
    },
    removeUser(userId: string): boolean {
      const index = this.users.findIndex((user) => user.id === userId)
      if (index < 0) {
        return false
      }
      this.users.splice(index, 1)
      if (this.selectedUserId === userId) {
        this.selectedUserId = this.users[0]?.id ?? null
      }
      return true
    },
  },
})
