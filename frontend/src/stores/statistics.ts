import axios from 'axios'
import { defineStore } from 'pinia'

import { fetchStatisticsOverview, type BackendStatisticsOverview } from '@/services/statisticsApi'
import { useAuthStore } from '@/stores/auth'

export interface StatisticsHourPoint {
  bucketStart: string
  label: string
  userCount: number
}

function parseStatisticsDate(value: string): Date | null {
  const normalized = value.trim()
  if (!normalized) {
    return null
  }
  const date = new Date(normalized)
  if (Number.isNaN(date.getTime())) {
    return null
  }
  return date
}

function formatHourLabel(bucketStart: string, fallbackLabel: string): string {
  const date = parseStatisticsDate(bucketStart)
  if (!date) {
    return fallbackLabel || '--:--'
  }
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
}

export interface StatisticsOverview {
  totalUsers: number
  managerUsers: number
  standardUsers: number
  newUsers7d: number
  activeUsers24h: number
  retentionRate7d: number
  avgSessionMinutes: number
  totalSessions: number
  totalMessages: number
}

function toOverview(data: BackendStatisticsOverview): StatisticsOverview {
  return {
    totalUsers: Number(data.total_users || 0),
    managerUsers: Number(data.manager_users || 0),
    standardUsers: Number(data.standard_users || 0),
    newUsers7d: Number(data.new_users_7d || 0),
    activeUsers24h: Number(data.active_users_24h || 0),
    retentionRate7d: Number(data.retention_rate_7d || 0),
    avgSessionMinutes: Number(data.avg_session_minutes || 0),
    totalSessions: Number(data.total_sessions || 0),
    totalMessages: Number(data.total_messages || 0),
  }
}

function toHourlyActivity(data: BackendStatisticsOverview): StatisticsHourPoint[] {
  return (data.hourly_activity || []).map((item) => ({
    bucketStart: item.bucket_start || '',
    label: formatHourLabel(item.bucket_start || '', item.label),
    userCount: Number(item.user_count || 0),
  }))
}

function emptyOverview(): StatisticsOverview {
  return {
    totalUsers: 0,
    managerUsers: 0,
    standardUsers: 0,
    newUsers7d: 0,
    activeUsers24h: 0,
    retentionRate7d: 0,
    avgSessionMinutes: 0,
    totalSessions: 0,
    totalMessages: 0,
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

export const useStatisticsStore = defineStore('statistics', {
  state: () => ({
    overview: emptyOverview(),
    hourlyActivity: [] as StatisticsHourPoint[],
    isLoading: false,
    errorMessage: '',
  }),
  actions: {
    reset() {
      this.overview = emptyOverview()
      this.hourlyActivity = []
      this.isLoading = false
      this.errorMessage = ''
    },
    async fetchOverview(accessToken: string): Promise<boolean> {
      this.isLoading = true
      this.errorMessage = ''
      try {
        const data = await fetchStatisticsOverview(accessToken)
        this.overview = toOverview(data)
        this.hourlyActivity = toHourlyActivity(data)
        return true
      } catch (error) {
        this.errorMessage = extractApiErrorMessage(error, 'Failed to load statistics.')
        return false
      } finally {
        this.isLoading = false
      }
    },
    async syncWithSession(): Promise<boolean> {
      const authStore = useAuthStore()
      if (!authStore.isAuthenticated || authStore.currentUser?.role !== 'manager') {
        this.reset()
        return false
      }
      const ok = await authStore.ensureSession()
      if (!ok || !authStore.accessToken.trim()) {
        this.reset()
        return false
      }
      return this.fetchOverview(authStore.accessToken.trim())
    },
  },
})
