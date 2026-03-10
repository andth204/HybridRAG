import axios from 'axios'

export interface BackendStatisticsHourPoint {
  bucket_start: string
  label: string
  user_count: number
}

export interface BackendStatisticsOverview {
  total_users: number
  manager_users: number
  standard_users: number
  new_users_7d: number
  active_users_24h: number
  retention_rate_7d: number
  avg_session_minutes: number
  total_sessions: number
  total_messages: number
  hourly_activity: BackendStatisticsHourPoint[]
}

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, '')
}

const apiBaseUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL?.trim() || '/api/v1')

const statisticsApiClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 15_000,
})

function authHeaders(accessToken: string) {
  return {
    Authorization: `Bearer ${accessToken}`,
  }
}

export async function fetchStatisticsOverview(accessToken: string): Promise<BackendStatisticsOverview> {
  const { data } = await statisticsApiClient.get<BackendStatisticsOverview>('/statistics/overview', {
    headers: authHeaders(accessToken),
  })
  return data
}
