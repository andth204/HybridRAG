import axios from 'axios'

export type BackendUserRole = 'manager' | 'user'

export interface BackendManagedUser {
  id: string
  email: string
  username: string | null
  google_id: string | null
  role: BackendUserRole
  is_blocked: boolean
  session_count: number
  message_count: number
  created_at: string
  updated_at: string
}

interface BackendUsersListResponse {
  items: BackendManagedUser[]
}

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, '')
}

const apiBaseUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL?.trim() || '/api/v1')

const usersApiClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 15_000,
})

function authHeaders(accessToken: string) {
  return {
    Authorization: `Bearer ${accessToken}`,
  }
}

export async function fetchUsersList(accessToken: string): Promise<BackendManagedUser[]> {
  const { data } = await usersApiClient.get<BackendUsersListResponse>('/users', {
    headers: authHeaders(accessToken),
  })
  return data.items
}

export async function patchManagedUserRole(
  accessToken: string,
  userId: string,
  role: BackendUserRole,
): Promise<BackendManagedUser> {
  const { data } = await usersApiClient.patch<BackendManagedUser>(
    `/users/${encodeURIComponent(userId)}/role`,
    { role },
    {
      headers: authHeaders(accessToken),
    },
  )
  return data
}

export async function patchManagedUserLoginAccess(
  accessToken: string,
  userId: string,
  isBlocked: boolean,
): Promise<BackendManagedUser> {
  const { data } = await usersApiClient.patch<BackendManagedUser>(
    `/users/${encodeURIComponent(userId)}/login-access`,
    { is_blocked: isBlocked },
    {
      headers: authHeaders(accessToken),
    },
  )
  return data
}

export async function deleteManagedUser(accessToken: string, userId: string): Promise<void> {
  await usersApiClient.delete(`/users/${encodeURIComponent(userId)}`, {
    headers: authHeaders(accessToken),
  })
}
