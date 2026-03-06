import axios from 'axios'

export interface BackendUserProfile {
  id: string
  email: string
  username: string | null
  google_id: string | null
  role: 'manager' | 'user'
  created_at: string
  updated_at: string
}

export interface BackendTokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
  user: BackendUserProfile
}

interface RefreshTokenRequest {
  refresh_token: string
}

interface LogoutRequest {
  refresh_token?: string
  revoke_all_user_tokens: boolean
}

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, '')
}

const apiBaseUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL?.trim() || '/api/v1')

const authApiClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 15_000,
})

export async function loginWithGoogleIdToken(idToken: string): Promise<BackendTokenResponse> {
  const { data } = await authApiClient.post<BackendTokenResponse>('/auth/google', {
    id_token: idToken,
  })
  return data
}

export async function refreshAuthToken(payload: RefreshTokenRequest): Promise<BackendTokenResponse> {
  const { data } = await authApiClient.post<BackendTokenResponse>('/auth/refresh', payload)
  return data
}

export async function fetchAuthProfile(accessToken: string): Promise<BackendUserProfile> {
  const { data } = await authApiClient.get<BackendUserProfile>('/auth/me', {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  })
  return data
}

export async function revokeAuthSession(accessToken: string, refreshToken?: string): Promise<void> {
  const payload: LogoutRequest = {
    revoke_all_user_tokens: false,
  }
  if (refreshToken?.trim()) {
    payload.refresh_token = refreshToken.trim()
  }

  await authApiClient.post('/auth/logout', payload, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  })
}
