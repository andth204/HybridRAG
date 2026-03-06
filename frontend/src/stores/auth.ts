import axios from 'axios'
import { defineStore } from 'pinia'
import {
  fetchAuthProfile,
  loginWithGoogleIdToken,
  refreshAuthToken,
  revokeAuthSession,
  type BackendTokenResponse,
  type BackendUserProfile,
} from '@/services/authApi'

export interface AuthUser {
  id: string
  fullName: string
  email: string
  role: 'manager' | 'user'
}

interface AuthActionResult {
  ok: boolean
  error?: string
}

interface StoredAuthSession {
  currentUser: AuthUser
  accessToken: string
  refreshToken: string
  tokenType: string
  accessTokenExpiresAt: number
}

const AUTH_STORAGE_KEY = 'hybridrag.auth.session'
const ACCESS_TOKEN_EXPIRY_SKEW_MS = 20_000

let refreshInFlight: Promise<boolean> | null = null

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

function toAuthUser(profile: BackendUserProfile): AuthUser {
  const fallbackName = profile.email.split('@')[0] || 'HybridRAG User'
  return {
    id: profile.id,
    fullName: normalizeDisplayName(profile.username || fallbackName),
    email: profile.email.trim().toLowerCase(),
    role: profile.role,
  }
}

function parseStoredSession(raw: string | null): StoredAuthSession | null {
  if (!raw) {
    return null
  }

  try {
    const parsed = JSON.parse(raw) as Partial<StoredAuthSession>
    const user = parsed.currentUser as Partial<AuthUser> | undefined
    if (
      !user ||
      typeof user.id !== 'string' ||
      typeof user.fullName !== 'string' ||
      typeof user.email !== 'string' ||
      (user.role !== 'manager' && user.role !== 'user' && typeof user.role !== 'undefined') ||
      typeof parsed.accessToken !== 'string' ||
      typeof parsed.refreshToken !== 'string' ||
      typeof parsed.tokenType !== 'string' ||
      typeof parsed.accessTokenExpiresAt !== 'number'
    ) {
      return null
    }

    return {
      currentUser: {
        id: user.id,
        fullName: user.fullName,
        email: user.email,
        role: user.role === 'manager' ? 'manager' : 'user',
      },
      accessToken: parsed.accessToken,
      refreshToken: parsed.refreshToken,
      tokenType: parsed.tokenType,
      accessTokenExpiresAt: parsed.accessTokenExpiresAt,
    }
  } catch {
    return null
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

export const useAuthStore = defineStore('auth', {
  state: () => ({
    currentUser: null as AuthUser | null,
    accessToken: '',
    refreshToken: '',
    tokenType: 'bearer',
    accessTokenExpiresAt: 0,
    isInitialized: false,
    hasValidatedSession: false,
  }),
  getters: {
    isAuthenticated: (state) => Boolean(state.currentUser && state.accessToken && state.refreshToken),
    authorizationHeader: (state) => {
      if (!state.accessToken) {
        return ''
      }
      return `${state.tokenType || 'bearer'} ${state.accessToken}`
    },
  },
  actions: {
    initAuth() {
      const session = parseStoredSession(localStorage.getItem(AUTH_STORAGE_KEY))
      if (!session) {
        this.clearSession()
        this.isInitialized = true
        return
      }

      this.currentUser = session.currentUser
      this.accessToken = session.accessToken
      this.refreshToken = session.refreshToken
      this.tokenType = session.tokenType || 'bearer'
      this.accessTokenExpiresAt = session.accessTokenExpiresAt
      this.hasValidatedSession = false
      this.isInitialized = true
    },

    persistSession() {
      if (!this.isAuthenticated || !this.currentUser) {
        localStorage.removeItem(AUTH_STORAGE_KEY)
        return
      }

      const payload: StoredAuthSession = {
        currentUser: this.currentUser,
        accessToken: this.accessToken,
        refreshToken: this.refreshToken,
        tokenType: this.tokenType,
        accessTokenExpiresAt: this.accessTokenExpiresAt,
      }
      localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(payload))
    },

    clearSession() {
      this.currentUser = null
      this.accessToken = ''
      this.refreshToken = ''
      this.tokenType = 'bearer'
      this.accessTokenExpiresAt = 0
      this.hasValidatedSession = false
      localStorage.removeItem(AUTH_STORAGE_KEY)
    },

    applyTokenResponse(payload: BackendTokenResponse) {
      this.currentUser = toAuthUser(payload.user)
      this.accessToken = payload.access_token
      this.refreshToken = payload.refresh_token
      this.tokenType = payload.token_type || 'bearer'
      this.accessTokenExpiresAt = Date.now() + Math.max(1, payload.expires_in) * 1000
      this.hasValidatedSession = true
      this.persistSession()
    },

    isAccessTokenNearExpiry(): boolean {
      if (!this.accessToken || !this.accessTokenExpiresAt) {
        return true
      }
      return Date.now() >= this.accessTokenExpiresAt - ACCESS_TOKEN_EXPIRY_SKEW_MS
    },

    async signInWithGoogleIdToken(idToken: string): Promise<AuthActionResult> {
      try {
        const payload = await loginWithGoogleIdToken(idToken)
        this.applyTokenResponse(payload)
        return { ok: true }
      } catch (error) {
        return {
          ok: false,
          error: extractApiErrorMessage(error, 'Google sign-in failed. Please try again.'),
        }
      }
    },

    async refreshSession(): Promise<boolean> {
      if (refreshInFlight) {
        return refreshInFlight
      }

      if (!this.refreshToken.trim()) {
        this.clearSession()
        return false
      }

      refreshInFlight = (async () => {
        try {
          const payload = await refreshAuthToken({
            refresh_token: this.refreshToken,
          })
          this.applyTokenResponse(payload)
          return true
        } catch {
          this.clearSession()
          return false
        } finally {
          refreshInFlight = null
        }
      })()

      return refreshInFlight
    },

    async validateCurrentProfile(): Promise<boolean> {
      if (!this.accessToken.trim()) {
        return false
      }

      try {
        const profile = await fetchAuthProfile(this.accessToken)
        this.currentUser = toAuthUser(profile)
        this.hasValidatedSession = true
        this.persistSession()
        return true
      } catch {
        return false
      }
    },

    async ensureSession(): Promise<boolean> {
      if (!this.isAuthenticated) {
        return false
      }

      if (this.isAccessTokenNearExpiry()) {
        const refreshed = await this.refreshSession()
        if (!refreshed) {
          return false
        }
      }

      if (this.hasValidatedSession) {
        return true
      }

      const validated = await this.validateCurrentProfile()
      if (validated) {
        return true
      }

      const refreshed = await this.refreshSession()
      if (!refreshed) {
        return false
      }

      const revalidated = await this.validateCurrentProfile()
      if (!revalidated) {
        this.clearSession()
        return false
      }

      return true
    },

    async logout(): Promise<void> {
      const accessToken = this.accessToken
      const refreshToken = this.refreshToken

      this.clearSession()

      if (!accessToken.trim()) {
        return
      }

      try {
        await revokeAuthSession(accessToken, refreshToken)
      } catch {
        // Best-effort revoke. Session is already removed locally.
      }
    },
  },
})
