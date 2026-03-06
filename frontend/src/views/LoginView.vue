<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useUiStore } from '@/stores/ui'
import { renderGoogleSignInButton } from '@/services/googleIdentity'

const router = useRouter()
const authStore = useAuthStore()
const uiStore = useUiStore()
const googleClientId = import.meta.env.VITE_GOOGLE_CLIENT_ID?.trim() || ''

const isSubmitting = ref(false)
const errorMessage = ref('')
const googleButtonContainer = ref<HTMLElement | null>(null)

async function handleGoogleLoginWithIdToken(idToken: string) {
  errorMessage.value = ''
  isSubmitting.value = true
  try {
    const result = await authStore.signInWithGoogleIdToken(idToken)
    if (!result.ok) {
      errorMessage.value = result.error || 'Google sign-in failed.'
      return
    }

    uiStore.switchMainView('chat')
    uiStore.setRightPanelCollapsed(true)
    await router.push('/workspace')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'Google sign-in failed.'
  } finally {
    isSubmitting.value = false
  }
}

async function mountGoogleButton() {
  if (!googleClientId) {
    errorMessage.value = 'VITE_GOOGLE_CLIENT_ID is empty. Configure it before signing in with Google.'
    return
  }

  const container = googleButtonContainer.value
  if (!container) {
    return
  }

  try {
    await renderGoogleSignInButton(
      googleClientId,
      container,
      (idToken) => {
        void handleGoogleLoginWithIdToken(idToken)
      },
      (message) => {
        errorMessage.value = message
      },
    )
  } catch (error) {
    errorMessage.value =
      error instanceof Error
        ? error.message
        : 'Google sign-in is unavailable right now. Please refresh and try again.'
  }
}

onMounted(() => {
  void mountGoogleButton()
})
</script>

<template>
  <div class="login-page">
    <div class="login-shell">
      <section class="login-left">
        <h1>Welcome back</h1>
        <p class="login-subtitle">
          Continue with your Google account to access the university assistant.
        </p>

        <div class="login-form">
          <div v-if="errorMessage" class="login-error">{{ errorMessage }}</div>
        </div>

        <div class="login-social-slot" :class="{ 'is-disabled': isSubmitting }">
          <div ref="googleButtonContainer" class="login-google-button" />
        </div>
      </section>

      <section class="login-right">
        <img src="/monstera.jpg" alt="Monstera plant" />
        <div class="login-image-overlay">
          <span>University Workspace</span>
          <h2>Sign in to ask about the university.</h2>
        </div>
      </section>
    </div>
  </div>
</template>
