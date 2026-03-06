interface GoogleCredentialResponse {
  credential?: string
}

interface GoogleButtonConfig {
  type?: 'standard' | 'icon'
  theme?: 'outline' | 'filled_blue' | 'filled_black'
  size?: 'large' | 'medium' | 'small'
  text?: 'signin_with' | 'signup_with' | 'continue_with' | 'signin'
  shape?: 'rectangular' | 'pill' | 'circle' | 'square'
  logo_alignment?: 'left' | 'center'
  width?: number
}

interface GoogleIdClient {
  initialize(config: {
    client_id: string
    callback: (response: GoogleCredentialResponse) => void
    auto_select?: boolean
    cancel_on_tap_outside?: boolean
    ux_mode?: 'popup' | 'redirect'
  }): void
  renderButton(parent: HTMLElement, options: GoogleButtonConfig): void
  cancel(): void
}

interface GoogleAccountsNamespace {
  id: GoogleIdClient
}

interface GoogleNamespace {
  accounts: GoogleAccountsNamespace
}

declare global {
  interface Window {
    google?: GoogleNamespace
  }
}

let googleScriptPromise: Promise<void> | null = null

function loadGoogleIdentityScript(): Promise<void> {
  if (window.google?.accounts?.id) {
    return Promise.resolve()
  }
  if (googleScriptPromise) {
    return googleScriptPromise
  }

  googleScriptPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>('script[data-google-identity="1"]')
    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true })
      existing.addEventListener('error', () => reject(new Error('Failed to load Google Identity script.')), {
        once: true,
      })
      return
    }

    const script = document.createElement('script')
    script.src = 'https://accounts.google.com/gsi/client'
    script.async = true
    script.defer = true
    script.dataset.googleIdentity = '1'
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('Failed to load Google Identity script.'))
    document.head.appendChild(script)
  })

  return googleScriptPromise
}

export async function renderGoogleSignInButton(
  clientId: string,
  container: HTMLElement,
  onCredential: (idToken: string) => void,
  onError?: (message: string) => void,
): Promise<void> {
  const trimmedClientId = clientId.trim()
  if (!trimmedClientId) {
    throw new Error('VITE_GOOGLE_CLIENT_ID is empty. Configure it before signing in with Google.')
  }

  await loadGoogleIdentityScript()
  const googleIdClient = window.google?.accounts?.id
  if (!googleIdClient) {
    throw new Error('Google Identity is unavailable in this browser context.')
  }

  googleIdClient.initialize({
    client_id: trimmedClientId,
    auto_select: false,
    cancel_on_tap_outside: true,
    ux_mode: 'popup',
    callback: (response) => {
      const credential = response.credential?.trim()
      if (!credential) {
        onError?.('Google did not return an id_token.')
        return
      }
      onCredential(credential)
    },
  })

  container.innerHTML = ''
  const width = Math.max(220, Math.floor(container.getBoundingClientRect().width || 0))

  googleIdClient.renderButton(container, {
    type: 'standard',
    theme: 'outline',
    size: 'large',
    text: 'continue_with',
    shape: 'rectangular',
    logo_alignment: 'left',
    width,
  })
}
