import { createRouter, createWebHistory } from 'vue-router'
import WorkspaceView from '../views/WorkspaceView.vue'
import LoginView from '../views/LoginView.vue'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/login',
    },
    {
      path: '/login',
      name: 'login',
      component: LoginView,
    },
    {
      path: '/workspace',
      name: 'workspace',
      component: WorkspaceView,
      meta: { requiresAuth: true },
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/login',
    },
  ],
})

router.beforeEach(async (to) => {
  const authStore = useAuthStore()
  if (!authStore.isInitialized) {
    authStore.initAuth()
  }

  if (to.meta.requiresAuth) {
    const ok = await authStore.ensureSession()
    if (!ok) {
      return { name: 'login' }
    }
    return true
  }

  if (to.name === 'login' && authStore.isAuthenticated) {
    const ok = await authStore.ensureSession()
    if (ok) {
      return { name: 'workspace' }
    }
  }

  return true
})

export default router
