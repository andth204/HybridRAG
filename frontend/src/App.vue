<script setup lang="ts">
import { computed } from 'vue'
import { darkTheme, NConfigProvider, NMessageProvider, type GlobalThemeOverrides } from 'naive-ui'
import { RouterView } from 'vue-router'
import { useUiStore } from '@/stores/ui'

const uiStore = useUiStore()

const naiveTheme = computed(() => (uiStore.isDark ? darkTheme : null))

const FONT_STACK = '"Be Vietnam Pro", "Rubik", system-ui, sans-serif'

const lightOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: '#1c7a4b',
    primaryColorHover: '#15613b',
    primaryColorPressed: '#124f31',
    primaryColorSuppl: '#1c7a4b',
    borderRadius: '10px',
    borderRadiusSmall: '8px',
    fontFamily: FONT_STACK,
  },
}

const darkOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: '#34d399',
    primaryColorHover: '#2bbe88',
    primaryColorPressed: '#26a878',
    primaryColorSuppl: '#34d399',
    borderRadius: '10px',
    borderRadiusSmall: '8px',
    fontFamily: FONT_STACK,
  },
}

const themeOverrides = computed(() => (uiStore.isDark ? darkOverrides : lightOverrides))
</script>

<template>
  <NConfigProvider :theme="naiveTheme" :theme-overrides="themeOverrides">
    <NMessageProvider>
      <RouterView />
    </NMessageProvider>
  </NConfigProvider>
</template>
