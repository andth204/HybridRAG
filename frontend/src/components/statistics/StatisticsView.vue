<script setup lang="ts">
import { computed } from 'vue'
import { type DocumentStatus, useDocumentsStore } from '@/stores/documents'
import { useHistoryStore } from '@/stores/history'
import { useChatStore } from '@/stores/chat'

const docsStore = useDocumentsStore()
const historyStore = useHistoryStore()
const chatStore = useChatStore()

const userStats = {
  totalUsers: 1284,
  newUsers7d: 96,
  activeNow: 173,
  avgSessionMinutes: 12.6,
  retentionRate: 67,
}

const hourlyActiveUsers = [
  { hour: '07:00', users: 42 },
  { hour: '09:00', users: 76 },
  { hour: '11:00', users: 91 },
  { hour: '14:00', users: 120 },
  { hour: '17:00', users: 108 },
  { hour: '20:00', users: 146 },
  { hour: '22:00', users: 97 },
]

const chartWidth = 720
const chartHeight = 248
const chartPadding = { top: 18, right: 16, bottom: 34, left: 36 }
const chartInnerWidth = chartWidth - chartPadding.left - chartPadding.right
const chartInnerHeight = chartHeight - chartPadding.top - chartPadding.bottom

const visibleHistoryItems = computed(() => {
  return historyStore.items.filter((item) => !historyStore.deletedIds.has(item.id))
})

const totalSessionCount = computed(() => visibleHistoryItems.value.length)
const totalChatMessageCount = computed(() => chatStore.messages.length)

const totalDocumentCount = computed(() => docsStore.documents.length)
const indexedDocumentCount = computed(() => docsStore.indexedCount)
const indexingDocumentCount = computed(() => docsStore.indexingCount)
const failedDocumentCount = computed(() => docsStore.failedCount)

const totalStorageMb = computed(() => {
  const totalBytes = docsStore.documents.reduce((sum, item) => sum + item.sizeBytes, 0)
  return (totalBytes / (1024 * 1024)).toFixed(1)
})

const avgIndexProgress = computed(() => {
  const activeItems = docsStore.documents.filter((item) => item.status === 'indexing')
  if (activeItems.length === 0) {
    return 100
  }
  const total = activeItems.reduce((sum, item) => sum + item.progress, 0)
  return Math.round(total / activeItems.length)
})

const peakHourData = computed(() => {
  return [...hourlyActiveUsers].sort((a, b) => b.users - a.users)[0]
})

const peakHourLabel = computed(() => {
  if (!peakHourData.value) {
    return '--'
  }
  const startHour = Number(peakHourData.value.hour.split(':')[0] ?? 0)
  const nextHour = String((startHour + 1) % 24).padStart(2, '0')
  return `${peakHourData.value.hour} - ${nextHour}:00`
})

const maxUsers = computed(() => {
  return Math.max(...hourlyActiveUsers.map((item) => item.users), 1)
})

const chartPoints = computed(() => {
  const denominator = Math.max(1, hourlyActiveUsers.length - 1)
  return hourlyActiveUsers.map((item, index) => {
    const x = chartPadding.left + (index / denominator) * chartInnerWidth
    const y = chartPadding.top + chartInnerHeight - (item.users / maxUsers.value) * chartInnerHeight
    return {
      ...item,
      x: Number(x.toFixed(2)),
      y: Number(y.toFixed(2)),
      isPeak: item.users === peakHourData.value?.users,
    }
  })
})

const userLinePath = computed(() => {
  return chartPoints.value.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ')
})

const userAreaPath = computed(() => {
  const points = chartPoints.value
  if (points.length === 0) {
    return ''
  }
  const firstPoint = points[0]
  const lastPoint = points[points.length - 1]
  if (!firstPoint || !lastPoint) {
    return ''
  }
  const baseline = chartPadding.top + chartInnerHeight
  const head = `M ${firstPoint.x} ${baseline}`
  const body = points.map((point) => `L ${point.x} ${point.y}`).join(' ')
  const tail = `L ${lastPoint.x} ${baseline} Z`
  return `${head} ${body} ${tail}`
})

const yTicks = computed(() => {
  const ratios = [0, 0.25, 0.5, 0.75, 1]
  return ratios.map((ratio) => ({
    label: Math.round(maxUsers.value * ratio),
    y: chartPadding.top + chartInnerHeight - ratio * chartInnerHeight,
  }))
})

const recentDocuments = computed(() => {
  return [...docsStore.documents].sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
})

const statusStats = computed(() => {
  const total = totalDocumentCount.value || 1
  return [
    {
      key: 'indexed' as DocumentStatus,
      label: 'Indexed',
      count: indexedDocumentCount.value,
      percent: Math.round((indexedDocumentCount.value / total) * 100),
    },
    {
      key: 'indexing' as DocumentStatus,
      label: 'Indexing',
      count: indexingDocumentCount.value,
      percent: Math.round((indexingDocumentCount.value / total) * 100),
    },
    {
      key: 'failed' as DocumentStatus,
      label: 'Failed',
      count: failedDocumentCount.value,
      percent: Math.round((failedDocumentCount.value / total) * 100),
    },
  ]
})

function formatTimestamp(value: string) {
  const date = new Date(value)
  return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
}

function statusClass(status: DocumentStatus) {
  return `stats-status-${status}`
}
</script>

<template>
  <div class="stats-view">
    <section class="stats-kpi-grid">
      <article class="stats-kpi-card">
        <span class="material-icons-outlined stats-kpi-icon">groups</span>
        <div class="stats-kpi-label">Users</div>
        <div class="stats-kpi-value">{{ userStats.totalUsers }}</div>
        <div class="stats-kpi-meta">total accounts</div>
      </article>

      <article class="stats-kpi-card">
        <span class="material-icons-outlined stats-kpi-icon">person_add</span>
        <div class="stats-kpi-label">New users (7d)</div>
        <div class="stats-kpi-value">{{ userStats.newUsers7d }}</div>
        <div class="stats-kpi-meta">+7.1% week-over-week</div>
      </article>

      <article class="stats-kpi-card">
        <span class="material-icons-outlined stats-kpi-icon">bolt</span>
        <div class="stats-kpi-label">Active now</div>
        <div class="stats-kpi-value">{{ userStats.activeNow }}</div>
        <div class="stats-kpi-meta">retention {{ userStats.retentionRate }}%</div>
      </article>

      <article class="stats-kpi-card">
        <span class="material-icons-outlined stats-kpi-icon">description</span>
        <div class="stats-kpi-label">Total documents</div>
        <div class="stats-kpi-value">{{ totalDocumentCount }}</div>
        <div class="stats-kpi-meta">{{ totalStorageMb }} MB stored</div>
      </article>
    </section>

    <section class="stats-panels-two">
      <article class="stats-panel">
        <div class="stats-panel-title">User activity by hour</div>
        <div class="stats-user-chips">
          <span class="stats-user-chip">Avg session {{ userStats.avgSessionMinutes }} min</span>
          <span class="stats-user-chip">Sessions {{ totalSessionCount }}</span>
          <span class="stats-user-chip">Messages {{ totalChatMessageCount }}</span>
        </div>

        <div class="stats-chart-wrap" role="img" aria-label="User activity trend by hour">
          <svg
            class="stats-chart"
            :viewBox="`0 0 ${chartWidth} ${chartHeight}`"
            xmlns="http://www.w3.org/2000/svg"
            preserveAspectRatio="none"
          >
            <g>
              <line
                v-for="tick in yTicks"
                :key="`grid-${tick.label}`"
                class="stats-chart-grid-line"
                :x1="chartPadding.left"
                :x2="chartWidth - chartPadding.right"
                :y1="tick.y"
                :y2="tick.y"
              />
              <text
                v-for="tick in yTicks"
                :key="`label-${tick.label}`"
                class="stats-chart-grid-label"
                :x="chartPadding.left - 8"
                :y="tick.y + 4"
              >
                {{ tick.label }}
              </text>
            </g>

            <path class="stats-chart-area" :d="userAreaPath" />
            <path class="stats-chart-line" :d="userLinePath" />

            <g v-for="point in chartPoints" :key="`point-${point.hour}`">
              <circle
                class="stats-chart-point"
                :class="{ 'is-peak': point.isPeak }"
                :cx="point.x"
                :cy="point.y"
                r="4"
              />
              <text class="stats-chart-xlabel" :x="point.x" :y="chartHeight - 10">
                {{ point.hour }}
              </text>
            </g>
          </svg>
        </div>
      </article>

      <article class="stats-panel">
        <div class="stats-panel-title">Document indexing health</div>
        <div class="stats-user-chips">
          <span class="stats-user-chip">Documents {{ totalDocumentCount }}</span>
          <span class="stats-user-chip">Storage {{ totalStorageMb }} MB</span>
        </div>
        <div class="stats-progress-list">
          <div v-for="item in statusStats" :key="item.key" class="stats-progress-item">
            <div class="stats-progress-head">
              <span class="stats-progress-label">{{ item.label }}</span>
              <span class="stats-progress-count">{{ item.count }} ({{ item.percent }}%)</span>
            </div>
            <div class="stats-progress-track">
              <span
                class="stats-progress-fill"
                :class="statusClass(item.key)"
                :style="{ width: `${item.percent}%` }"
              ></span>
            </div>
          </div>
        </div>
        <p class="stats-panel-foot">Average active indexing progress: {{ avgIndexProgress }}%</p>
      </article>
    </section>

    <section class="stats-panel stats-documents-panel">
      <div class="stats-panel-title">Recent documents</div>
      <div v-if="recentDocuments.length" class="stats-doc-list">
        <div v-for="doc in recentDocuments" :key="doc.id" class="stats-doc-row">
          <div class="stats-doc-main">
            <div class="stats-doc-name" :title="doc.key">{{ doc.key }}</div>
            <div class="stats-doc-time">{{ formatTimestamp(doc.updatedAt) }}</div>
          </div>
          <div class="stats-doc-side">
            <span class="stats-doc-status" :class="statusClass(doc.status)">{{ doc.status }}</span>
            <span class="stats-doc-progress">{{ doc.progress }}%</span>
          </div>
        </div>
      </div>
      <div v-else class="stats-empty">No document data yet.</div>
    </section>
  </div>
</template>
