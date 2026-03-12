<script setup lang="ts">
import dayjs from 'dayjs'
import { storeToRefs } from 'pinia'
import { onMounted, ref } from 'vue'
import { useDocumentsStore, type DocumentItem, type DocumentStatus } from '@/stores/documents'

const docsStore = useDocumentsStore()
const fileInputRef = ref<HTMLInputElement | null>(null)

const { failedCount, filteredDocuments, indexedCount, indexingCount, isUploading, searchTerm } =
  storeToRefs(docsStore)

onMounted(() => {
  void docsStore.syncWithSession()
})

function openPicker() {
  fileInputRef.value?.click()
}

function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length) {
    return
  }
  void docsStore.uploadFiles(Array.from(input.files))
  input.value = ''
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 B'
  }
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  const fixed = size >= 100 ? 0 : 1
  return `${size.toFixed(fixed)} ${units[unitIndex]}`
}

function formatTimestamp(value: string) {
  return dayjs(value).format('DD/MM/YYYY HH:mm')
}

function getStatusText(status: DocumentStatus) {
  if (status === 'indexed') {
    return 'Indexed'
  }
  if (status === 'indexing') {
    return 'Indexing'
  }
  if (status === 'deleting') {
    return 'Deleting'
  }
  return 'Failed'
}

function statusClass(status: DocumentStatus) {
  return `docs-status-${status}`
}

function onReindex(item: DocumentItem) {
  docsStore.requestReindex(item.id)
}

function onDelete(item: DocumentItem) {
  docsStore.requestDelete(item.id)
}

</script>

<template>
  <div class="documents-view">
    <div class="docs-header">
      <div class="docs-actions">
        <input
          ref="fileInputRef"
          class="docs-file-input"
          type="file"
          multiple
          @change="handleFileSelect"
        />
        <button class="docs-primary-btn" type="button" :disabled="isUploading" @click="openPicker">
          <span class="material-icons-outlined">upload_file</span>
          {{ isUploading ? 'Uploading...' : 'Add Files' }}
        </button>
      </div>
    </div>

    <div class="docs-toolbar">
      <div class="docs-search-wrap">
        <span class="material-icons-outlined">search</span>
        <input v-model="searchTerm" type="text" placeholder="Search by file name..." />
      </div>
      <div class="docs-stats">
        <span class="docs-stat-pill">{{ filteredDocuments.length }} documents</span>
        <span class="docs-stat-pill status-indexed">{{ indexedCount }} indexed</span>
        <span class="docs-stat-pill status-indexing">{{ indexingCount }} indexing</span>
        <span class="docs-stat-pill status-failed">{{ failedCount }} failed</span>
      </div>
    </div>

    <div class="docs-list">
      <div v-if="filteredDocuments.length === 0" class="docs-empty">
        <span class="material-icons-outlined">folder_open</span>
        <p>No documents yet. Upload files to start indexing.</p>
      </div>

      <article v-for="item in filteredDocuments" :key="item.id" class="docs-item">
        <div class="docs-item-left">
          <div class="docs-file-icon">
            <span class="material-icons-outlined">description</span>
          </div>
          <div class="docs-file-meta">
            <p class="docs-file-name" :title="item.key">{{ item.key }}</p>
            <p class="docs-file-sub">
              <span>{{ formatBytes(item.sizeBytes) }}</span>
              <span>|</span>
              <span>{{ formatTimestamp(item.updatedAt) }}</span>
              <span v-if="item.fileId">|</span>
              <span v-if="item.fileId">ID: {{ item.fileId }}</span>
            </p>
            <div v-if="item.status === 'indexing' || item.status === 'failed'" class="docs-progress-row">
              <div
                class="docs-progress-track"
                role="progressbar"
                :aria-valuemin="0"
                :aria-valuemax="100"
                :aria-valuenow="item.progress"
                :aria-label="`Indexing progress for ${item.key}`"
              >
                <span
                  class="docs-progress-fill"
                  :class="{ failed: item.status === 'failed' }"
                  :style="{ width: `${item.progress}%` }"
                ></span>
              </div>
              <span class="docs-progress-value" :class="{ failed: item.status === 'failed' }">
                {{ item.progress }}%
              </span>
            </div>
          </div>
        </div>

        <div class="docs-item-right">
          <span class="docs-status-pill" :class="statusClass(item.status)">
            {{ getStatusText(item.status) }}
          </span>
          <button
            class="docs-mini-btn"
            type="button"
            :disabled="item.status === 'indexing' || item.status === 'deleting'"
            @click="onReindex(item)"
          >
            <span class="material-icons-outlined">sync</span>
            Re-index
          </button>
          <button
            class="docs-mini-btn danger"
            type="button"
            :disabled="item.status === 'indexing'"
            @click="onDelete(item)"
          >
            <span class="material-icons-outlined">delete_outline</span>
            {{ item.status === 'deleting' ? 'Retry delete' : 'Delete' }}
          </button>
        </div>
      </article>
    </div>
  </div>
</template>

