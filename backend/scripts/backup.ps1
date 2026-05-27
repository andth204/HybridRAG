# Phase 6.8 — Nightly backup (Windows PowerShell variant)
# Postgres + Weaviate snapshot + FAISS/BM25 tarball -> MinIO

$ErrorActionPreference = "Stop"

$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$BackupDir = if ($env:BACKUP_DIR) { $env:BACKUP_DIR } else { "$env:TEMP\hybridrag-backups" }
$MinioBucket = if ($env:MINIO_BUCKET) { $env:MINIO_BUCKET } else { "hybridrag-backups" }
$MinioAlias = if ($env:MINIO_ALIAS) { $env:MINIO_ALIAS } else { "local" }
$RetentionDays = if ($env:RETENTION_DAYS) { [int]$env:RETENTION_DAYS } else { 7 }
$WeaviateUrl = if ($env:WEAVIATE_URL) { $env:WEAVIATE_URL } else { "http://localhost:8080" }
$VectorStoreDir = if ($env:VECTOR_STORE_DIR) { $env:VECTOR_STORE_DIR } else { ".\data\vector_store" }

New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null

Write-Host "[$(Get-Date -Format o)] backup start ts=$Timestamp"

# 1. Postgres
if ($env:POSTGRES_HOST -and $env:POSTGRES_USER -and $env:POSTGRES_DB) {
    $env:PGPASSWORD = $env:POSTGRES_PASSWORD
    $pgPort = if ($env:POSTGRES_PORT) { $env:POSTGRES_PORT } else { "5432" }
    & pg_dump -h $env:POSTGRES_HOST -p $pgPort -U $env:POSTGRES_USER -d $env:POSTGRES_DB `
        -F c -f "$BackupDir\pg_$Timestamp.dump"
    if ($LASTEXITCODE -eq 0) { Write-Host "[ok] pg_dump -> pg_$Timestamp.dump" }
    else { Write-Host "[warn] pg_dump exit=$LASTEXITCODE" }
} else {
    Write-Host "[skip] Postgres env vars missing"
}

# 2. Weaviate snapshot
try {
    $ready = Invoke-WebRequest -Uri "$WeaviateUrl/v1/.well-known/ready" -UseBasicParsing -TimeoutSec 5
    if ($ready.StatusCode -eq 200) {
        $backupId = "hybridrag_$Timestamp"
        $body = @{ id = $backupId; include = @("DocChunk") } | ConvertTo-Json -Compress
        Invoke-WebRequest -Uri "$WeaviateUrl/v1/backups/filesystem" -Method POST `
            -ContentType "application/json" -Body $body `
            -UseBasicParsing -TimeoutSec 30 `
            -OutFile "$BackupDir\weaviate_$Timestamp.json" | Out-Null
        Write-Host "[ok] weaviate snapshot id=$backupId"
    }
} catch {
    Write-Host "[skip] weaviate unreachable at $WeaviateUrl"
}

# 3. FAISS + BM25 tarball
if (Test-Path $VectorStoreDir) {
    $archivePath = "$BackupDir\vectorstore_$Timestamp.zip"
    Compress-Archive -Path "$VectorStoreDir\*" -DestinationPath $archivePath -Force
    Write-Host "[ok] vectorstore zip created"
} else {
    Write-Host "[skip] $VectorStoreDir not found"
}

# 4. Upload to MinIO
if (Get-Command mc -ErrorAction SilentlyContinue) {
    try { & mc mb -p "$MinioAlias/$MinioBucket" 2>$null } catch {}
    Get-ChildItem -Path $BackupDir -Filter "*_$Timestamp.*" | ForEach-Object {
        & mc cp $_.FullName "$MinioAlias/$MinioBucket/"
        if ($LASTEXITCODE -eq 0) { Write-Host "[ok] uploaded $($_.Name)" }
        else { Write-Host "[warn] mc upload failed for $($_.Name)" }
    }
} else {
    Write-Host "[skip] mc CLI not installed"
}

# 5. Retention sweep
$cutoff = (Get-Date).AddDays(-$RetentionDays)
Get-ChildItem -Path $BackupDir -File | Where-Object { $_.LastWriteTime -lt $cutoff } | Remove-Item -Force
Write-Host "[ok] retention sweep (kept $RetentionDays days)"

Write-Host "[$(Get-Date -Format o)] backup done"
