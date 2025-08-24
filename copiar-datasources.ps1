param(
  [switch]$Flatten  # Usa -Flatten si quieres copiar sin subcarpetas
)

# === Configuración ===
$project = "C:\Users\MarteenMunevarBarrer\Documents\pds-proyecto-final"
$rawDest = Join-Path $project "data\raw"
$tempDir = Join-Path $env:TEMP ("btg-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$repoUrl = "https://github.com/feliperussi/bridging-the-gap-in-health-literacy.git"
$repoSrc = Join-Path $tempDir "data_collection_and_processing\Data Sources"

# === Preparación ===
Write-Host "Creando carpeta temporal: $tempDir"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

Write-Host "Clonando repositorio..."
git clone $repoUrl $tempDir | Out-Null

if (-not (Test-Path $repoSrc)) {
  Write-Error "No se encontró la carpeta de origen: $repoSrc"
  exit 1
}

Write-Host "Asegurando destino: $rawDest"
New-Item -ItemType Directory -Force -Path $rawDest | Out-Null

# === Copia de archivos ===
if ($Flatten) {
  Write-Host "Copiando archivos (aplanado, sin subcarpetas) ..."
  $items = Get-ChildItem $repoSrc -Recurse -Include *.txt,*.csv -File
  $count = 0
  foreach ($it in $items) {
    $target = Join-Path $rawDest $it.Name
    # Si te preocupa colisiones de nombre, usa un prefijo por carpeta:
    # $prefix = $it.Directory.Name
    # $target = Join-Path $rawDest ("{0}_{1}" -f $prefix, $it.Name)
    Copy-Item $it.FullName $target -Force
    $count++
  }
  Write-Host "Copiados $count archivos."
}
else {
  Write-Host "Copiando archivos (conservando estructura de subcarpetas) ..."
  # Importante: comodines entre comillas simples para que PowerShell no los expanda
  robocopy $repoSrc $rawDest '*.txt' '*.csv' /E | Out-Null

  # Conteo simple post-copia
  $count = (Get-ChildItem $rawDest -Recurse -Include *.txt,*.csv -File).Count
  Write-Host "Copiados (o ya presentes) $count archivos en $rawDest."
}

# === Limpieza ===
Write-Host "Eliminando carpeta temporal..."
Remove-Item -Recurse -Force $tempDir

Write-Host "Listo. Datos en: $rawDest"
