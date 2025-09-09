param(
  [Parameter(Mandatory=$true)][string]$Root
)

$ErrorActionPreference='Stop'

if (-not (Test-Path $Root)) {
  Write-Error "Root path not found: $Root"
  exit 1
}

$filtersDir = Join-Path $Root 'filters'
New-Item -ItemType Directory -Force -Path $filtersDir | Out-Null

Get-ChildItem -Path $Root -Directory -Filter 'run_*' | ForEach-Object {
  $runDir  = $_.FullName
  $runName = $_.Name

  if ($runName -match 'run_(\d+)_s\d+_layer\d+_f(?<f>\d+)_it(?<it>[^_]+)_lr(?<lr>[^_]+)_tv(?<tv>[^_]+)_l2(?<l2>[^_]+)_sup(?<sup>[^_]+)_act(?<act>.+)$') {
    $f   = $Matches['f']
    $lr  = $Matches['lr']
    $tv  = $Matches['tv']
    $l2  = $Matches['l2']
    $sup = $Matches['sup']
    $act = $Matches['act']

    $destFilter = Join-Path $filtersDir ("f$($f)")
    $destRun    = Join-Path $destFilter $runName
    New-Item -ItemType Directory -Force -Path $destFilter,$destRun | Out-Null

    Get-ChildItem -Path $runDir -Filter 'comprehensive*.png' -File | ForEach-Object {
      $newName = ("comprehensive_filter_{0}_lr{1}_tv{2}_l2{3}_sup{4}_act{5}.png" -f $f,$lr,$tv,$l2,$sup,$act)
      Copy-Item -Path $_.FullName -Destination (Join-Path $destRun $newName) -Force
    }

    $summary = Join-Path $runDir 'summary.json'
    if (Test-Path $summary) {
      Copy-Item $summary -Destination (Join-Path $destRun 'summary.json') -Force
    }
  }
}

Write-Output 'Done'


