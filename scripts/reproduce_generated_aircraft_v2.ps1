[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$asset = Join-Path $root 'assets\aircraft\generated_plane_v2'
$python = Join-Path $root '.venv311\Scripts\python.exe'
$generator = Join-Path $root 'scripts\generate_animated_aircraft.py'
$audit = Join-Path $root 'scripts\audit_generated_aircraft_v2.py'
$manifestPath = Join-Path $asset 'generation_manifest.json'
$geometryPath = Join-Path $asset 'geometry_audit.json'
$blendPath = Join-Path $asset 'generated_plane_v2.blend'
$glbPath = Join-Path $asset 'generated_plane_v2.glb'

if (-not (Test-Path $python)) { throw "Missing repository interpreter: $python" }
if (-not (Test-Path $manifestPath)) { throw "Missing V2 baseline manifest: $manifestPath" }

$candidates = @(
  $env:BLENDER_PATH,
  'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe',
  'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe',
  'C:\Program Files\Blender Foundation\Blender 4.3\blender.exe'
) | Where-Object { $_ -and (Test-Path $_) }
if (-not $candidates) {
  $command = Get-Command blender -ErrorAction SilentlyContinue
  if ($command) { $candidates = @($command.Source) }
}
if (-not $candidates) { throw 'Blender was not detected. Set BLENDER_PATH explicitly.' }
$blender = @($candidates)[0]
$version = (& $blender --background --version | Select-Object -First 1)
if ($LASTEXITCODE -ne 0) { throw 'Blender background-mode version check failed.' }

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$geometryTriangles = if (Test-Path $geometryPath) {
  (Get-Content $geometryPath -Raw | ConvertFrom-Json).totals.triangle_count
} else { $manifest.triangle_count }
$baseline = [ordered]@{
  manifest = $manifest
  geometry_triangle_count = $geometryTriangles
  blend_sha256 = (Get-FileHash $blendPath -Algorithm SHA256).Hash.ToLowerInvariant()
  glb_sha256 = (Get-FileHash $glbPath -Algorithm SHA256).Hash.ToLowerInvariant()
}
$baselinePath = Join-Path $env:TEMP 'generated_plane_v2_reproduction_baseline.json'
[IO.File]::WriteAllText($baselinePath, ($baseline | ConvertTo-Json -Depth 100), (New-Object Text.UTF8Encoding($false)))

Write-Host "Blender: $blender"
Write-Host "Version: $version"
Write-Host 'Generating V2 headlessly...'
& $blender --background --python $generator
if ($LASTEXITCODE -ne 0) { throw 'Fatal: Blender generation failed.' }

Write-Host 'Running geometry, material, animation, schema, Panda3D, and checksum audits...'
& $python $audit --baseline-manifest $baselinePath
if ($LASTEXITCODE -ne 0) { throw 'Fatal: generated-aircraft audit failed.' }

Write-Host 'Running focused audit tests...'
& $python -m pytest tests\unit\test_generated_aircraft.py tests\unit\test_generated_aircraft_v2.py tests\unit\test_generated_aircraft_audit.py -q
if ($LASTEXITCODE -ne 0) { throw 'Fatal: generated-aircraft tests failed.' }

& $python $audit --verify-only
if ($LASTEXITCODE -ne 0) { throw 'Fatal: immutable-artifact checksum verification failed.' }

[ordered]@{
  status = 'REVIEW_REQUIRED'
  blender_path = $blender
  blender_version = $version
  semantic_reproducibility = (Get-Content (Join-Path $asset 'reproduction_comparison.json') -Raw | ConvertFrom-Json).semantic_reproducibility
  audit_summary = 'assets/aircraft/generated_plane_v2/audit_summary.json'
  limitations = @('Visualization-only; dimensional sanity ranges are not certification standards.', 'Binary hashes may vary when Blender embeds metadata.')
} | ConvertTo-Json -Depth 5 | ForEach-Object {
  [IO.File]::WriteAllText((Join-Path $asset 'reproduction_summary.json'), $_, (New-Object Text.UTF8Encoding($false)))
}

Write-Host 'Controlled reproduction completed. Overall status remains REVIEW_REQUIRED because aesthetic assumptions are not physically validated.'
