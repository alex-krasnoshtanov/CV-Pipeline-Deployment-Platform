# One-command local startup for Windows PowerShell.
# Usage: .\scripts\start.ps1 [extra docker compose flags]
$ErrorActionPreference = "Stop"

$RepoRoot    = (Resolve-Path "$PSScriptRoot\..").Path
$EnvFile     = Join-Path $RepoRoot "configs\env\.env"
$ComposeFile = Join-Path $RepoRoot "infra\local\docker-compose.yml"

function Get-EnvTemplate {
    $hiddenTemplate = Join-Path $RepoRoot "configs\env\.env.example"
    if (Test-Path $hiddenTemplate) {
        return $hiddenTemplate
    }

    $visibleTemplate = Join-Path $RepoRoot "configs\env\env.example"
    if (Test-Path $visibleTemplate) {
        return $visibleTemplate
    }

    return $null
}

function New-HexSecret {
    $Bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($Bytes)
    return [BitConverter]::ToString($Bytes).Replace("-", "").ToLower()
}

function Set-EnvValue {
    param(
        [string]$Path,
        [string]$Key,
        [string]$Value
    )

    $Lines = Get-Content $Path
    $KeyPattern = "^$([regex]::Escape($Key))=.*$"

    if ($Lines -match $KeyPattern) {
        $Updated = $Lines -replace $KeyPattern, "$Key=$Value"
        $Updated | Set-Content $Path
    } else {
        Add-Content -Path $Path -Value "$Key=$Value"
    }
}

function Ensure-Secret {
    param(
        [string]$Path,
        [string]$Key,
        [string]$Placeholder = ""
    )

    $Line = Get-Content $Path | Where-Object { $_ -match "^$([regex]::Escape($Key))=" } | Select-Object -First 1
    $Value = if ($Line) { $Line.Split("=", 2)[1] } else { "" }

    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -eq $Placeholder) {
        Set-EnvValue -Path $Path -Key $Key -Value (New-HexSecret)
    }
}

# ---- prerequisites ----

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker not found. Install Docker Desktop: https://docs.docker.com/get-docker/"
    exit 1
}

try { docker compose version | Out-Null } catch {
    Write-Error "Docker Compose plugin not found."
    exit 1
}

# ---- first-run .env setup ----

if (-not (Test-Path $EnvFile)) {
    $Template = Get-EnvTemplate
    if (-not $Template) {
        Write-Error "Could not find configs/env/.env.example or configs/env/env.example."
        exit 1
    }

    Copy-Item $Template $EnvFile
    Write-Host "Created $EnvFile from $(Split-Path $Template -Leaf)."
}

# ---- ensure required secrets ----

Ensure-Secret -Path $EnvFile -Key "API_KEY" -Placeholder "your-api-key-here"
Ensure-Secret -Path $EnvFile -Key "POSTGRES_PASSWORD"
Ensure-Secret -Path $EnvFile -Key "ADMIN_API_KEY" -Placeholder "your-admin-api-key-here"
Ensure-Secret -Path $EnvFile -Key "JWT_SIGNING_KEY"
Ensure-Secret -Path $EnvFile -Key "SESSION_SECRET"

$ApiKeyLine = (Get-Content $EnvFile | Where-Object { $_ -match '^API_KEY=' } | Select-Object -First 1)
$PostgresPasswordLine = (Get-Content $EnvFile | Where-Object { $_ -match '^POSTGRES_PASSWORD=' } | Select-Object -First 1)
$AdminApiKeyLine = (Get-Content $EnvFile | Where-Object { $_ -match '^ADMIN_API_KEY=' } | Select-Object -First 1)
$JwtSigningKeyLine = (Get-Content $EnvFile | Where-Object { $_ -match '^JWT_SIGNING_KEY=' } | Select-Object -First 1)
$SessionSecretLine = (Get-Content $EnvFile | Where-Object { $_ -match '^SESSION_SECRET=' } | Select-Object -First 1)

$ApiKey = $ApiKeyLine.Split("=", 2)[1]
$env:POSTGRES_PASSWORD = $PostgresPasswordLine.Split("=", 2)[1]
$env:ADMIN_API_KEY = $AdminApiKeyLine.Split("=", 2)[1]
$env:JWT_SIGNING_KEY = $JwtSigningKeyLine.Split("=", 2)[1]
$env:SESSION_SECRET = $SessionSecretLine.Split("=", 2)[1]

$env:API_KEY = $ApiKey

Write-Host "Using environment from $EnvFile"

# ---- start ----

# DOCKER_BUILDKIT=1 enables BuildKit for plain `docker build` calls.
# BUILDKIT_INLINE_CACHE=1 is not needed here (that's for registry cache export).
# Together these ensure the `uv sync` deps layer is cached between rebuilds —
# without this, uv re-downloads all packages on every `up --build` even if
# only source files changed.
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"

Write-Host "Starting CV Platform..."
Write-Host "  Frontend : http://localhost:3000"
Write-Host "  Backend  : http://localhost:8000/docs"
Write-Host ""
docker compose --env-file $EnvFile -f $ComposeFile up --build $args
