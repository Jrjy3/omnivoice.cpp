[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BuildDirectory,

    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildRoot = (Resolve-Path $BuildDirectory).Path
$manifestPath = Join-Path $repositoryRoot "packaging\sonorus-runtime-v1.0.0.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if (-not $OutputPath) {
    $OutputPath = Join-Path $repositoryRoot (
        "dist\omnivoice-runtime-windows-x64-avx2-vulkan-v1.0.0.zip"
    )
}
$outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $outputFullPath

New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$stageRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "omnivoice-runtime-" + [Guid]::NewGuid().ToString("N")
)
New-Item -ItemType Directory -Path $stageRoot | Out-Null

try {
    foreach ($entry in $manifest.files.PSObject.Properties) {
        $source = Join-Path $buildRoot $entry.Name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Missing runtime file: $source"
        }

        $file = Get-Item -LiteralPath $source
        if ($file.Length -ne [long]$entry.Value.size) {
            throw "$($entry.Name) has size $($file.Length), expected $($entry.Value.size)"
        }

        $digest = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -ne $entry.Value.sha256) {
            throw "$($entry.Name) has SHA-256 $digest, expected $($entry.Value.sha256)"
        }

        Copy-Item -LiteralPath $source -Destination (Join-Path $stageRoot $entry.Name)
    }

    Copy-Item -LiteralPath $manifestPath -Destination (
        Join-Path $stageRoot "RUNTIME-MANIFEST.json"
    )
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "LICENSE") -Destination (
        Join-Path $stageRoot "omnivoice.cpp.LICENSE"
    )
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "ggml\LICENSE") -Destination (
        Join-Path $stageRoot "ggml.LICENSE"
    )

    Compress-Archive -LiteralPath (
        Get-ChildItem -LiteralPath $stageRoot | ForEach-Object FullName
    ) -DestinationPath $outputFullPath -CompressionLevel Optimal -Force

    $archive = Get-Item -LiteralPath $outputFullPath
    $archiveHash = (Get-FileHash -LiteralPath $outputFullPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Output "Created $outputFullPath"
    Write-Output "Size: $($archive.Length)"
    Write-Output "SHA-256: $archiveHash"
}
finally {
    if (Test-Path -LiteralPath $stageRoot) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
}
