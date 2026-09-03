param(
    [string]$RemoteUrl = 'https://github.com/Heaplyn/NeuralNet.git'
)

Write-Host "Preparing to push to $RemoteUrl"

# Ensure we're in repo root
$cwd = Get-Location

# Add remote if missing
$hasOrigin = (& git remote get-url origin) 2>$null
if (-not $hasOrigin) {
    Write-Host "Adding remote 'origin' -> $RemoteUrl"
    git remote add origin $RemoteUrl
}
else {
    Write-Host "Remote 'origin' already present: $hasOrigin"
}

# Ensure branch
git branch --show-current 2>$null | Out-Null
$branch = (& git rev-parse --abbrev-ref HEAD)
if ($branch -eq 'HEAD') {
    git checkout -b main
    $branch = 'main'
}

Write-Host "Staging changes..."
git add -A

if ((git status --porcelain) -ne '') {
    git commit -m "Repository updates: fixes and scripts"
}
else {
    Write-Host "No changes to commit."
}

Write-Host "Pushing to origin/$branch (you may be prompted for credentials)..."
try {
    git push -u origin $branch
}
catch {
    Write-Host "Push failed. Try authenticating with 'gh auth login' or set a PAT as credential."
    exit 1
}

Write-Host "Push completed."
