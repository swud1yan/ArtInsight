param(
  [int[]]$Seeds = @(42, 2024, 2025, 3407, 8888),
  [int]$MaxImagesPerStyle = 800,
  [int]$Components = 12,
  [int]$Neighbors = 5,
  [int]$Folds = 5,
  [int]$PcaRepeats = 10,
  [int]$ProgressInterval = 25,
  [int]$HeartbeatSeconds = 15
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path "$PSScriptRoot\..")

$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "overnight_experiments_$stamp.log"
$env:OPENCV_LOG_LEVEL = "ERROR"
$env:PYTHONUNBUFFERED = "1"

function Write-Step {
  param([string]$Message)
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
  Write-Host $line
  Add-Content -Path $logPath -Value $line
}

function ConvertTo-CommandLineArgument {
  param([string]$Value)
  if ($Value -match '[\s"]') {
    return '"' + ($Value -replace '"', '\"') + '"'
  }
  return $Value
}

function Invoke-LoggedProcess {
  param(
    [string]$StepName,
    [int]$StepNumber,
    [int]$TotalSteps,
    [string]$FilePath,
    [string[]]$Arguments
  )
  $percent = [Math]::Round((($StepNumber - 1) / [Math]::Max($TotalSteps, 1)) * 100, 1)
  Write-Progress -Activity "ArtInsight overnight experiments" -Status "[$StepNumber/$TotalSteps] $StepName" -PercentComplete $percent
  Write-Step "START [$StepNumber/$TotalSteps] $StepName"
  Write-Step ("COMMAND {0} {1}" -f $FilePath, (($Arguments | ForEach-Object { ConvertTo-CommandLineArgument $_ }) -join " "))
  $processInfo = New-Object System.Diagnostics.ProcessStartInfo
  $processInfo.FileName = $FilePath
  $processInfo.Arguments = ($Arguments | ForEach-Object { ConvertTo-CommandLineArgument $_ }) -join " "
  $processInfo.UseShellExecute = $false
  $processInfo.RedirectStandardOutput = $true
  $processInfo.RedirectStandardError = $true
  $processInfo.CreateNoWindow = $true

  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $processInfo

  $outputHandler = [System.Diagnostics.DataReceivedEventHandler]{
    param($sender, $eventArgs)
    if ($eventArgs.Data) {
      Write-Host $eventArgs.Data
      Add-Content -Path $logPath -Value $eventArgs.Data
    }
  }
  $errorHandler = [System.Diagnostics.DataReceivedEventHandler]{
    param($sender, $eventArgs)
    if ($eventArgs.Data) {
      $line = "[stderr] " + $eventArgs.Data
      if ($eventArgs.Data -notlike "libpng warning: iCCP:*") {
        Write-Host $line -ForegroundColor DarkYellow
      }
      Add-Content -Path $logPath -Value $line
    }
  }

  $process.add_OutputDataReceived($outputHandler)
  $process.add_ErrorDataReceived($errorHandler)
  [void]$process.Start()
  $process.BeginOutputReadLine()
  $process.BeginErrorReadLine()

  $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
  $nextHeartbeat = [TimeSpan]::FromSeconds([Math]::Max($HeartbeatSeconds, 1))
  while (-not $process.WaitForExit(1000)) {
    if ($stopwatch.Elapsed -ge $nextHeartbeat) {
      Write-Step ("RUNNING [{0}/{1}] {2} elapsed={3}" -f $StepNumber, $TotalSteps, $StepName, $stopwatch.Elapsed.ToString())
      $nextHeartbeat = $stopwatch.Elapsed + [TimeSpan]::FromSeconds([Math]::Max($HeartbeatSeconds, 1))
    }
  }
  $process.WaitForExit()
  $stopwatch.Stop()

  if ($process.ExitCode -ne 0) {
    throw "Command failed with exit code $($process.ExitCode) during $StepName"
  }
  Write-Step ("DONE [{0}/{1}] {2} elapsed={3}" -f $StepNumber, $TotalSteps, $StepName, $stopwatch.Elapsed.ToString())
  $donePercent = [Math]::Round(($StepNumber / [Math]::Max($TotalSteps, 1)) * 100, 1)
  Write-Progress -Activity "ArtInsight overnight experiments" -Status "Completed $StepName" -PercentComplete $donePercent
}

function Skip-Step {
  param(
    [string]$StepName,
    [int]$StepNumber,
    [int]$TotalSteps,
    [string]$Reason
  )
  $donePercent = [Math]::Round(($StepNumber / [Math]::Max($TotalSteps, 1)) * 100, 1)
  Write-Progress -Activity "ArtInsight overnight experiments" -Status "Skipped [$StepNumber/$TotalSteps] $StepName" -PercentComplete $donePercent
  Write-Step "SKIP [$StepNumber/$TotalSteps] $StepName - $Reason"
}

function Test-ReportComplete {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    return $false
  }
  $item = Get-Item -LiteralPath $Path
  if ($item.Length -lt 500) {
    return $false
  }
  return [bool](Select-String -LiteralPath $Path -Pattern "PCA and Feature Comparison Experiment" -Quiet)
}

function Test-SampledDatasetComplete {
  param([int]$Seed)
  $courseReport = "data/course_dataset_seed_$Seed/sample_report.txt"
  $paperReport = "data/paper_dataset_seed_$Seed/sample_report.txt"
  if (-not (Test-Path -LiteralPath $courseReport)) {
    return $false
  }
  if (-not (Test-Path -LiteralPath $paperReport)) {
    return $false
  }
  $coursePass = Select-String -LiteralPath $courseReport -Pattern "Verification status: PASS" -Quiet
  $paperPass = Select-String -LiteralPath $paperReport -Pattern "Verification status: PASS" -Quiet
  return [bool]($coursePass -and $paperPass)
}

Write-Step "ArtInsight overnight experiments"
Write-Step "Seeds: $($Seeds -join ', ')"
Write-Step "MaxImagesPerStyle: $MaxImagesPerStyle"
Write-Step "Components: $Components; Neighbors: $Neighbors; Folds: $Folds; PCA repeats: $PcaRepeats"
Write-Step "Progress interval: every $ProgressInterval images"

$totalSteps = 1 + ($Seeds.Count * 2)
$stepNumber = 1
$baselineReport = "docs/pca_feature_experiment_report_paper_dataset.md"

if (Test-ReportComplete $baselineReport) {
  Skip-Step "baseline paper_dataset full PCA feature comparison" $stepNumber $totalSteps "existing report found: $baselineReport"
} else {
  Invoke-LoggedProcess "baseline paper_dataset full PCA feature comparison" $stepNumber $totalSteps "python" @(
    "-m", "experiments.pca_feature_comparison",
    "--dataset-root", "D:/PCA/data/paper_dataset",
    "--dataset-format", "wikiart",
    "--max-images-per-style", "$MaxImagesPerStyle",
    "--min-classes", "6",
    "--min-images-per-class", "$MaxImagesPerStyle",
    "--components", "$Components",
    "--neighbors", "$Neighbors",
    "--folds", "$Folds",
    "--pca-repeats", "$PcaRepeats",
    "--progress-interval", "$ProgressInterval",
    "--seed", "42",
    "--output", $baselineReport
  )
}
$stepNumber += 1

foreach ($seed in $Seeds) {
  if (Test-SampledDatasetComplete $seed) {
    Skip-Step "sample datasets seed=$seed" $stepNumber $totalSteps "existing PASS sample reports found"
  } else {
    Invoke-LoggedProcess "sample datasets seed=$seed" $stepNumber $totalSteps "python" @(
      "-m", "experiments.dataset_sampler",
      "--source-root", "D:/PCA/data/raw/wikiart",
      "--course-root", "D:/PCA/data/course_dataset_seed_$seed",
      "--paper-root", "D:/PCA/data/paper_dataset_seed_$seed",
      "--overview-csv", "D:/PCA/data/dataset_overview_seed_$seed.csv",
      "--progress-interval", "$ProgressInterval",
      "--seed", "$seed"
    )
  }
  $stepNumber += 1

  $seedReport = "docs/pca_feature_experiment_report_seed_$seed.md"
  if (Test-ReportComplete $seedReport) {
    Skip-Step "PCA feature comparison seed=$seed" $stepNumber $totalSteps "existing report found: $seedReport"
  } else {
    Invoke-LoggedProcess "PCA feature comparison seed=$seed" $stepNumber $totalSteps "python" @(
      "-m", "experiments.pca_feature_comparison",
      "--dataset-root", "D:/PCA/data/paper_dataset_seed_$seed",
      "--dataset-format", "wikiart",
      "--max-images-per-style", "$MaxImagesPerStyle",
      "--min-classes", "6",
      "--min-images-per-class", "$MaxImagesPerStyle",
      "--components", "$Components",
      "--neighbors", "$Neighbors",
      "--folds", "$Folds",
      "--pca-repeats", "$PcaRepeats",
      "--progress-interval", "$ProgressInterval",
      "--seed", "$seed",
      "--output", $seedReport
    )
  }
  $stepNumber += 1
}

Write-Progress -Activity "ArtInsight overnight experiments" -Completed
Write-Step "ALL DONE"
Write-Step "Baseline report: docs/pca_feature_experiment_report_paper_dataset.md"
foreach ($seed in $Seeds) {
  Write-Step "Seed report: docs/pca_feature_experiment_report_seed_$seed.md"
}
Write-Step "Log: $logPath"
