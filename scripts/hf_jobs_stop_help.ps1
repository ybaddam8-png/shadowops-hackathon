# Hugging Face Jobs stop/cancel help. Do not leave GPU jobs running.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Use hf jobs ps -a to find the job id."
Write-Host "Use hf jobs cancel <JOB_ID>, or run hf jobs --help and use the current stop/cancel command shown by your CLI version."
Write-Host "Do not leave GPU jobs running. Cancel stuck jobs immediately to protect credits."
