# setup_briefing_scheduler.ps1
# 設定 Windows Task Scheduler 每天早上 8:30 產生盤前快訊

$TaskName = "StockPremarketBriefing"
$ScriptPath = "C:\Users\soye6\.openclaw\workspace\tw-stock-analyzer\professional\auto_briefing.py"
$PythonPath = "C:\Users\soye6\AppData\Local\Programs\Python\Python313\python.exe"
$WorkDir = "C:\Users\soye6\.openclaw\workspace\tw-stock-analyzer"
$LogFile = "$WorkDir\memory\briefing\scheduler_log.txt"

# 先刪除舊任務
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# 動作：執行 Python 腳本，輸出紀錄到 log
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$ScriptPath`" premarket >> `"$LogFile`" 2>&1" -WorkingDirectory $WorkDir

# 觸發：每日 8:30 AM
$Trigger = New-ScheduledTaskTrigger -Daily -At "08:30AM"

# 設定
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -MultipleInstances IgnoreNew

# 註冊
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -RunLevel Limited -Force

Write-Host "✅ Task '$TaskName' registered!"
Write-Host "Schedule: Daily 8:30 AM"
Write-Host "Script: $ScriptPath"

# 檢查狀態
$t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($t) {
    Write-Host "State: $($t.State)"
    Write-Host "Triggers:"
    $t.Triggers | ForEach-Object { Write-Host "  $($_.StartBoundary) (enabled: $($_.Enabled))" }
} else {
    Write-Host "⚠️  Task not found - may need admin rights"
}
