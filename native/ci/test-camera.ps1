# End-to-end test of the VirtualCam camera on a Windows machine.
# 1. register the camera  2. send frames  3. capture through DirectShow with ffmpeg  4. check the colour.
param(
  [Parameter(Mandatory = $true)][string]$Bin,     # folder with x64\softcam.dll, x64\vcam-bridge.exe, x86\softcam.dll
  [Parameter(Mandatory = $true)][string]$LogFile
)

$ErrorActionPreference = 'Stop'
$failures = 0
$clsid = '{9CF79DDE-DA6F-46F4-886A-4ED6ED6A99A4}'
$width = 1280
$height = 720

function Log($text) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $text
  Write-Host $line
  Add-Content -Path $LogFile -Value $line
}
function Fail($text) { $script:failures++; Log "FAIL: $text" }
function Pass($text) { Log "PASS: $text" }

# Captures one frame from the VirtualCam device and returns the RGB of its centre pixel.
function Capture-Centre($name) {
  $raw = Join-Path $env:TEMP "$name.raw"
  Remove-Item $raw -ErrorAction SilentlyContinue
  $out = & ffmpeg -hide_banner -loglevel error -f dshow -i 'video=VirtualCam' -frames:v 1 -pix_fmt rgb24 -s "${width}x${height}" -f rawvideo -y $raw 2>&1
  if ($out) { Log "ffmpeg: $out" }
  if (-not (Test-Path $raw)) { return $null }
  $bytes = [System.IO.File]::ReadAllBytes($raw)
  $i = (($height / 2) * $width + ($width / 2)) * 3
  if ($bytes.Length -lt $i + 3) { return $null }
  return @($bytes[$i], $bytes[$i + 1], $bytes[$i + 2])
}

function Near($rgb, $r, $g, $b) {
  if (-not $rgb) { return $false }
  return ([math]::Abs($rgb[0] - $r) -le 12) -and ([math]::Abs($rgb[1] - $g) -le 12) -and ([math]::Abs($rgb[2] - $b) -le 12)
}

Set-Content -Path $LogFile -Value "VirtualCam Windows camera test - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Log ("Windows: " + (Get-CimInstance Win32_OperatingSystem).Caption + " build " + [Environment]::OSVersion.Version.Build)

# --- dependencies of the built files (so we know what users need installed) ---
foreach ($f in @("$Bin\x64\softcam.dll", "$Bin\x64\vcam-bridge.exe", "$Bin\x86\softcam.dll")) {
  $deps = (& dumpbin /nologo /dependents $f) -match '\.dll' | ForEach-Object { $_.Trim() }
  Log ("deps of " + (Split-Path $f -Leaf) + " (" + (Split-Path (Split-Path $f) -Leaf) + "): " + ($deps -join ', '))
}

# --- 1. register 64-bit and 32-bit ---
$p = Start-Process -FilePath "$env:WINDIR\System32\regsvr32.exe" -ArgumentList '/s', "`"$Bin\x64\softcam.dll`"" -Wait -PassThru
if ($p.ExitCode -eq 0) { Pass "64-bit camera registered" } else { Fail "64-bit regsvr32 exit code $($p.ExitCode)" }
$p = Start-Process -FilePath "$env:WINDIR\SysWOW64\regsvr32.exe" -ArgumentList '/s', "`"$Bin\x86\softcam.dll`"" -Wait -PassThru
if ($p.ExitCode -eq 0) { Pass "32-bit camera registered" } else { Fail "32-bit regsvr32 exit code $($p.ExitCode)" }

if (Test-Path "Registry::HKEY_CLASSES_ROOT\CLSID\$clsid\InprocServer32") { Pass "64-bit class registered" } else { Fail "64-bit class missing in registry" }
if (Test-Path "Registry::HKEY_CLASSES_ROOT\WOW6432Node\CLSID\$clsid\InprocServer32") { Pass "32-bit class registered" } else { Fail "32-bit class missing in registry" }

# --- 2. listed as a camera ---
$list = (& ffmpeg -hide_banner -list_devices true -f dshow -i dummy 2>&1) -join "`n"
if ($list -match '"VirtualCam"') { Pass "ffmpeg lists the VirtualCam camera" } else { Fail "VirtualCam not in DirectShow device list"; Log $list }

# --- 3. test-pattern mode ---
$bridge = Start-Process -FilePath "$Bin\x64\vcam-bridge.exe" -ArgumentList "$width $height --test-pattern 30 200 10 20" -PassThru -NoNewWindow -RedirectStandardOutput "$env:TEMP\bridge1.txt"
Start-Sleep -Seconds 2
$rgb = Capture-Centre 'pattern'
Log ("pattern frame centre: " + ($rgb -join ','))
if (Near $rgb 30 200 10) { Pass "camera shows the test colour (30,200,10)" } else { Fail "wrong or no picture from test pattern" }
Stop-Process -Id $bridge.Id -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
Log ("bridge said: " + ((Get-Content "$env:TEMP\bridge1.txt" -ErrorAction SilentlyContinue) -join ' | '))

# --- 4. stdin mode, the way the app feeds it ---
$feeder = @"
const { spawn } = require('child_process');
const W = $width, H = $height;
const frame = Buffer.alloc(W * H * 3);
for (let i = 0; i < frame.length; i += 3) { frame[i] = 10; frame[i + 1] = 30; frame[i + 2] = 200; } // BGR -> RGB (200,30,10)
const p = spawn(process.argv[2], [String(W), String(H)], { stdio: ['pipe', 'inherit', 'inherit'] });
const end = Date.now() + 15000;
const tick = () => { if (Date.now() > end) { p.stdin.end(); return; } p.stdin.write(frame, () => setTimeout(tick, 33)); };
tick();
"@
Set-Content -Path "$env:TEMP\feeder.js" -Value $feeder
$feed = Start-Process -FilePath 'node' -ArgumentList "`"$env:TEMP\feeder.js`" `"$Bin\x64\vcam-bridge.exe`"" -PassThru -NoNewWindow -RedirectStandardOutput "$env:TEMP\bridge2.txt"
Start-Sleep -Seconds 3
$rgb = Capture-Centre 'stdin'
Log ("stdin frame centre: " + ($rgb -join ','))
if (Near $rgb 200 30 10) { Pass "frames piped from the app reach the camera (200,30,10)" } else { Fail "wrong or no picture from piped frames" }
$feed | Wait-Process -Timeout 30 -ErrorAction SilentlyContinue
Log ("bridge said: " + ((Get-Content "$env:TEMP\bridge2.txt" -ErrorAction SilentlyContinue) -join ' | '))

# --- 5. unregister cleanly ---
$p = Start-Process -FilePath "$env:WINDIR\System32\regsvr32.exe" -ArgumentList '/s', '/u', "`"$Bin\x64\softcam.dll`"" -Wait -PassThru
if ($p.ExitCode -eq 0) { Pass "64-bit camera unregistered" } else { Fail "unregister exit code $($p.ExitCode)" }

if ($failures -eq 0) { Log "ALL TESTS PASSED" } else { Log "$failures TEST(S) FAILED" }
exit $failures
