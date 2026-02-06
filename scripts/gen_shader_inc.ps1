# Generate MSVC-compatible .inc file from WGSL shader
# Splits into multiple string literals to avoid C2026 (max 16380 chars)

$projectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$inPath = Join-Path $projectDir "shaders\gemma3_kernels.wgsl"
$outPath = Join-Path $projectDir "shaders\gemma3_kernels.wgsl.inc"

$wgslLines = Get-Content $inPath

$output = [System.Collections.ArrayList]::new()
[void]$output.Add("// Auto-generated from gemma3_kernels.wgsl - do not edit")
[void]$output.Add("// Split into chunks for MSVC compatibility (C2026: max 16380 chars per literal)")

$charCount = 0
$needOpen = $true

foreach ($rawLine in $wgslLines) {
    $line = $rawLine.TrimEnd()
    $escaped = $line.Replace('\', '\\').Replace('"', '\"')
    $cLine = "$escaped\n\"

    if ($needOpen) {
        [void]$output.Add("`"$cLine")
        $needOpen = $false
        $charCount = $cLine.Length
    } else {
        # Check if adding this line would exceed the limit
        if (($charCount + $cLine.Length) -gt 14000) {
            # Close current string literal, empty line starts concatenation
            $lastIdx = $output.Count - 1
            $lastLine = $output[$lastIdx]
            # Remove trailing \ from last line and close quote
            if ($lastLine.EndsWith('\')) {
                $output[$lastIdx] = $lastLine.Substring(0, $lastLine.Length - 1) + '"'
            }
            # Start new string literal
            [void]$output.Add("`"$cLine")
            $charCount = $cLine.Length
        } else {
            [void]$output.Add($cLine)
            $charCount += $cLine.Length
        }
    }
}

# Close the last string literal
$lastIdx = $output.Count - 1
$lastLine = $output[$lastIdx]
if ($lastLine.EndsWith('\')) {
    $output[$lastIdx] = $lastLine.Substring(0, $lastLine.Length - 1) + '"'
}

$result = $output -join "`r`n"
[System.IO.File]::WriteAllText($outPath, $result, [System.Text.UTF8Encoding]::new($false))
Write-Output "Generated: $outPath"
Write-Output "Size: $((Get-Item $outPath).Length) bytes"
Write-Output "Chunks: $(($output | Where-Object { $_.StartsWith('`"') -and -not $_.StartsWith('//') }).Count)"
