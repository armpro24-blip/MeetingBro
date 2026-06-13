param(
    [Parameter(Mandatory = $true)][string]$Voice,
    [Parameter(Mandatory = $true)][string]$TextFile,
    [Parameter(Mandatory = $true)][string]$Out
)
# Render one text segment to a 16 kHz mono PCM WAV with the named SAPI voice.
# Text is read from a UTF-8 file to avoid CLI encoding loss (e.g. Chinese).
Add-Type -AssemblyName System.Speech
$text = Get-Content -Raw -Encoding UTF8 -Path $TextFile
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice($Voice)
$s.SetOutputToWaveFile($Out, $fmt)
$s.Speak($text)
$s.Dispose()
