# generate_colab_requirements.ps1

# 1. Exportiere alle Poetry-Abhängigkeiten
poetry export --without-hashes --without-urls -f requirements.txt -o raw_requirements.txt

# 2. Projektabhängigkeiten bereinigen (lösche problematische Kernpakete)
$projectDeps = Get-Content raw_requirements.txt |
    Where-Object {
        ($_ -notmatch 'ipython') -and
        ($_ -notmatch 'ipykernel') -and
        ($_ -notmatch 'notebook') -and
        ($_ -notmatch 'jupyter') -and
        ($_ -notmatch 'matplotlib-inline') -and
        ($_ -notmatch 'pandas==') -and
        ($_ -notmatch 'pyarrow==') -and
        ($_ -notmatch 'torch==') -and
        ($_ -notmatch 'torchvision==') -and
        ($_ -notmatch 'tensorflow==') -and
        ($_ -notmatch 'decorator==') -and
        ($_ -notmatch 'numpy==') -and
        ($_ -notmatch 'numba') -and
        ($_ -notmatch 'cudf') -and
        ($_ -notmatch 'pylibcudf') -and
        ($_ -notmatch 'nvidia') -and
        ($_ -notmatch 'moviepy')
    }

# 3. Feste Basis-Abhängigkeiten definieren
$baseDeps = @"
ipykernel==6.17.1
ipython==7.34.0
notebook==6.5.7
pandas==2.2.2
numpy>=1.26,<2.1
pyarrow<20
torch
torchvision
decorator<5.0
"@ -split "`n"

# 4. Alles zusammen in requirements_colab.txt schreiben
$baseDeps + $projectDeps | Set-Content requirements_colab.txt

# 5. Aufräumen
Remove-Item raw_requirements.txt

# 6. Fertigmeldung
Write-Output "requirements_colab.txt wurde erstellt!"