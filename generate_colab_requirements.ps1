# generate_colab_requirements.ps1

# Exportiere die Abhängigkeiten
poetry export --without-hashes --without-urls -f requirements.txt -o raw_requirements.txt

# Filtere problematische Pakete überall in der Zeile raus
Get-Content raw_requirements.txt |
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
    } |
    ForEach-Object {
        $_ -replace '^numpy==.+$', 'numpy>=1.26,<2.1' `
           -replace '^pyarrow==.+$', 'pyarrow<20' `
           -replace '^torch==.+$', 'torch' `
           -replace '^torchvision==.+$', 'torchvision' `
           -replace '^pandas==.+$', 'pandas'
    } | Set-Content requirements_colab.txt

# Aufräumen
Remove-Item raw_requirements.txt

Write-Output "✅ Sauberes requirements_colab.txt erstellt!"