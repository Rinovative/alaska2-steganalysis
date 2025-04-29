# generate_colab_requirements.ps1

# Exportiere die Abhängigkeiten aus Poetry
poetry export --without-hashes --without-urls -f requirements.txt -o raw_requirements.txt

# Bereinige die requirements für Colab
Get-Content raw_requirements.txt |
    Where-Object {
        $_ -notmatch '^(ipython|ipykernel|notebook|jupyter|matplotlib-inline|pandas==|pyarrow==|torch==|torchvision==|tensorflow==|decorator==|numpy==|numba|cudf|pylibcudf|nvidia|moviepy)'
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

Write-Output "requirements_colab.txt wurde erstellt und bereinigt."