# 1. Exportiere Poetry-Abhängigkeiten für Colab (ohne URLs und ohne fsspec und andere doppelten Pakete)
$colabContent = poetry export --without-hashes --without-urls -f requirements.txt | `
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
        ($_ -notmatch 'moviepy') -and
        ($_ -notmatch 'requests') -and
        ($_ -notmatch 'seaborn') -and
        ($_ -notmatch 'scikit-learn') -and
        ($_ -notmatch 'plotly') -and
        ($_ -notmatch 'debugpy') -and
        ($_ -notmatch 'dill') -and
        ($_ -notmatch 'cffi') -and
        ($_ -notmatch 'pycparser') -and
        ($_ -notmatch 'pip') -and
        ($_ -notmatch 'setuptools') -and
        ($_ -notmatch 'packaging') -and
        ($_ -notmatch 'beautifulsoup4') -and
        ($_ -notmatch 'colorama') -and
        ($_ -notmatch 'pyyaml') -and
        ($_ -notmatch 'pillow') -and
        ($_ -notmatch 'plotly') -and
        ($_ -notmatch 'joblib') -and
        ($_ -notmatch 'tqdm') -and
        ($_ -notmatch 'aiohttp') -and
        ($_ -notmatch 'httpx') -and
        ($_ -notmatch 'pandas') -and
        ($_ -notmatch 'matplotlib') -and
        ($_ -notmatch 'lightning-utilities') -and
        ($_ -notmatch 'psutil') -and
        ($_ -notmatch 'networkx') -and
        ($_ -notmatch 'scipy') -and
        ($_ -notmatch 'seaborn') -and
        ($_ -notmatch 'jpegio') -and
        ($_ -notmatch 'jpegio') -and
        ($_ -notmatch 'attrs') -and
        ($_ -notmatch 'babel') -and
        ($_ -notmatch 'charset-normalizer') -and
        ($_ -notmatch 'comm') -and
        ($_ -notmatch 'defusedxml') -and
        ($_ -notmatch 'executing') -and
        ($_ -notmatch 'filelock') -and
        ($_ -notmatch 'fonttools') -and
        ($_ -notmatch 'frozenlist') -and
        ($_ -notmatch 'kiwisolver') -and
        ($_ -notmatch 'lazy-loader') -and
        ($_ -notmatch 'markupsafe') -and
        ($_ -notmatch 'mistune') -and
        ($_ -notmatch 'mpmath') -and
        ($_ -notmatch 'multiprocess') -and
        ($_ -notmatch 'parso') -and
        ($_ -notmatch 'pexpect') -and
        ($_ -notmatch 'platformdirs') -and
        ($_ -notmatch 'prometheus-client') -and
        ($_ -notmatch 'prompt-toolkit') -and
        ($_ -notmatch 'propcache') -and
        ($_ -notmatch 'psutil') -and
        ($_ -notmatch 'ptyprocess') -and
        ($_ -notmatch 'pure-eval') -and
        ($_ -notmatch 'pygments') -and
        ($_ -notmatch 'pyparsing') -and
        ($_ -notmatch 'python-dateutil') -and
        ($_ -notmatch 'python-json-logger') -and
        ($_ -notmatch 'pytz') -and
        ($_ -notmatch 'pyzmq') -and
        ($_ -notmatch 'referencing') -and
        ($_ -notmatch 'regex') -and
        ($_ -notmatch 'rfc3339-validator') -and
        ($_ -notmatch 'rfc3986-validator') -and
        ($_ -notmatch 'rpds-py') -and
        ($_ -notmatch 'safetensors') -and
        ($_ -notmatch 'scikit-image') -and
        ($_ -notmatch 'scipy') -and
        ($_ -notmatch 'send2trash') -and
        ($_ -notmatch 'six') -and
        ($_ -notmatch 'sniffio') -and
        ($_ -notmatch 'soupsieve') -and
        ($_ -notmatch 'stack-data') -and
        ($_ -notmatch 'sympy') -and
        ($_ -notmatch 'terminado') -and
        ($_ -notmatch 'threadpoolctl') -and
        ($_ -notmatch 'tifffile') -and
        ($_ -notmatch 'timm') -and
        ($_ -notmatch 'tinycss2') -and
        ($_ -notmatch 'tornado') -and
        ($_ -notmatch 'traitlets') -and
        ($_ -notmatch 'types-python-dateutil') -and
        ($_ -notmatch 'tzdata') -and
        ($_ -notmatch 'uri-template') -and
        ($_ -notmatch 'urllib3') -and
        ($_ -notmatch 'wcwidth') -and
        ($_ -notmatch 'webcolors') -and
        ($_ -notmatch 'webencodings') -and
        ($_ -notmatch 'websocket-client') -and
        ($_ -notmatch 'widgetsnbextension') -and
        ($_ -notmatch 'xxhash') -and
        ($_ -notmatch 'yarl') -and
        ($_ -notmatch 'fsspec') -and
        ($_ -notmatch 'triton')
    }

# 2. Feste Basis-Abhängigkeiten definieren
$baseDeps = @"
# Definiere nur die wirklich benötigten Pakete
clip-anytorch==2.6.0 ; python_version >= "3.11" and python_version < "4.0"
fsspec==2025.3.2 ; python_version >= "3.11" and python_version < "4.0"
jpegio==0.2.8 ; python_version >= "3.11" and python_version < "4.0"

# Fixierte Versionen für torch und torchaudio, um Konflikte zu vermeiden
torch==2.6.0 ; python_version >= "3.11" and python_version < "4.0"
torchmetrics==1.7.3 ; python_version >= "3.11" and python_version < "4.0"
torchinfo==1.8.0 ; python_version >= "3.11" and python_version < "4.0"
"@ -split "`n"

# 3. Alles zusammen in requirements_colab.txt schreiben
$baseDeps + $colabContent | Set-Content requirements_colab.txt

# 4. Fertigmeldung
Write-Output "requirements_colab.txt wurde erstellt!"
