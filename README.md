[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rinovative/alaska2-steganalysis/blob/main/ANN_Projekt_Rino_Albertin_Steganalyse.ipynb)  
_Das vollständige Jupyter Notebook direkt im Browser mit Google Colab öffnen_

# Deep Learning für Steganalyse mit ALASKA2  
### *Wahlfachprojekt Applied Neural Networks – BSc Systemtechnik, Frühjahr 2025*  

Bachelor of Science in Systemtechnik – Vertiefung Computational Engineering  
OST – Ostschweizer Fachhochschule  
**Autor:** Rino M. Albertin  

## 📌 Projektübersicht

Dieses Projekt untersucht die Erkennung steganographischer Manipulationen in JPEG-Bildern mittels Deep Learning. Als primäre Datengrundlage dient der ALASKA2-Datensatz, ein Benchmark für moderne Verfahren der Bildsteganalyse.

Die untersuchten Stego-Verfahren verändern keine unmittelbar sichtbaren Bildinhalte. Stattdessen werden ausgewählte JPEG-DCT-Koeffizienten so modifiziert, dass Informationen im Frequenzraum verborgen werden können. Die Analyse kombiniert deshalb visuelle Bildmerkmale, YCbCr-Farbkanäle, JPEG-Metadaten und DCT-basierte Eigenschaften.

Die zentrale Forschungsfrage lautet:

> Wie zuverlässig lassen sich subtile steganographische Veränderungen in JPEG-Bildern mit einer kompakten TinyCNN-Architektur und einem vortrainierten EfficientNet-B0 unter identischen Datenbedingungen erkennen?

<details>
<summary><strong>🎯 Untersuchungsdesign und Umfang</strong></summary>

Das Projekt vergleicht zwei binäre Klassifikationsmodelle:

1. **TinyCNN**
   Eine kompakte Referenzarchitektur, die ausschliesslich den Y-Kanal verarbeitet und mit blockausgerichteten Bildausschnitten sowie Tile-Shuffling trainiert wird.

2. **EfficientNet-B0**
   Ein ImageNet-vortrainiertes Modell mit angepasstem YCbCr-Eingang und stufenweisem Fine-Tuning einzelner Feature-Blöcke.

Beide Modelle verwendeten dieselben quellenweise gruppierten Daten:

- 6'000 Trainingsgruppen mit 24'000 Bildern;
- 750 Validierungsgruppen mit 3'000 Bildern;
- 7'500 finale Testgruppen mit 30'000 Bildern;
- Seed 42 für die reproduzierbare Gruppierung und Auswahl.

Eine Quellgruppe umfasst jeweils das unveränderte Cover-Bild und die drei zugehörigen Stego-Varianten JMiPOD, JUNIWARD und UERD. Alle Varianten einer Quelle werden stets gemeinsam einem Split zugeordnet. Dadurch kann derselbe Bildinhalt nicht gleichzeitig in Training, Validierung und Test auftreten.

Die Modellauswahl erfolgt anhand der Weighted AUC auf dem Validierungssplit. Der gemeinsame Testsplit wird erst nach abgeschlossener Modellauswahl genau einmal pro Modell ausgewertet.

Da der originale ALASKA2-Datensatz nicht frei weitergegeben werden darf, unterstützt das Projekt zusätzlich einen öffentlichen synthetischen Ersatzdatensatz auf Basis von PD12M. ALASKA2 und der Ersatzdatensatz bleiben vollständig getrennt und werden nicht miteinander vermischt.

**Hinweis zum synthetischen Ersatzdatensatz:** Die im Projektworkflow als JMiPOD bezeichnete Kompatibilitätsklasse wird technisch mit nsF5 erzeugt. Sie ist deshalb wissenschaftlich nicht mit echtem ALASKA2-JMiPOD gleichzusetzen. Das synthetische Demo-Subset enthält keine echten Nachrichten, sondern simuliert typische Frequenzmodifikationen von Steganographieverfahren.

</details>

## 🧱 Projektkomponenten

Das Repository ist in folgende fachliche Komponenten gegliedert:

<details>
<summary><strong>🗂️ Datengrundlage und synthetischer Ersatzdatensatz</strong></summary>

Der originale Datensatz wird unter `data/ALASKA2/` erwartet und umfasst vier Verzeichnisse:

- `Cover`
- `JMiPOD`
- `JUNIWARD`
- `UERD`

Zusammengehörige Dateien besitzen in allen vier Klassen denselben Basisnamen. Die Pipeline indexiert ausschliesslich vollständige Quellgruppen.

Wenn ALASKA2 nicht verfügbar ist, kann unter `data/PD12M/` ein öffentlicher synthetischer Ersatzdatensatz vorbereitet werden. Dieser enthält dieselbe Verzeichnisstruktur, wird jedoch vollständig getrennt von ALASKA2 verarbeitet.

Die Datenquellenauswahl übergibt stets genau einen Dataset-Root an die Indexierung, die Split-Erzeugung und die DataLoader.

</details>

<details>
<summary><strong>🔬 Explorative Datenanalyse</strong></summary>

Die explorative Datenanalyse untersucht die Bilder auf mehreren Darstellungsebenen:

- Klassen- und Datensatzstruktur
- visuelle Cover- und Stego-Beispiele
- Verteilungen der RGB- und YCbCr-Kanäle
- lokale Unterschiede zwischen Cover- und Stego-Bildern
- JPEG-Qualität und Quantisierungstabellen
- DCT-Koeffizienten und methodenspezifische Flip-Muster

Für ALASKA2 verwendet die EDA eine separate, deterministische Stichprobe von 7'500 vollständigen Quellgruppen. Diese Stichprobe beeinflusst weder die Trainings- noch die Testmitgliedschaften.

</details>

<details>
<summary><strong>🧭 Datenaufbereitung und gruppierte Splits</strong></summary>

Die Datenaufbereitung indexiert zunächst alle 75'000 vollständigen ALASKA2-Quellgruppen mit insgesamt 300'000 Bildern.

Anschliessend werden mit Seed 42 drei disjunkte Reservoirs erzeugt:

- 60'000 Trainingsgruppen
- 7'500 Validierungsgruppen
- 7'500 finale Testgruppen

Aus dem Trainingsreservoir werden 6'000 Gruppen für das dokumentierte Training ausgewählt. Aus dem Validierungsreservoir werden 750 Gruppen für die Modellauswahl verwendet. Der finale Testsplit bleibt vollständig erhalten.

Die Gruppierung nach Quellidentität verhindert Information Leakage zwischen inhaltlich zusammengehörigen Cover- und Stego-Bildern.

</details>

<details>
<summary><strong>🧠 TinyCNN und EfficientNet-B0</strong></summary>

TinyCNN dient als kompakte Baseline. Das Modell verarbeitet den Y-Kanal und verwendet blockausgerichtete Bildausschnitte mit 256 × 256 Pixeln sowie Tile-Shuffling.

EfficientNet-B0 verwendet offizielle ImageNet-Gewichte und einen angepassten YCbCr-Eingang. Die Normalisierungsstatistiken werden ausschliesslich aus dem Trainingssplit bestimmt. Das nicht kumulative Fine-Tuning beginnt mit der Stufe `head_stem` und durchläuft anschliessend die Feature-Blöcke von `feature_8` bis `feature_1`. Nach jeder Stufe wird der beste Zustand anhand der Validierungs-Weighted-AUC bestimmt. Die nächste Stufe beginnt jeweils mit dem global besten Validierungszustand.

</details>

<details>
<summary><strong>⚙️ Trainingsframework</strong></summary>

Das Trainingsframework stellt für beide Modelle gemeinsame Abläufe bereit:

- getrennte Trainings-, Validierungs- und Testloader
- reproduzierbare Seeds
- eine klassengewichtete binäre Verlustfunktion (`BCEWithLogitsLoss` mit `pos_weight = 1/3`), welche das Verhältnis der drei Stego-Varianten zur Cover-Klasse ausgleicht
- Auswahl anhand der Validierungs-Weighted-AUC
- Early Stopping
- Speicherung und Wiederherstellung des besten Modellzustands
- genau eine finale Testevaluation nach abgeschlossener Modellauswahl
- typsichere Pfade für Checkpoints und Trainingsverläufe

Training und Evaluation bleiben strikt getrennt. Der Testsplit wird nicht für Hyperparameterwahl, Early Stopping oder Auswahl der Fine-Tuning-Stufe verwendet.

</details>

<details>
<summary><strong>📋 Evaluation und Ergebnisdarstellung</strong></summary>

Die Evaluation verwendet die offizielle Weighted-AUC-Metrik der ALASKA2 Steganalysis Challenge.

Für jedes ausgewählte Modell werden folgende Ansichten bereitgestellt:

- Trainings- und Validierungsverlauf
- Konfusionsmatrix
- ROC-Kurve
- Score-Verteilung
- numerische Testmetriken
- datenidentischer Modellvergleich

Das interaktive Evaluationswidget rekonstruiert die Ansichten aus kleinen, versionierten CSV- und JSON-Dateien. Checkpoints und vollständige Testvorhersagen werden für die Darstellung nicht benötigt.

TinyCNN und EfficientNet-B0 werden auf denselben Trainings-, Validierungs- und Testmitgliedschaften verglichen.

</details>

## 📊 Ergebnisse

### 🔬 Explorative Erkenntnisse

Cover- und Stego-Bilder sind visuell kaum unterscheidbar. Im Pixelraum werden Unterschiede erst durch feine kanalweise Verteilungsverschiebungen sichtbar.

Die DCT-Analyse zeigt dagegen methodenspezifische Flip-Muster. Die Änderungen konzentrieren sich überwiegend auf den Y-Kanal. JUNIWARD berücksichtigt zusätzlich texturreiche und chromatische Bildbereiche. Die JPEG-Quantisierungstabellen bleiben durch die untersuchten Stego-Verfahren unverändert.

### 📈 Quantitative Ergebnisse

Für den datenidentischen 10-%-Vergleich wurden beide Modelle auf denselben 6'000 von 60'000 Trainingsgruppen trainiert und anhand derselben 750 von 7'500 Validierungsgruppen ausgewählt. Die abschliessende Evaluation erfolgte auf dem vollständigen gemeinsamen Testsplit mit 7'500 Gruppen und 30'000 Bildern.

| Modell | Weighted AUC (Validierung) | Weighted AUC (Test) | Laufzeit |
|---|---:|---:|---:|
| TinyCNN | 0.589254 | **0.589367** | 27 min 55 s |
| EfficientNet-B0 | **0.589604** | 0.586146 | 54 min 02 s |

EfficientNet-B0 erreichte auf der Validierung den geringfügig höheren Wert. Dieser Vorteil übertrug sich jedoch nicht auf den Testsplit: Dort lag TinyCNN um 0.003221 Weighted-AUC-Punkte vorne und benötigte annähernd nur die Hälfte der Laufzeit. Aufgrund des einzelnen Versuchslaufs lässt sich daraus keine statistisch gesicherte Überlegenheit einer Architektur ableiten.

Die zentrale Forschungsfrage ist für die untersuchte Konfiguration klar zu beantworten: Mit TinyCNN und EfficientNet-B0 wurde keine zuverlässige Erkennung der subtilen steganographischen Veränderungen erreicht. Beide Testwerte liegen nahe am Referenzwert von rund 0.585714 für eine diagonale ROC-Kurve unter der verwendeten ALASKA2-Gewichtung.

Eine wesentliche Einschränkung ist der reduzierte Trainingsumfang von 6'000 Gruppen beziehungsweise 10 % des Trainingsreservoirs. Die geringere Bildvielfalt könnte die Generalisierung eingeschränkt haben; der Einfluss der Datenmenge wurde jedoch nicht isoliert untersucht.

Die vollständige Methodik, modellbezogene Interpretation und interaktive Evaluation befinden sich in [Kapitel 5 des akademischen Notebooks](ANN_Projekt_Rino_Albertin_Steganalyse.ipynb).

## ⚙️ Lokale Ausführung

<details>
<summary><strong>VS Code Dev Container mit NVIDIA-GPU</strong></summary>

Vorausgesetzt werden:

- Docker
- Visual Studio Code mit der Erweiterung Dev Containers
- eine NVIDIA-GPU
- ein funktionsfähiges NVIDIA Container Toolkit

Das Repository wird geklont und in Visual Studio Code geöffnet:

```bash
git clone https://github.com/Rinovative/alaska2-steganalysis.git
cd alaska2-steganalysis
code .
```

Anschliessend wird in Visual Studio Code über `F1` der Befehl `Dev Containers: Reopen in Container` ausgeführt.

Im Container verwenden Terminal, Poetry, Pylance und Jupyter gemeinsam:

```text
/opt/conda/bin/python
```

Poetry erstellt oder verwendet im Container keine separate virtuelle Umgebung.

Das Notebook kann danach direkt im VS-Code-Explorer geöffnet werden. Als Kernel muss `/opt/conda/bin/python` ausgewählt werden.

</details>

<details>
<summary><strong>Datensätze ablegen und prüfen</strong></summary>

### Originaler ALASKA2-Datensatz

Die vier aus Kaggle entpackten Klassenordner werden direkt unter `data/ALASKA2/` abgelegt.

```text
data/ALASKA2/
├── Cover/
│   ├── 00001.jpg
│   └── ...
├── JMiPOD/
│   ├── 00001.jpg
│   └── ...
├── JUNIWARD/
│   ├── 00001.jpg
│   └── ...
└── UERD/
    ├── 00001.jpg
    └── ...
```

Alle vier Klassen müssen:

- direkt unter `data/ALASKA2/` liegen
- JPEG-Dateien enthalten
- für zusammengehörige Cover- und Stego-Bilder dieselben Basisnamen verwenden

Die Endungen `.jpg` und `.jpeg` werden unabhängig von der Gross- und Kleinschreibung unterstützt. Das unbeschriftete Kaggle-Verzeichnis `Test/` wird nicht benötigt.

Der Datensatz und die GPU-Umgebung können im Dev Container geprüft werden mit:

```bash
poetry run alaska2-dataset-preflight --root data/ALASKA2
poetry run alaska2-gpu-preflight
```

Der Dataset-Preflight liest und validiert die JPEG-Dateien, verändert sie jedoch nicht.

### Synthetischer Ersatzdatensatz

Bei automatischer Datenwahl bereitet das Notebook den öffentlichen Ersatzdatensatz bei Bedarf unter `data/PD12M/` vor.

Downloads und Generierung verwenden kurzlebige, versteckte Arbeitsverzeichnisse direkt unter `data/`. Dauerhaft gespeichert werden nur die ausgewählten Cover-Bilder und die drei synthetischen Varianten.

Die Konfiguration `DATASET_SOURCE` steuert die Auswahl:

- `"auto"` bevorzugt eine vollständige ALASKA2-Struktur und verwendet andernfalls den synthetischen Ersatzdatensatz
- `"alaska2"` verlangt ausdrücklich den realen Datensatz und deaktiviert den synthetischen Fallback
- `"synthetic"` wählt ausschliesslich den Ersatzdatensatz

Fehlt der synthetische Ersatzdatensatz, kann ihn das Notebook bei aktiviertem `DOWNLOAD_PUBLIC_PROXY_IF_NEEDED` vorbereiten.

Reale ALASKA2-Bilder und synthetische Bilder werden weder im Dateisystem noch innerhalb eines Modelllaufs vermischt.

</details>

<details>
<summary><strong>TinyCNN und EfficientNet-B0 trainieren</strong></summary>

Das Training wird in der Codezelle **«Zentrale Daten- und Trainingskonfiguration»** aktiviert:

```python
DATASET_SOURCE = "alaska2"
TRAIN_TINYCNN = True
TRAIN_EFFICIENTNET = True
```

Mit den beiden Trainingsflags können TinyCNN, EfficientNet-B0 oder beide Modelle nacheinander ausgeführt werden. Sie sind standardmässig deaktiviert, damit das Öffnen und Ausführen der Präsentationszellen kein unbeabsichtigtes Training startet.

Vor dem ersten Optimierungsschritt prüfen der Dataset- und der GPU-Preflight die erforderlichen Laufzeitbedingungen. Beide Modelle verwenden die dokumentierten gemeinsamen Trainings-, Validierungs- und Testmitgliedschaften. Die Modellauswahl erfolgt ausschliesslich anhand der Validierungs-Weighted-AUC. Der vollständige Testsplit wird erst nach der Wiederherstellung des besten Modellzustands einmalig ausgewertet.

Checkpoints und Trainingsverläufe werden nach Datensatz, Modell und Laufkennung unter `checkpoints/` beziehungsweise `reports/` gespeichert. Diese Laufzeitartefakte bleiben lokal und werden nicht versioniert. Die kleine, dauerhaft benötigte Ergebnisevidenz für Notebook und Evaluationswidget liegt unter `artifacts/`.

</details>

## 📂 Projektstruktur

<details>
<summary><strong>Projektstruktur anzeigen</strong></summary>

```text
.
├── .devcontainer/                                                       # GPU-Entwicklungscontainer
│   ├── Dockerfile                                                       # PyTorch-, CUDA- und Poetry-Umgebung
│   └── devcontainer.json                                                # Workspace, GPU und VS-Code-Werkzeuge
│
├── .github/
│   └── workflows/
│       └── lint.yml                                                     # CPU-basierte Qualitätsprüfung
│
├── .vscode/
│   └── settings.json                                                    # Gemeinsame VS-Code-Einstellungen
│
├── artifacts/                                                           # Kuratierte Ergebnisläufe
│   └── alaska2/
│       └── alaska2_retrain_tiny10_effnet10_seed42_20260721/
│           ├── comparison.csv                                           # Datenidentischer Modellvergleich
│           ├── final_summary.json                                       # Zusammenfassung des Ergebnislaufs
│           ├── manifest/                                                # Konfiguration, Treiber und Prüfsummen
│           ├── splits/
│           │   └── split_membership.json                                # Gemeinsame Split-Mitgliedschaften
│           ├── tinycnn/
│           │   ├── evaluation/                                          # Metriken, ROC- und Score-Daten
│           │   ├── histories/                                           # Vollständiger Trainingsverlauf
│           │   ├── checkpoints/                                         # Ausgewähltes Modell, lokal ignoriert
│           │   └── predictions/                                         # Testvorhersagen, lokal ignoriert
│           └── efficientnet_b0/
│               ├── evaluation/                                          # Metriken, ROC- und Score-Daten
│               ├── histories/                                           # Kombinierter Stufenverlauf
│               ├── checkpoints/                                         # Ausgewähltes Modell, lokal ignoriert
│               └── predictions/                                         # Testvorhersagen, lokal ignoriert
│
├── assets/                                                              # Direkt eingebundene Notebook-Medien
│   └── notebook/                                                        # Logo, Signatur sowie DCT-/IDCT-Lehrmedien
│
├── data/                                                                # Lokale Datensatzwurzeln
│   ├── ALASKA2/                                                         # Cover, JMiPOD, JUNIWARD und UERD
│   └── PD12M/                                                           # Öffentlicher synthetischer Ersatzdatensatz
│
├── src/
│   ├── __init__.py                                                      # Exportiert die öffentlichen Teilpakete
│   │
│   ├── config/
│   │   ├── __init__.py                                                  # Exportiert die Konfigurationsschnittstellen
│   │   ├── config_device.py                                             # Prüft CUDA und wählt das Ausführungsgerät
│   │   ├── config_paths.py                                              # Definiert Projekt-, Daten- und Artefaktpfade
│   │   └── config_runtime.py                                            # Konfiguriert Seeds und deterministische Ausführung
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_index.py                                                # Indexiert vollständige Cover- und Stego-Gruppen
│   │   ├── data_metadata.py                                             # Extrahiert JPEG-Metadaten und Quantisierungstabellen
│   │   ├── data_preparation.py                                          # Orchestriert Indexierung, Splits und EDA-Auswahl
│   │   ├── data_preflight.py                                            # Validiert den lokalen ALASKA2-Datensatz
│   │   ├── data_split.py                                                # Erzeugt quellenweise getrennte Datensplits
│   │   └── data_synthetic.py                                            # Bereitet den synthetischen PD12M-Proxy vor
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── datasets_dct.py                                              # Lädt DCT-Koeffizienten und Fusionstensoren
│   │   ├── datasets_images.py                                           # Lädt dekodierte JPEG-Bildtensoren
│   │   └── datasets_loaders.py                                          # Erstellt Trainings-, Validierungs- und Testloader
│   │
│   ├── eda/
│   │   ├── __init__.py
│   │   ├── eda_channels.py                                              # Analysiert Pixel- und Kanalverteilungen
│   │   ├── eda_controls.py                                              # Steuert interaktive EDA-Auswahlkomponenten
│   │   ├── eda_dct.py                                                   # Analysiert Quantisierung und DCT-Änderungen
│   │   ├── eda_examples.py                                              # Visualisiert Cover- und Stego-Beispiele
│   │   ├── eda_overview.py                                              # Fasst Datensatzstruktur und Klassen zusammen
│   │   └── eda_style.py                                                 # Vereinheitlicht Reihenfolge und Darstellungsstil
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluation_metrics.py                                        # Berechnet Weighted AUC und Klassifikationsmetriken
│   │   ├── evaluation_plots.py                                          # Lädt Ergebnisevidenz und erzeugt Auswertungsplots
│   │   └── evaluation_runner.py                                         # Evaluiert ein Modell in einem Loader-Durchlauf
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── models_efficientnet.py                                       # Konfiguriert EfficientNet-B0 für YCbCr-Eingaben
│   │   ├── models_freezing.py                                           # Definiert die reproduzierbaren Fine-Tuning-Stufen
│   │   └── models_tinycnn.py                                            # Implementiert die kompakte TinyCNN-Baseline
│   │
│   ├── presentation/
│   │   ├── __init__.py
│   │   ├── presentation_cache.py                                        # Verwaltet den lokalen EDA-Abbildungscache
│   │   └── presentation_widgets.py                                      # Erstellt die interaktiven EDA- und Evaluationswidgets
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── training_artifacts.py                                        # Erzeugt typsichere Pfade für Trainingsartefakte
│   │   ├── training_checkpoint.py                                       # Speichert und restauriert Modellzustände
│   │   ├── training_loop.py                                             # Trainiert Modelle mit Validierung und Early Stopping
│   │   └── training_staged.py                                           # Steuert das stufenweise EfficientNet-Fine-Tuning
│   │
│   └── transforms/
│       ├── __init__.py
│       ├── transforms_shuffle.py                                        # Ordnet Bildkacheln zufällig neu
│       └── transforms_spatial.py                                        # Erzeugt blockausgerichtete Bildausschnitte
│
├── scripts/
│   └── setup_colab.py                                                   # Idempotente Colab-Umgebungseinrichtung
│
├── tests/                                                               # Verhaltens- und Vertragsprüfungen
│
├── checkpoints/                                                         # Lokal erzeugte Modellzustände
├── reports/                                                             # Lokal erzeugte Trainingsverläufe
│
├── .dockerignore                                                        # Begrenzt den Docker-Build-Kontext
├── .gitattributes                                                       # Repository-weite Dateiattribute
├── .gitignore                                                           # Ignoriert Daten und Laufzeitartefakte
├── ANN_Projekt_Rino_Albertin_Steganalyse.ipynb                          # Akademische Ausarbeitung
├── LICENSE.md                                                           # MIT-Lizenz
├── README.md                                                            # Projektübersicht und Ausführung
├── poetry.lock                                                          # Aufgelöste Abhängigkeiten
├── poetry.toml                                                          # Lokale Poetry-Konfiguration
└── pyproject.toml                                                       # Paket und Qualitätswerkzeuge
```

Die Verzeichnisse `checkpoints/` und `reports/` werden bei Bedarf durch Trainingsläufe erstellt und sind nicht Bestandteil des versionierten Repository-Inhalts.

</details>

## 📄 Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE.md).

## 📚 Quellen

- Kaggle, **ALASKA2 Image Steganalysis Challenge**:
  [https://www.kaggle.com/competitions/alaska2-image-steganalysis](https://www.kaggle.com/competitions/alaska2-image-steganalysis)
- Rinovative, **PD12M DCT-based Synthetic Steganography Dataset**:
  [https://huggingface.co/datasets/Rinovative/pd12m_dct_based_synthetic_stegano](https://huggingface.co/datasets/Rinovative/pd12m_dct_based_synthetic_stegano)
- OST – Ostschweizer Fachhochschule, **Lehrunterlagen Applied Neural Networks**, Frühjahr 2025.