[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rinovative/alaska2-steganalysis/blob/main/ANN_Projekt_Rino_Albertin_Steganalyse.ipynb)
_Interaktives Jupyter Notebook direkt im Browser öffnen (via Colab)_

> **Hinweis:**  
> Wenn der originale ALASKA2-Datensatz nicht verfügbar ist, wird automatisch ein synthetisches Demo-Datensatz geladen.
> Das synthetische Demo-Subset enthält keine echten Nachrichten, sondern simuliert typische Frequenzmodifikationen echter Steganographieverfahren.

# Deep Learning für Steganalyse – ALASKA2-Datensatz (Walhfachprojekt)

**Walhfachprojekt** im Rahmen des Studiengangs  
**BSc Systemtechnik – Vertiefung Computational Engineering**  
**Frühjahr 2025** – OST – Ostschweizer Fachhochschule  
**Autor:** Rino Albertin

---

## 📌 Projektbeschreibung

Dieses Projekt befasst sich mit der Erkennung steganographischer Manipulationen in JPEG-Bildern mittels Deep Learning.  
Primärer Datensatz ist der **ALASKA2-Datensatz**, ein Benchmark für moderne Bildsteganalyseverfahren.

Da ALASKA2 aus Lizenzgründen nicht frei verfügbar ist, wird für Demonstrationszwecke ein **synthetisches Ersatz-Datensatz** auf Basis des **PD12M-Datensatzes** bereitgestellt.  
Der Code passt sich automatisch an den verfügbaren Datensatz an.

Das Projekt umfasst:

- Falls der ALASKA2-Datensatz nicht vorhanden ist, die Erstellung synthetischer Stego-Varianten (JMiPOD, JUNIWARD, UERD),
- strukturierte explorative Datenanalyse (EDA),
- Training und Hyperparametertuning von CNN-basierten Modellen zur Steganalyse,
- Evaluation und Visualisierung der Ergebnisse.

---

## ⚙️ Lokale Ausführung
<details>
<summary><strong>Variante A – Ausführung in Visual Studio Code mit Docker</strong> (empfohlen)</summary>

**Voraussetzungen:**

- [Docker Desktop](https://www.docker.com/products/docker-desktop) ist installiert
- [Visual Studio Code](https://code.visualstudio.com/) ist installiert
- Die Erweiterung **"Dev Containers"** ist in VS Code aktiviert

**Vorgehen:**

1. Repository klonen:
   ```bash
   git clone https://github.com/Rinovative/alaska2-steganalysis.git
   cd alaska2-steganalysis
   ```

2. Projektverzeichnis in Visual Studio Code öffnen

3. Container starten:
   - Entweder über die Schaltfläche `Reopen in Container` unten rechts  
   - oder über `F1` → `Dev Containers: Reopen in Container`

4. Container schliessen
   Nach dem ersten Build-Fenster:
   -  Unten links auf das grüne Remote-Symbol klciken → `Close Remote Connection`

5. Dev-Container erneut öffnen
   -  Wieder F1 → `Dev Containers: Reopen in Container`

6. Notebook starten
   -  Öffne `ANN_Projekt_Rino_Albertin_Steganalyse.ipynb` in VS Code.  
   
</details>

<details>
<summary><strong>Variante B – Ausführung über Docker CLI (ohne VS Code)</strong></summary>

**Voraussetzungen:**

- [Docker](https://www.docker.com/) ist installiert und lauffähig

**Vorgehen:**

1. Repository klonen:
   ```bash
   git clone https://github.com/Rinovative/alaska2-steganalysis.git
   cd alaska2-steganalysis
   ```

2. Docker-Image erstellen:
   ```bash
   docker build -t stego-dev .
   ```

3. Container starten und Projektverzeichnis einbinden:
   ```bash
   docker run -it --rm -p 8888:8888 -v $(pwd):/app stego-dev
   ```

4. Innerhalb des Containers Jupyter Notebook starten:
   ```bash
   jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
   ```

5. Die in der Konsole ausgegebene URL kann verwendet werden, um über einen lokalen Browser auf das Notebook zuzugreifen.

</details>

---

## 📂 Projektstruktur
<details>
<summary><strong>Projektstruktur anzeigen</strong></summary>

```bash
.
├── .devcontainer/                        # Docker-Container-Konfiguration für die Entwicklung
│   ├── devcontainer.json                 # Konfigurationsdatei für Visual Studio Code DevContainer
│   └── Dockerfile                        # Dockerfile zur Erstellung eines Entwicklungscontainers für die Umgebung
│
├── .github/                              # GitHub-spezifische Workflows und Aktionen
│   └── workflows/                        # Enthält CI/CD-Workflows für GitHub Actions
│       └── lint.yml                      # Linter-Workflow, der bei jeder Codeänderung ausgeführt wird, um den Code zu prüfen und zu formatieren
│
├── cache/                                # Zwischengespeicherte Daten (z.B. vorverarbeitete Bilder, Trainingsdaten)
│   ├── alaska2/                          # Enthält Zwischenspeicher-Daten für den ALASKA2-Datensatz
│   └── pd12m/                            # Enthält Zwischenspeicher-Daten für den PD12M-Datensatz (synthetische Stego-Varianten)
│
├── data/                                 # Datenverzeichnis
│   └── raw/                              # Rohdaten
│       ├── alaska2-image-steganalysis/   # Enthält Cover + Stego-Varianten (JMiPOD, JUNIWARD, UERD)
│       └── PD12M/                        # Enthält Cover + synthetische Stego-Varianten (JMiPOD, JUNIWARD, UERD)
├── images/                               # Grafiken für Visualisierungen (ROC, AUC, etc.)
│
├── src/                                  # Quellcode des Projekts
│   ├── eda/                              # Explorative Datenanalyse (Modulstruktur)
│   │   ├── __init__.py                   # Initialisierungsdatei für das EDA-Modul
│   │   ├── eda_color_channel_statistics.py  # Analyse der Farbkanäle in den Bildern (Erklärung und Visualisierung)
│   │   ├── eda_dct.py                    # DCT-basierte Bildanalyse, zur Untersuchung der Frequenzkomponenten
│   │   ├── eda_examples.py               # Beispielvisualisierungen der Bilder (Cover vs. Stego)
│   │   └── eda_overview.py               # Übersicht und Zusammenfassung der explorativen Datenanalyse
│   │
│   ├── model/                            # Modellarchitektur, Training und Evaluation
│   │   ├── __init__.py                   # Initialisierungsdatei für das Modellmodul
│   │   ├── model_train.py                # Trainingsskript für das Modell (Modellaufbau, Training, Optimierung)
│   │   ├── model_evaluation.py           # Evaluierung des Modells (z.B. mit AUC, ROC, Konfusionsmatrix)
│   │   ├── model_metrics.py              # Berechnung und Visualisierung von Metriken (Loss, Accuracy, AUC)
│   │   ├── model_plot.py                 # Visualisierung von Ergebnissen (z.B. Konfusionsmatrix, Feature-Importanz)
│   │
│   ├── util/                             # Hilfsfunktionen für Datenvorverarbeitung und Notebook-Unterstützung
│   │   ├── util_cache.py                 # Caching-Funktionen für Plots und Berechnungen
│   │   ├── util_data.py                  # Funktionen für das Laden und Vorverarbeiten von Daten
│   │   ├── util_nb.py                    # Funktionen zur Unterstützung von Jupyter-Notebooks (z.B. Widgets, Panels)
│   │   └── poetry/                       # CI/CD-Linting-Konfiguration für Poetry
│   │       └── poetry_lint.py
│   │
│   └── __init__.py
│
├── .gitignore                            # Ausschlussregeln für Git (z.B. temporäre Dateien, IDE-Settings, etc.)
├── ANN_Projekt_Rino_Albertin_Steganalyse.ipynb  # Haupt-Jupyter-Notebook für das Steganalyse-Projekt
│   └── Das Notebook enthält:
│       ├── Einleitung und Beschreibung des Projekts
│       ├── Explorative Datenanalyse
│       ├── Modelltraining und Hyperparameter-Tuning
│       ├── Evaluierung und Visualisierung der Ergebnisse
├── LICENSE                               # Lizenzdatei für das Projekt (MIT License)
├── poetry.lock                           # Fixierte Abhängigkeiten (Poetry), um Abhängigkeiten für das Projekt zu sperren
├── pyproject.toml                        # Projektdefinition und Abhängigkeiten (Poetry), welche die Python-Pakete und Versionen festlegt
├── README.md                             # Projektübersicht und Erklärung der Zielsetzung und Methodik
└── requirements.txt                      # Alternativ für Pip / Binder / Colab, um die Abhängigkeiten zu installieren
```
</details>

---

## 📄 Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).

---

## 📚 Quellen

- ALASKA2-Datensatz:  
  [Kaggle – ALASKA2 Steganalysis Challenge](https://www.kaggle.com/competitions/alaska2-image-steganalysis)

- Synthetischer Demo-Datensatz:  
  [Hugging Face – Rinovative/pd12m_dct_based_synthetic_stegano](https://huggingface.co/datasets/Rinovative/pd12m_dct_based_synthetic_stegano)

- Lehrunterlagen „Applied Neural Networks“ – OST – Ostschweizer Fachhochschule