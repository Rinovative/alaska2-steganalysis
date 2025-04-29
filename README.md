[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rinovativ/alaska2-steganalysis/blob/main/ANN_Projekt_Rino_Albertin_Steganalyse.ipynb)
_Interaktives Jupyter Notebook direkt im Browser öffnen (via Colab)_

> **Hinweis:**  
> Wenn der originale ALASKA2-Datensatz nicht verfügbar ist, wird automatisch ein synthetisches Demo-Datensatz geladen.

# Deep Learning für Steganalyse – ALASKA2-Datensatz (Walhfachprojekt)

**Semesterprojekt** im Rahmen des Studiengangs  
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

1. Repository klonen:
   ```bash
   git clone https://github.com/DEIN_USERNAME/REPO.git
   cd REPO
   ```

2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

3. Notebook starten:
   ```bash
   jupyter notebook
   ```

4. Notebook öffnen:  
   `ANN_Projekt_Rino_Albertin_Steganalyse.ipynb`

---

## 📂 Projektstruktur

```bash
.
├── data/
│   └── raw/
│       ├── alaska2-image-steganalysis/   # Enthält Cover + Stego-Varianten (JMiPOD, JUNIWARD, UERD)
│       └── PD12M/                        # Enthält Cover + synthetischer Stego-Varianten (JMiPOD, JUNIWARD, UERD)
├── images/                               # Grafiken
├── cache/                                # Zwischengespeicherte Daten und Plots
├── src/
│   ├── eda/                              # Explorative Datenanalyse (Modulstruktur)
│   │   ├── __init__.py
│   ├── util/                             # Hilfsfunktionen
│   │   ├── util_cache.py                 # Plot-Caching und Steuerung
│   │   ├── util_data.py                  # Datensatz-Vorverarbeitung und -Download
│   │   ├── util_nb.py                    # Notebook-Unterstützung (Widgets, Panels)
│   │   └── poetry/                       # CI/CD-Linting-Konfiguration
│   │       └── poetry_lint.py
│   └── __init__.py
├── .gitignore                            # Ausschlussregeln für Git
├── ANN_Projekt_Rino_Albertin_Steganalyse.ipynb  # Hauptnotebook
├── LICENSE                               # Lizenzdatei (MIT License)
├── poetry.lock                           # Fixierte Abhängigkeiten (Poetry)
├── pyproject.toml                        # Projektdefinition (Poetry)
├── README.md                             # Projektübersicht (diese Datei)
└── requirements.txt                      # Alternativ für Pip / Binder / Colab
```

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

---

**Hinweis:**  
Das synthetische Demo-Subset enthält keine echten Nachrichten, sondern simuliert typische Frequenzmodifikationen echter Steganographieverfahren.