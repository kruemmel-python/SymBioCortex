# SymBioCortex – Wissenschaftliche Dokumentation

Diese Dokumentation vermittelt einen ganzheitlichen Blick auf SymBioCortex für drei Zielgruppen: **Wissenschaftler\*innen**, **Programmierer\*innen** und **Einsteiger\*innen**. Sie bündelt Systemübersichten, algorithmische Details, Bedienhinweise und didaktische Leitfäden, sodass Forschung, Entwicklung und praktisches Lernen nahtlos ineinandergreifen.

---

## Inhaltsverzeichnis
1. [Projektüberblick](#projektüberblick)
2. [Systemarchitektur](#systemarchitektur)
3. [Daten, Tokenisierung und Training](#daten-tokenisierung-und-training)
4. [Generierung, Neologie und Morphologie](#generierung-neologie-und-morphologie)
5. [Reranking und Coaching](#reranking-und-coaching)
6. [Schwarmökologie und Feldphysik](#schwarmökologie-und-feldphysik)
7. [Feedback-Loops und Autopoiesis](#feedback-loops-und-autopoiesis)
8. [Werkzeuge für Programmierer\*innen](#werkzeuge-für-programmiererinnen)
9. [Forschungsleitfaden für Wissenschaftler\*innen](#forschungsleitfaden-für-wissenschaftlerinnen)
10. [Lernpfad für Einsteiger\*innen](#lernpfad-für-einsteigerinnen)
11. [Reproduzierbarkeit, Logging und Tests](#reproduzierbarkeit-logging-und-tests)
12. [Glossar zentraler Komponenten](#glossar-zentraler-komponenten)

---

## Projektüberblick
SymBioCortex verbindet ein symbolisches Langzeitgedächtnis mit einem energie-basierten Schwarmfeld. Der BioCortex übernimmt Tokenisierung, Sprachmodellierung, Myzel-Graph und Replay-Speicher, während das HPIO-System (Hyperspatial Pulse Interaction Organism) Impulse in ein Φ-Feld einbettet und von Agenten erkunden lässt.【F:src/symbio/biocortex.py†L18-L119】【F:src/symbio/hpio.py†L13-L58】 Der Orchestrator koppelt beide Welten über Ereignisse wie Pulse, Tick-Schritte und Feedback, wodurch eine zyklische Symbiose aus Denken, Handeln und Rückkopplung entsteht.【F:src/symbio/orchestrator.py†L17-L98】

---

## Systemarchitektur
Die Architektur folgt einem modularen Aufbau:

- **BioCortex**: kapselt Tokenizer, Kneser-Ney-Sprachmodell, Myzel-Graph, Neuromodulatorik und Replay-Speicher. Er trainiert auf Sequenzen, aktualisiert Lexika und speichert Relationen als gewichtete Kanten.【F:src/symbio/biocortex.py†L18-L72】
- **HPIO**: kombiniert ein 2D-Φ-Feld mit einem agentenbasierten Schwarm, der Gradienten folgt, Puls-Energie einbringt und Trails hinterlässt.【F:src/symbio/hpio.py†L13-L58】【F:src/symbio/swarm.py†L24-L104】
- **Orchestrator**: verwaltet Events, injiziert Pulse aus Textprompts und triggert Feedback-Schleifen inklusive Hotspot-Detektion und Neuromodulation.【F:src/symbio/orchestrator.py†L33-L98】
- **Streamlit-UI**: stellt Tabs für Denken, Handeln und Symbiose bereit, erlaubt Parametersteuerung (Neologismen, Gewichte) und visualisiert Graphen sowie Feldwerte.【F:apps/streamlit_app.py†L45-L138】

Die modulare Gliederung erleichtert wissenschaftliche Experimente (z. B. Variation der Neologismenrate) ebenso wie Softwaretests und UI-Prototyping.

---

## Daten, Tokenisierung und Training
Der BioCortex nutzt einen Byte-Pair-Tokenizer (BioBPETokenizer) und lernt Sequenzen über ein Kneser-Ney-n-Gramm-Modell. Beim `partial_fit` werden Texte tokenisiert, das Sprachmodell aktualisiert, der Replay-Puffer gefüllt und der Myzel-Graph mit Pheromonen verstärkt.【F:src/symbio/biocortex.py†L45-L72】 Gleichzeitig entsteht ein Lexikon, das später für Neologie-Metriken und Snapping genutzt wird.【F:src/symbio/biocortex.py†L62-L63】【F:src/symbio/metrics/neology.py†L8-L31】 

Für reproduzierbare Experimente sollten Korpora als UTF-8-Textdateien bereitgestellt werden. Das CLI-Kommando `symbio train --data ...` lädt mehrere Dateien, ruft `partial_fit` auf und speichert das Modellverzeichnis für nachfolgende Generierungsschritte.【F:src/symbio/apps/cli.py†L24-L74】

---

## Generierung, Neologie und Morphologie
Beim Generieren dekodiert der BioCortex das Prompt, bestimmt Wahrscheinlichkeiten via Kneser-Ney und mischt bekannte Tokens mit einem Neologismen-Gate. Die Funktion `sample_mixed` ermöglicht kontrolliertes Auswürfeln neuer Wörter basierend auf einer `neo_rate`.【F:src/symbio/biocortex.py†L74-L118】【F:src/symbio/generate/mix_sampler.py†L1-L28】 Guardrails um den Morphologie-Generator sorgen für wohlgeformte Wortkandidaten, indem Präfix-/Suffix-Priors und Qualitätsfilter angewendet werden.【F:src/symbio/morph/guardrails.py†L1-L42】 Neologie-Statistiken werden live geloggt, was Regressionstests und kreative Experimente unterstützt.【F:src/symbio/biocortex.py†L105-L118】【F:src/symbio/metrics/neology.py†L8-L31】

Wesentliche Parameter:
- `max_new_tokens`: steuert die Länge der Fortsetzung.
- `nucleus_p` (Top-p) und `temperature`: kontrollieren Vielfalt versus Determinismus.
- `neo_rate`: reguliert den Anteil neuer Wörter.

---

## Reranking und Coaching
Zur Qualitätssteigerung generiert das CLI mehrere Kandidaten und bewertet sie mit dem Coaching-Modul:

- **Fluency**: log-probabilistischer Score des Kneser-Ney-Modells, sigmoidal normalisiert.【F:src/symbio/coach/rerank.py†L45-L106】
- **Semantik**: Kosinus-Ähnlichkeit zwischen TF-IDF-Vektoren von Prompt und Kandidat.【F:src/symbio/coach/rerank.py†L32-L71】
- **Form**: heuristischer Score für Satzzeichen, Großschreibung und Tokenlängen.【F:src/symbio/coach/rerank.py†L73-L91】
- **Neologie**: Abgleich mit dem Korpuslexikon inklusive Zielband-Penalty.【F:src/symbio/coach/rerank.py†L93-L135】
- **Snapping**: optionales Anpassen instabiler Tokens an nahe Lexikoneinträge via Damerau-Levenshtein-Distanz.【F:src/symbio/coach/levenshtein.py†L1-L40】【F:src/symbio/coach/rerank.py†L137-L174】

Gewichte und Zielband lassen sich sowohl im CLI (`--w-*`, `--neo-low`, `--neo-high`, `--snap`) als auch in der Streamlit-Oberfläche anpassen.【F:src/symbio/apps/cli.py†L34-L118】【F:apps/streamlit_app.py†L64-L120】 Das Feedback-Log dokumentiert jede Session inklusive Parameter und Scores und dient als Datengrundlage für das Tuning-Skript `symbio tune` (gradientenfreies Gewichtsanpassen).【F:src/symbio/apps/cli.py†L87-L152】【F:src/symbio/apps/cli.py†L184-L208】

---

## Schwarmökologie und Feldphysik
Das Φ-Feld speichert Pulsenergie und tagbasierte Imprints, diffundiert Energie über Nachbarn und lässt sie langsam verdampfen.【F:src/symbio/field.py†L10-L78】 Agenten des Schwarms ertasten Gradienten, interagieren durch Kohäsion und Vermeidung und injizieren Pulse entlang ihrer Trails.【F:src/symbio/swarm.py†L24-L104】 Das HPIO-Subsystem synchronisiert Feld und Schwarm, verfolgt den aktuell besten Hotspot und führt Relaxationsschritte aus.【F:src/symbio/hpio.py†L13-L58】 Diese Kopplung bildet die Grundlage für autopoietisches Verhalten und Feedback-getriebene Exploration.

---

## Feedback-Loops und Autopoiesis
Der Orchestrator übersetzt Texte in Pulse, lässt das HPIO-System iterativ reagieren und wertet entstehende Hotspots als Feedback für den BioCortex aus.【F:src/symbio/orchestrator.py†L33-L98】 Die Funktion `autopoietic_cycle` fügt mehrere Texte ins Feld ein, wartet auf stabile Hotspots und synthetisiert daraus neue Sätze über den BioCortex, wodurch eine Form autopoietischer Selbstreferenz entsteht.【F:src/symbio/orchestrator.py†L66-L98】

---

## Werkzeuge für Programmierer\*innen
- **CLI**: `symbio` unterstützt die Kommandos `train`, `generate`, `run`, `autopoiesis` und `tune`. Pattern Matching (PEP 634) mappt Subbefehle auf Funktionen, und `generate` erstellt N-Kandidaten, führt das Coaching aus und schreibt optionale Logs.【F:src/symbio/apps/cli.py†L16-L208】
- **Streamlit-App**: bietet interaktive Kontrolle, Visualisierung und Exportpfade unter `apps/streamlit_app.py`.
- **Konfigurationssystem**: Standardkonfigurationen liegen in `symbio/config.py` und können bei Bedarf überschrieben werden.
- **Modularer Code**: Tokenisierung (`symbio/core/tokenize.py`), Mix-Sampler, Morph-Guardrails und Coaching-Bibliothek sind klar getrennt, was Unit-Tests und alternative Implementierungen erleichtert.

Programmierer\*innen können so gezielt in einzelne Schichten eingreifen, beispielsweise neue Reranking-Features hinzufügen oder alternative Tokenizer experimentell anbinden.

---

## Forschungsleitfaden für Wissenschaftler\*innen
1. **Korpusgestaltung**: Nutzen Sie `datasets/sample_corpus.txt` als Referenz und erweitern Sie das Trainingsmaterial gezielt, um Domänenwissen einzubringen.【F:README.md†L33-L71】
2. **Neologie-Kontrolle**: Variieren Sie `neo_rate`, Zielband und Guardrails, um kreative vs. konservative Sprachmuster zu untersuchen.【F:src/symbio/biocortex.py†L74-L118】【F:src/symbio/coach/rerank.py†L93-L135】
3. **Schwarmmetriken**: Überwachen Sie Hotspots, Agentenpfade und Feldenergie über das HPIO-Subsystem sowie die Streamlit-Visualisierung.【F:src/symbio/hpio.py†L13-L58】【F:apps/streamlit_app.py†L125-L168】
4. **Feedback-Tuning**: Sammeln Sie Nutzerpräferenzen im JSONL-Log und führen Sie `symbio tune` aus, um Heuristiken datengetrieben nachzujustieren.【F:src/symbio/apps/cli.py†L124-L208】
5. **Autopoiesis-Studien**: Analysieren Sie `autopoietic_cycle`, um zu erforschen, wie Feldreaktionen neue Konzepte hervorbringen.【F:src/symbio/orchestrator.py†L66-L98】

---

## Lernpfad für Einsteiger\*innen
1. **Setup**: Python 3.12-Umgebung anlegen und `pip install -e .` ausführen (siehe README).【F:README.md†L19-L30】
2. **Erstes Training**: `symbio train --data datasets/sample_corpus.txt` erzeugt ein Grundmodell.【F:README.md†L33-L52】【F:src/symbio/apps/cli.py†L24-L74】
3. **Text generieren**: `symbio generate --prompt "Das Leben ist" --n-candidates 4 --neo-rate 0.2 --debug` zeigt Top-Ausgaben inklusive Score-Details.【F:README.md†L33-L52】【F:src/symbio/apps/cli.py†L75-L152】
4. **UI ausprobieren**: `streamlit run apps/streamlit_app.py` öffnen, Parameter schieben und Hotspots beobachten.【F:README.md†L54-L71】【F:apps/streamlit_app.py†L45-L168】
5. **Neologie verstehen**: Vergleichen Sie die Neologismus-Quote in Debug-Ausgaben mit den Guardrail-Reglern und diskutieren Sie, warum bestimmte Wörter „neu“ sind.

Dieser Lernpfad erleichtert den Übergang von der Theorie zur praktischen Exploration.

---

## Reproduzierbarkeit, Logging und Tests
SymBioCortex legt Wert auf Nachvollziehbarkeit:
- **Logging**: Das Generate-Kommando speichert Prompts, Parameter, Scores und Ränge in einer JSONL-Datei für spätere Analyse und Gewichts-Tuning.【F:src/symbio/apps/cli.py†L124-L152】
- **Tuning**: `symbio tune --log ... --step 0.05` passt Gewichte iterativ an Nutzerfeedback an.【F:src/symbio/apps/cli.py†L184-L208】
- **Tests**: `pytest -q` deckt zentrale Komponenten (Neologie, Reranking, Feld-Schwarm-Interaktion) ab und sollte nach Änderungen laufen.【F:README.md†L73-L79】

---

## Glossar zentraler Komponenten
- **BioCortex**: Gedächtnisarchitektur mit Tokenizer, Sprachmodell, Myzelgraph und Neuromodulation.【F:src/symbio/biocortex.py†L18-L119】
- **HPIO**: Feld- und Schwarmverbund für Pulsverarbeitung und räumliche Exploration.【F:src/symbio/hpio.py†L13-L58】
- **Pulse**: Gaußförmige Energieeinträge, die Tags tragen und Hotspots auslösen.【F:src/symbio/field.py†L40-L67】
- **Hotspots**: Feldregionen mit hoher Energie, dienen als Feedbackanker.【F:src/symbio/field.py†L68-L78】
- **Neologie**: Anteil neuer Tokens gegenüber dem Korpuslexikon, messbar über `NeologyStats`.【F:src/symbio/metrics/neology.py†L8-L31】
- **Reranking**: Bewertung mehrerer Kandidaten nach Fluency, Semantik, Form und Neologie, optional mit Snapping.【F:src/symbio/coach/rerank.py†L32-L174】
- **Autopoiesis**: Zyklus, in dem Feldreaktionen neue Sätze hervorbringen.【F:src/symbio/orchestrator.py†L66-L98】

---

Mit dieser Dokumentation erhalten Forschung, Entwicklung und Lernende einen gemeinsamen Referenzpunkt, um SymBioCortex verantwortungsvoll weiterzuentwickeln und seine Kreativität zielgerichtet zu steuern.
