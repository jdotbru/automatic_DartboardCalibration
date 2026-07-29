# Projektidee
Für ein automatisiertes Zählersystem auf Basis einer schräg positionierten Kamera
ist es vor dem Beginn des Spiels essentiell, das Board sehr detailliert und exakt zu identifizieren. 
Schon leichte Abweichungen führen zu regelmäßigen Fehlinterpretationen und einem Stören des Spielflusses. 
Bislang wurde in meinem System die Dartscheibe per Markierung der Eckpunkte manuell
vorgegeben. Deutlich genauer und komfortabler wäre jedoch eine automatische, eventuell regelmäßig aktualisierte, Erkennung und Berechnung des Boards durch das System selber, um leichte Veränderungen des Kamerawinkels auszugleichen.
Dafür soll ein Skript für die automatisierte Erkennung und Kalibrierung einer Dartscheibe entwickelt werden. Die Funktionalitäten umfassen dabei:
- das Erkennen der Dartscheibe und ihrer Sektoren
- das Erkennen des 20-Punkte-Segmentes und der darauf basierenden Zuordnung aller Punktzahlen
- das Erkennen einer zu stark gedrehten Dartscheibe und einer daraus folgenden Warnung an den Spieler
- einer Transformation einer Homographie um einen schrägen Kamerwawinkel auszugleichen

# Related Work und Datensätze
Der Großteil der dokumentierten Umsetzungen von automatischen Dartzählersystemen
fokussiert sich vorrangig auf die Erkennung der Pfeile und deren Treffpunkt mit der Scheibe. 
Dennoch lassen sich Inspirationen zu der Erkennung der Dartscheibe in Projekten wie DeepDarts (https://openaccess.thecvf.com/content/CVPR2021W/CVSports/papers/McNally_DeepDarts_Modeling_Keypoints_as_Objects_for_AutomaJc_Scorekeeping_in_Darts_CVPRW_2021_paper.pdf) oder Dartboard-Detector-Computer-Vision (https://github.com/MichaelRol/Dartboard-Detector-Computer-Vision) finden und nutzen. Mit Datensätze, wie beispielsweise von Roboflow (https://universe.roboflow.com/board-detecJon/board-dart-detect), lässt sich der Algorithmus sinnvoll mit öffentlich zugänglichen Datensätzen testen. Für eine Live-Kamera-Anwendung wie in diesem Projekt wird jedoch vorrangig eine reale Dartscheibe benötigt, an der das System getestet werden kann.

# Vorgehen
Der erste Ansatz des Projektes basierte auf einer, wie in der Vorlesung gelernten, Kantendetektierung und Erkennung im Graustufenbild. Dieser Ansatz zeigte jedoch recht schnell Probleme damit, die exakten Ellipsen des Dartboards zu erkennen, was für eine funktionale Kalibrierung jedoch essenziell ist.
<p align="center">
  <img src="Bilder/Kantenerkennung.png" width="50%"/>
</p>

Im zweiten Ansatz wurde der farbige Aufbau der Scheibe genutzt.
Dabei sucht der Algorithmus im ersten Schritt nach größeren Clustern von grünen oder roten Pixeln.
Durch die Berechnung der Mittelpunkte jedes Clusters, ist eine Einteilung in einen äußeren und einen inneren Ring möglich. Dadurch besitzt das System einen äußeren Ring an Segmenten und einen inneren Ring an Segmenten, welche über eine Rot-Grün-Maske sehr deutlich wird.
Um diese "Segment-Ringe" wird jeweils um die Punkte mit dem höchsten und dem niedrigsten Radius eine Ellipse gelegt, um die Grenzen der Doppel- und Dreifachfelder und damit die grobe Form der Scheibe zu definieren.
<p align="center">
  <img src="Bilder/Farbsegmente_gerade.png" width="25%" hspace="30"/>
  <img src="Bilder/Farbsegmente_schräg.png" width="25%" hspace="30"/>
  <img src="Bilder/Rot_Gruen_Maske.png" width="25%" hspace="30"/>
</p>
![alt text](image.png)

Der nächste Schritt sieht das Identifizieren der Punktsektoren-Linien vor. Dafür wird an der kürzeren Seite der, im ersten Schritt identifizierten, Farbsegmente eine angenäherte Gerade angelegt, welche bis zum äußeren "Single-Bull" (grüner Ring, 25 Punkte) führt. Damit sind Einzel-, Doppel- und Dreifach-Felder, sowie Bull und Single Bull eindeutig definiert und abgegrenzt.
<p align="center">
  <img src="Bilder/Sektoren_ohne20.png" width="50%"/>
</p>

Nun ist es wichtig, zu erkennen wie die Scheibe hängt und wo welche Punktzahl liegt.
Dafür wird von den äußersten Eckpunkten jedes Segments ein Trapez nach außen aufgespannt, um die Zahl am Segment einzuschließen. Jedes dieser Trapeze wird geometrisch entzerrt und mit einem vortrainierten easyOCR-Modell ausgewertet. Wird eine 20 erkannt, so wird diese umrandet und auf Basis davon die typische Dartboard-Punktereihenfolge eingefügt. Bei einer Identifikation der 20 über einem gegebenen Wert (in diesem Projekt 5 Grad) wird eine Warnung ausgegeben, dass die Scheibe gerade gedreht werden sollte.
<p align="center">
  <img src="Bilder/Zahlensektor_verzerrt.png" width="30%" hspace="40"/>
  <img src="Bilder/Zahlensektor_entzerrt.png" width="30%" hspace="40"/>
</p>

Im Anschluss wird das aufgenommene Dartboard in Referenz zu einem Bild von vorne transformiert, sodass eine Homographie von vorne entsteht. Dies passiert auf Basis von identifizierten Merkmalen im Referenz- und im Livebild. Durch diese Homographie ist es im Anschluss möglich, die automatische Kalibrierung in ein komplettes Scoring-System zu integrieren.

Der letzte Schritt ist das Testen der Robustheit des Systems auf Störungen. Dafür wurden unterschiedliche, realistische Szenarien getestet. Abgesehen von drei typischen Störfiltern wurde das System mit einer teilweise verdeckten Scheibe und mit steckenden Darts getestet.

# Ergebnisse und Auswertung
Das System zeigt eine sinnvolle und robuste Erkennung der Dartscheibe. 
Die einzelnen Segment werden in Echtzeit mit einem Live-Kamerabild verhätlnismäßig sehr genau erkannt. Hierbei funktioniert sowohl die Farb-Cluster-Erkennung, als auch die Berechnung der Segmentgrenzen stabil von vorne und von der Seite und eine transformierte Homographie des Dartboards auf ein Referenzbild wird erstellt.
Auch die Erkennung der 20 funktioniert hinreichend genau, sodass aus verschiedenen Winkeln und bei leichter Störung zuverlässig der Drehwinkel der Scheibe erkannt werden kann.
<p align="center">
  <img src="Bilder/Endergebnis_gerade.png" width="20%" hspace="25" />
  <img src="Bilder/Endergebnis_schräg.png" width="20%" hspace="25" />
  <img src="Bilder/Homographie.png" width="20%" hspace="25" />
  <img src="Bilder/Endergebnis_gedreht.png" width="20%" hspace="25" />
</p>
![alt text](image.png)
Ein Auswertungsvergleich zwischen einer manuelle und einer automatischen Kalibrierung ergab eine prozentuale Quote von korrekt erkannten Würfen von 80% bei einer manuellen Kalibrierung und von 81% bei einer automatischen Kalibrierung mit diesem System.

Bereits steckende Pfeile und das Abhängen eines kleinen Bereiches der Scheibe beeinflusst die Funktionalität kaum. Lediglich die Erkennung der 20 und damit die Definition der Punktefelder wird durch das Abhängen der 20 vollständig verhindert.
Probleme weist das System jedoch bei starken Störungsfiltern auf. Gerade die Erkennung der Farb-Cluster wird stark von nicht-farbigen Pixeln gestört und unterbrochen, wodurch bei einem zu hohen Störgrad falsche Segmente identifiziert werden.
<p align="center">
  <img src="Bilder/steckende_Pfeile.png" width="15%" />
  <img src="Bilder/verdecktes_Board.png" width="15%" />
  <img src="Bilder/Gaussian.png" width="15%" />
  <img src="Bilder/Salt_Pepper.png" width="15%" />
  <img src="Bilder/Speckle.png" width="15%" />
</p>

# Poster und unterstützende Medien
[Projektposter (PDF)](docs/Poster.pdf)


## Voraussetzungen

- macOS
- Python 3.10 oder neuer

## Installation

Im Terminal in diesen Projektordner wechseln und eine virtuelle Umgebung
anlegen:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Starten

```bash
python main.py
```

Beim ersten Start fragt macOS eventuell nach dem Kamerazugriff. Diesen Zugriff
erlauben. Die Einstellung befindet sich unter **Systemeinstellungen >
Datenschutz & Sicherheit > Kamera**.

Zum Beenden im Kamerafenster `Esc` drücken. Alternativ kann das Fenster
über die Titelleiste geschlossen werden.

Zum Speichern eines neuen Referenzbildes im Kamerafenster `r` drücken.

Falls Kamera `0` nicht die gewünschte Kamera ist, kann ein anderer Index
ausgewählt werden:

```bash
python main.py --camera 1 oder in der main-Datei ändern
```
