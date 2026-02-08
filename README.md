# Multi-Agent Reinforcement Learning per il Controllo Traffico
**Teoria, Algoritmi e Valutazione Sperimentale in SUMO (TraCI)** 

<font color="#57606a">Autore:</font> **Paolo Pangallo** — <font color="#57606a">Matricola:</font> **263594**  
<font color="#57606a">Università:</font> **Università della Calabria (DIMES) — Ingegneria Informatica**  
<font color="#57606a">Anno Accademico:</font> **2025/2026** 

---

## Indice
1. Panoramica
2. Formalizzazione del problema (Markov Game / Dec-POMDP)
3. Scenario & benchmark (RESCO: Cologne, Ingolstadt)
4. Metriche e pipeline sperimentale (SUMO + TraCI)
5. Algoritmi MARL implementati
6. Risultati sperimentali (tabelle complete)
7. Discussione e conclusioni
8. Riproducibilità e note pratiche (Ray/RLlib + TraCI)
9. Riferimenti

---

## 1) Panoramica
Il **controllo semaforico su reti urbane** multi-intersezione è un problema **intrinsecamente multi-agente**: ogni incrocio decide localmente, ma la qualità globale dipende dalle **interazioni** tra incroci connessi (spillback, propagazione code, sincronizzazione “green wave”, trade-off tra corridoi).

Questo repository confronta, in SUMO, famiglie di algoritmi:
- **Independent** (baseline scalabili ma senza coordinazione esplicita)
- **Value Decomposition** (cooperazione tramite decomposizione del valore)
- **CTDE** (Centralized Training, Decentralized Execution), incluse varianti con **attenzione** e **graph attention**

---

## 2) Formalizzazione (Markov Game / Dec-POMDP)
Il problema viene discusso come **Markov Game**: stato globale `s`, azione congiunta `a=(a1,...,an)`, transizione `P(s'|s,a)`, reward (cooperativo), e fattore di sconto. Nel traffico:
- `s` include densità/code/velocità/occupazioni sull’intera rete,
- ogni agente `i` sceglie `ai` (fase o azione discreta del semaforo),
- l’evoluzione è governata dal simulatore (SUMO).}

In esecuzione la policy è decentralizzata e fattorizzata:
\[
\pi(a|o)=\prod_i \pi_i(a_i|o_i)
\]
e in training (a seconda dell’algoritmo) può usare informazione globale (CTDE).

---

## 3) Scenari sperimentali (RESCO)
Il progetto usa sottoreti del benchmark **RESCO** (Cologne e Ingolstadt), disponibili in `nets/RESCO` del framework **sumo-rl**.

### 3.1 Scenari usati e lettura qualitativa
| Scenario | Città | # Agenti | Lettura qualitativa |
|---|---:|---:|---|
| cologne1 | Cologne (DE) | 1 | baseline single-agent, utile per tarare reward/osservazioni |
| ingolstadt1 | Ingolstadt (DE) | 1 | baseline single-agent con dinamiche diverse |
| cologne3 | Cologne (DE) | 3 | primo salto multi-agente (coordinazione locale) |
| ingolstadt7 | Ingolstadt (DE) | 7 | cresce l’accoppiamento, coordinazione centrale |
| ingolstadt21 | Ingolstadt (DE) | 21 | scala urbana, massima complessità di coordinazione |
:contentReference[oaicite:6]{index=6}

### 3.2 Domanda di traffico (route-based)
Ogni episodio copre una finestra temporale di **1 ora**, con:
- Cologne: **[25200, 28800]**
- Ingolstadt: **[57600, 61200]** 

**Cologne**
| Scenario | Veicoli/ora | Veicoli/min | Picco 5-min | Depart min–max |
|---|---:|---:|---:|---|
| cologne1 | 2015 | 33.58 | 231 | 25205.00 – 28799.00 |
| cologne3 | 2856 | 47.60 | 381 | 23512.00 – 28798.00 |


**Ingolstadt**
| Scenario | Trip/ora | Veicoli/min | Picco 5-min | Depart min–max |
|---|---:|---:|---:|---|
| ingolstadt1 | 1716 | 28.60 | 173 | 57600.20 – 61198.00 |
| ingolstadt7 | 3031 | 50.52 | 292 | 57600.20 – 61199.70 |
| ingolstadt21 | 4281 | 71.35 | 395 | 57600.00 – 61200.80 |


---

## 4) Ambiente e pipeline (SUMO + TraCI)
L’ambiente è basato su **SUMO** (simulatore microscopico). L’interazione online avviene via **TraCI**, che consente di:
- osservare lo stato (code, velocità, occupazioni, ecc.)
- modificare dinamicamente le logiche semaforiche a runtime

SUMO richiede tipicamente:
- rete `.net.xml`
- domanda/percorsi `.rou.xml`
- eventuali file addizionali (logiche semaforiche, sensori, output)

---

## 5) Algoritmi MARL implementati
Legenda (solo testo):
- <font color="#0969da">Independent</font>
- <font color="#8250df">CTDE</font>
- <font color="#d1242f">Value Decomposition</font>

### 5.1 Tabella comparativa (qualitativa)
| Algoritmo | Categoria | Meccanismo | Pro | Contro / limite atteso |
|---|---|---|---|---|
| IDQN | <font color="#0969da">Independent</font> | Q-learning per agente | semplice, stabile su compiti locali | non-stazionarietà, credit assignment debole |
| IPPO | <font color="#0969da">Independent</font> | PPO per agente | baseline forte, scalabile | coordinazione solo emergente, trade-off globali difficili |
| QMIX | <font color="#d1242f">Value Decomposition</font> | mixer monotono per Q_tot | cooperazione senza critic globale diretto | monotonicità restrittiva, instabilità off-policy |
| MAPPO | <font color="#8250df">CTDE</font> | PPO con critic centralizzato | stabilizza training, migliore credit assignment | critic può non scalare bene senza bias strutturale |
| MAPPO-ATN | <font color="#8250df">CTDE</font> | critic con attenzione | selezione dinamica interazioni rilevanti | costo/complessità architetturale |
| MAPPO-GAT | <font color="#8250df">CTDE</font> | critic con Graph Attention | bias coerente con topologia rete | training più pesante, tuning più delicato |


---

## 6) Risultati sperimentali (metriche SUMO)
Metriche dai log SUMO, mediate sugli episodi di valutazione:
- `mean_wait` (s) — tempo medio di attesa
- `mean_speed` (m/s) — velocità media
- `mean_queue` — lunghezza media code

### 6.1 ingolstadt1 (1 agente)
| Controller | Mean wait [s] | Mean speed [m/s] | Mean queue |
|---|---:|---:|---:|
| Fixed-time | 6.09 | 4.53 | 10.23 |
| IDQN | 229.84 | 3.37 | 16.01 |
| IPPO | 2.82 | 5.28 | 6.72 |
| QMIX | 13.70 | 3.38 | 14.76 |
| MAPPO | 2.24 | 5.58 | 4.54 |
| MAPPO-ATN | 1.97 | 5.61 | 5.32 |
| <font color="#2da44e"><b>MAPPO-GAT</b></font> | <font color="#2da44e"><b>1.47</b></font> | 5.52 | 4.82 |


Interpretazione (sintesi): in regime non saturo l’RL supera nettamente fixed-time; policy-gradient/CTDE risultano più stabili dei value-based. 

---

### 6.2 cologne1 (1 agente) — failure case del RL
| Controller | Mean wait [s] | Mean speed [m/s] | Mean queue |
|---|---:|---:|---:|
| <font color="#2da44e"><b>Fixed-time</b></font> | <font color="#2da44e"><b>7.88</b></font> | <font color="#2da44e"><b>5.21</b></font> | <font color="#2da44e"><b>16.28</b></font> |
| IDQN | 186.54 | 3.84 | 29.75 |
| IPPO | 154.35 | 3.91 | 28.26 |
| QMIX | 248.88 | 3.13 | 38.38 |
| MAPPO | 101.53 | 3.77 | 26.44 |
| MAPPO-ATN | 159.35 | 4.78 | 19.48 |
| MAPPO-GAT | 135.51 | 4.93 | 18.11 |


Interpretazione: domanda più intensa/irregolare → saturazione rapida; anche breve esplorazione produce code persistenti e reward fortemente negativo, degradando l’apprendimento. 

---

### 6.3 ingolstadt7 (7 agenti)
| Controller | Mean wait [s] | Mean speed [m/s] | Mean queue |
|---|---:|---:|---:|
| Fixed-time | 10.79 | 3.42 | 72.17 |
| IDQN | 9.61 | 5.82 | 31.59 |
| IPPO | 20.55 | 5.61 | 34.27 |
| QMIX | 399.68 | 2.83 | 124.68 |
| MAPPO | 15.16 | 6.03 | 32.78 |
| MAPPO-ATN | 7.18 | <font color="#2da44e"><b>6.60</b></font> | <font color="#2da44e"><b>22.78</b></font> |
| <font color="#2da44e"><b>MAPPO-GAT</b></font> | <font color="#2da44e"><b>4.39</b></font> | 6.24 | 24.53 |


Interpretazione: qui emerge il vantaggio MARL “vero” (dipendenze upstream/downstream, spillback, green-wave). Le varianti con critic strutturato (attenzione/GAT) producono il salto maggiore. 

---

### 6.4 ingolstadt21 (21 agenti) — stress test di scala
| Controller | Mean wait [s] | Mean speed [m/s] | Mean queue |
|---|---:|---:|---:|
| <font color="#2da44e"><b>Fixed-time</b></font> | <font color="#2da44e"><b>85.50</b></font> | <font color="#2da44e"><b>4.37</b></font> | 346.99 |
| IDQN | 584.69 | 0.92 | 735.49 |
| IPPO | 445.87 | 3.94 | 230.37 |
| MAPPO | 521.08 | 3.76 | 242.72 |
| MAPPO-ATN | 473.89 | 3.31 | <font color="#2da44e"><b>229.20</b></font> |
| <font color="#2da44e"><b>MAPPO-GAT</b></font> | <font color="#2da44e"><b>281.35</b></font> | 3.85 | 272.75 |


Interpretazione: regime congestion-driven e dipendenze diffuse; lo scenario è uno **stress test** e mostra che scalare a 21 nodi resta difficile (in questa configurazione, fixed-time rimane molto competitivo). :contentReference[oaicite:20]{index=20}

---

## 7) Discussione e conclusioni (messaggi chiave)
- L’efficacia dell’RL dipende dal **regime di traffico**: in contesti non saturi può migliorare molto; in saturazione locale (cologne1) può fallire.   
- Su scenari multi-intersezione (ingolstadt7) l’aumento di accoppiamento/interferenza rende cruciali metodi **CTDE** e critic **strutturati** (attenzione/GAT).   
- La difficoltà non è solo “#agenti”: contano **domanda**, **accoppiamento**, **trade-off** e **interferenza**. 

---

## 8) Riproducibilità e note pratiche (importante con Ray/RLlib)
### 8.1 TraCI e parallelismo
Se lanci rollout in parallelo (Ray/RLlib), attenzione a:
- porte TraCI distinte per ogni istanza SUMO
- sincronizzazione step/ordine di esecuzione  
Altrimenti possono emergere errori di connessione/desincronizzazione. 

### 8.2 Struttura attesa delle reti (RESCO)
Gli scenari sono disponibili (nel framework usato) sotto:
nets/RESCO/<scenario>/
├─ <scenario>.net.xml
└─ <scenario>.rou.xml



---

## 9) Quick-start (template)
> Nota: i nomi degli script possono variare in base alla tua struttura di repo; questi comandi sono un template coerente con pipeline SUMO-RL/RLlib.

### 9.1 Prerequisiti
- SUMO installato (con TraCI)
- Python + dipendenze (Ray/RLlib, PyTorch, sumo-rl)

### 9.2 Install
python -m venv .venv

attiva venv
pip install -r requirements.txt


### 9.3 Esecuzione (esempi)
Fixed-time baseline (valutazione)
python experiments/<scenario>/run_fixed_time.py

Training
python experiments/<scenario>/run_idqn.py
python experiments/<scenario>/run_ippo.py
python experiments/<scenario>/run_qmix.py
python experiments/<scenario>/run_mappo.py
python experiments/<scenario>/run_mappo_atn.py
python experiments/<scenario>/run_mappo_gat.py


---

## Riferimenti (selezione)
- SUMO + TraCI (simulazione e controllo online) 
- RESCO benchmark (scenari Cologne / Ingolstadt) 
- Lente “complessità di coordinazione” (dipendenze/interferenza/goal overlap) 

---

## Licenza

---

## Citazione
Se usi questo lavoro in una relazione o tesi, cita la relazione del progetto:
@misc{pangallo2026marl_traffic_sumo,
title = {Multi-Agent Reinforcement Learning per il Controllo Traffico: Teoria, Algoritmi e Valutazione Sperimentale in SUMO},
author = {Paolo Pangallo},
year = {2026},
note = {Universit`a della Calabria, DIMES, A.A. 2025/2026}
}

