# SUMO_MARL — Come raccontare il progetto

> Una guida in italiano per spiegare **motivazioni, architettura, famiglie di algoritmi e limiti di valutazione** di SUMO_MARL. Le affermazioni sono ancorate agli script presenti nel repository.
>
> [System Design con diagrammi](SYSTEM_DESIGN.md) · [README](../README.md)

## 1. Perché nasce il progetto?

Regolare i semafori è un problema di decisione sequenziale: concedere il verde a una direzione può ridurne la coda, ma può anche creare congestione più avanti. Quando nella stessa rete ci sono più incroci, le decisioni non sono indipendenti.

Da qui nasce il problema progettuale:

**È possibile adattare il controllo dei semafori allo stato del traffico e come cambia il comportamento degli algoritmi quando le intersezioni da coordinare aumentano?**

Il progetto usa **SUMO**, un simulatore microscopico del traffico, per studiare questa domanda senza intervenire su infrastrutture reali. Confronta un riferimento **fixed-time**, basato su tempi di fase predefiniti, con diversi approcci di reinforcement learning multi-agente.

La motivazione non è dimostrare a priori che il MARL sia sempre migliore: il punto è capire **quando l'apprendimento e il coordinamento sono utili, quali compromessi introducono e quanto conta la complessità dello scenario**.

## 2. La soluzione spiegata semplicemente

L'architettura mette insieme tre livelli.

**Simulazione del traffico.** SUMO gestisce veicoli, percorsi, corsie e semafori su una rete urbana. TraCI permette a un programma Python di comunicare con la simulazione.

**Ambiente multi-agente.** Gli esperimenti RL usano **SUMO-RL** e un'interfaccia in stile **PettingZoo**, adattata a **Ray RLlib**. Ogni semaforo può essere trattato come un agente: osserva informazioni locali e sceglie una fase. L'ambiente restituisce la nuova osservazione e una ricompensa.

**Apprendimento e analisi.** RLlib addestra le configurazioni DQN, PPO e QMIX. Ulteriori script sperimentano modelli di critic centralizzati o dotati di attention. Infine gli script di confronto analizzano le metriche di traffico prodotte dai rollout.

### Flusso essenziale

```text
      Rete stradale e itinerari RESCO
                       |
                 Simulatore SUMO
                       |
                  SUMO-RL
                       |
             Osservazioni per TLS
                       |
           DQN / PPO / QMIX / critic
                       |
                  Scelta fase
                       |
                  SUMO avanza
                       |
              Reward + nuovo stato
                       |
                Training RLlib
                       |
           Metriche e grafici storici

           Percorso separato:
       Fixed-time -> TraCI -> SUMO
```

**Nota importante:** gli algoritmi sono **alternative sperimentali**, non parti di una sequenza in cui DQN alimenta PPO e poi QMIX.

## 3. Perché SUMO, TraCI e RLlib?

| Tecnologia | Motivazione |
| --- | --- |
| **SUMO** | Riprodurre il movimento dei veicoli e l'effetto delle politiche semaforiche in simulazione |
| **TraCI** | Controllare il simulatore e leggere grandezze di traffico da Python |
| **SUMO-RL** | Trasformare i semafori in un ambiente di apprendimento con stati, azioni e ricompense |
| **PettingZoo e SuperSuit** | Gestire il caso multi-agente e, dove necessario, uniformare spazi di osservazione/azione |
| **Ray RLlib** | Configurare e addestrare modelli di reinforcement learning e raccogliere rollout |
| **PyTorch** | Implementare le reti e i critic personalizzati |
| **Pandas / Matplotlib** | Esaminare CSV storici e grafici di waiting time, velocità e code |

Non basta scrivere «ho usato MARL»: devi saper spiegare **quale componente genera le osservazioni, quale decide l'azione e quale misura il risultato**.

## 4. Quali approcci sono stati studiati?

### A. Fixed-time: il controllo di riferimento

Il semaforo segue una sequenza di fasi con durate prefissate. Non apprende una politica e non si adatta in tempo reale nello stesso modo di un agente RL.

**Perché includerlo?** Permette di valutare se l'aumento di complessità degli algoritmi appresi è giustificato da un miglioramento del traffico.

Nei test multi-incrocio, però, le implementazioni fixed-time richiedono attenzione: in alcune versioni la progressione delle fasi è sincronizzata usando un incrocio di riferimento e la raccolta delle metriche non è perfettamente uniformata agli altri controller.

### B. IDQN e IPPO: controllo con informazioni locali

Gli script etichettati IDQN utilizzano `DQNConfig`, una famiglia value-based; quelli etichettati IPPO utilizzano `PPOConfig`, una famiglia policy-gradient.

**Perché studiarli?** Sono un riferimento importante per valutare se una decisione basata prevalentemente sull'osservazione dell'agente basti o se servano meccanismi espliciti di coordinamento.

**Precisione sul codice:** i runner ispezionati mappano più agenti alla policy RLlib chiamata `shared`. Quindi **non sarebbe corretto affermare che ogni semaforo dispone necessariamente di una rete completamente indipendente**. Si tratta di configurazioni DQN/PPO applicate all'ambiente multi-agente con **parameter sharing**.

### C. QMIX: decomposizione del valore

L'idea alla base di QMIX è utilizzare una forma di valore di squadra che combina le stime dei singoli agenti, mantenendo una struttura utile alla selezione decentralizzata delle azioni.

Gli script presenti configurano `QMixConfig` e raggruppano gli agenti in una struttura `all_tls` prima dell'addestramento.

**Motivazione:** esplorare un coordinamento più esplicito rispetto al caso in cui ogni agente ottimizzi soltanto la propria esperienza.

**Limite:** il codice dipende da API legacy di RLlib. La presenza di `QMixConfig` non sostituisce una prova di compatibilità eseguendo davvero l'esperimento con versioni fissate.

### D. MAPPO, critic centralizzati e attention

L'idea di **Centralized Training, Decentralized Execution (CTDE)** è usare informazioni più ampie per addestrare la funzione di valore, lasciando che l'attore scelga le proprie azioni tramite le osservazioni disponibili localmente.

Il repository contiene runner PPO con classi personalizzate di critic e varianti che includono **self-attention** sui vettori degli agenti.

**Questa è la parte che va raccontata con maggiore rigore.** Nei runner esaminati non compare un collegamento completo ed esplicito fra osservazioni concatenate di tutti gli agenti, chiamate a `forward_critic` e calcolo della loss PPO. Non basta definire una classe `CentralizedCriticModel` per dimostrare che il training sia davvero centralizzato.

Lo presenterei così:

> Ho esplorato architetture orientate al CTDE, implementando varianti di critic centralizzato e attention. Per considerarle implementazioni MAPPO pienamente verificate, il passo successivo è validare l'effettiva costruzione degli input globali e il flusso dei gradienti nel training.

### E. GAT-style e temporal observations

Alcuni critic chiamati “GAT” usano `nn.MultiheadAttention` sui vettori degli agenti. È un esperimento interessante, ma **non è automaticamente una Graph Attention Network che usa archi e topologia stradale**.

Il ramo `resco_ingolstadt1/lstm_mappo/` include un wrapper che concatena le osservazioni di più istanti temporali. Nel modello ispezionato la rete è composta da layer fully-connected; non si vede una vera cella LSTM. Per il portfolio è più corretto parlare di **finestra temporale di osservazione**, non di LSTM comprovato.

## 5. Gli scenari e cosa cambia

| Scenario | Obiettivo del confronto |
| --- | --- |
| **Cologne1** | Studiare il comportamento nel caso etichettato come incrocio singolo |
| **Cologne3** | Aumentare l'interazione tra segnali |
| **Ingolstadt1** | Analizzare un differente contesto di traffico, anche con varianti temporali |
| **Ingolstadt7** | Esaminare controller in una rete più articolata |
| **Ingolstadt21** | Esplorare i limiti di scala e di coordinamento in uno scenario più ampio |

I numeri sono quelli dei nomi di scenario utilizzati dal progetto. Senza i file XML originali non possiamo verificare qui tutti i dettagli effettivi delle reti e dei loro flussi.

Gli script contengono spesso configurazioni come episodio di **3600 secondi**, scelta dell'azione ogni **10 secondi**, `yellow_time=3` e reward `diff-waiting-time`, ma l'ora iniziale varia da scenario a scenario.

## 6. Come vengono valutati i controller?

Il lavoro analizza principalmente tre grandezze:

- **Tempo medio di attesa:** in generale si vuole ridurlo.
- **Lunghezza o entità delle code:** si vuole contenere il numero di veicoli in attesa.
- **Velocità media:** descrive la fluidità del traffico, da interpretare insieme alle altre metriche.

Queste sono **metriche di valutazione del traffico**, da non confondere con la reward impiegata nel training.

Il README e le figure conservate mostrano andamenti e confronti tra controller. Tuttavia alcuni script costruiscono il riepilogo selezionando il **miglior episodio di training per waiting time**. Questo può produrre una valutazione ottimistica e non sostituisce il confronto tra policy congelate su episodi indipendenti.

Nei colloqui parlerei dei **trend esplorati**, ma eviterei di attribuire percentuali di miglioramento non derivate da una valutazione omogenea con molteplici seed.

## 7. Presentazione da 30 secondi

> Ho sviluppato un progetto sperimentale di Multi-Agent Reinforcement Learning per il controllo dei semafori in SUMO. L'obiettivo era studiare come algoritmi diversi reagiscono alla congestione e come cambia la necessità di coordinamento passando da un singolo incrocio a reti più grandi. Ho configurato controller fixed-time, DQN, PPO e QMIX, esplorando anche critic centralizzati e meccanismi di attention. Ho analizzato tempi di attesa, code e velocità per confrontare le strategie in più scenari di traffico. Il punto centrale è capire quali benefici e difficoltà emergono quando più agenti condividono una rete stradale.

## 8. Presentazione tecnica da circa 90 secondi

> Il progetto nasce da un problema di controllo sequenziale: ogni semaforo prende decisioni locali, ma le sue scelte influenzano il traffico agli incroci vicini. Per questo ho studiato diverse strategie di Multi-Agent Reinforcement Learning in ambienti simulati.
>
> Ho utilizzato SUMO per simulare il traffico e TraCI per comunicare con l'ambiente. Gli script RL sfruttano SUMO-RL e PettingZoo per rappresentare i segnali come agenti e Ray RLlib per configurare e addestrare le politiche.
>
> Ho considerato innanzitutto un controller fixed-time come riferimento non adattivo. Ho poi sperimentato DQN e PPO con policy sharing e un approccio QMIX basato sul raggruppamento degli agenti. Ho esplorato inoltre architetture PPO con critic personalizzati e varianti attention-oriented, ispirate al paradigma CTDE.
>
> Gli esperimenti sono organizzati in scenari Cologne e Ingolstadt di scala differente. Per studiarne il comportamento ho utilizzato metriche come waiting time, numero di veicoli fermi e velocità media, raccogliendo curve e confronti dai CSV di simulazione.
>
> La parte più interessante è la relazione tra **complessità della rete, informazioni disponibili a ogni agente e necessità di coordinamento**. Il limite principale del progetto è che alcuni esperimenti di critic centralizzato richiedono una verifica più completa del data flow e i risultati storici non costituiscono ancora un benchmark rigoroso con checkpoint congelati e test su seed indipendenti.

## 9. Domande tecniche da colloquio

| Domanda | Risposta difendibile |
| --- | --- |
| **Perché MARL invece di un solo agente?** | Perché la rete contiene più controllori locali che interagiscono indirettamente tramite il traffico e le code. |
| **Che cosa fa SUMO?** | Simula la dinamica dei veicoli e della rete; non è l'algoritmo RL. |
| **A cosa serve TraCI?** | A leggere grandezze e comandare il simulatore da Python. |
| **Quali sono stato, azione e reward?** | Stato/azione dipendono dall'interfaccia SUMO-RL; l'azione sceglie una fase, la reward configurata è `diff-waiting-time`. |
| **Perché usare un fixed-time?** | Per avere una baseline senza apprendimento e verificare se la complessità aggiunta è utile. |
| **IDQN e IPPO sono implementati con una rete per ogni semaforo?** | Nei runner letti no: c'è una policy `shared` per gli agenti; la denominazione è quella sperimentale, non una prova di parametrizzazione indipendente. |
| **Che cos'è QMIX?** | Un algoritmo di value decomposition che combina stime degli agenti per apprendere un valore di squadra; gli script configurano un gruppo di agenti. |
| **Cosa vuol dire CTDE?** | Training con informazioni centralizzate e decisioni decentralizzate durante l'esecuzione. |
| **Hai dimostrato un MAPPO centralizzato completo?** | Non ancora: i modelli critic ci sono, ma va verificata l'integrazione delle osservazioni globali e della critic loss in RLlib. |
| **Hai usato una vera Graph Attention Network?** | Le varianti ispezionate usano self-attention sui vettori degli agenti; non vedo un messaggio-passaggio esplicito basato sugli archi della rete stradale. |
| **Perché osservazioni temporali?** | Per fornire indizi sulla dinamica recente del traffico; il ramo temporale ispezionato concatena osservazioni, non prova l'esistenza di una cella LSTM. |
| **Come valuti gli algoritmi?** | Waiting time, queue/stopped vehicles e speed, preferibilmente su rollout standardizzati e policy congelate. |
| **Perché non basta confrontare il miglior episodio?** | Perché scegliere il miglior valore fra molti episodi introduce una stima ottimistica delle prestazioni. |
| **Quale sarebbe il prossimo miglioramento?** | Verificare il critic centralizzato, ripristinare le reti RESCO, fissare le versioni e adottare un protocollo di valutazione multi-seed. |

## 10. Punti da conoscere prima di mostrarlo come demo

I file necessari sotto `nets/RESCO/` **non sono presenti** nel repository. Le dipendenze Python non hanno versioni fissate e alcuni script usano API RLlib legacy. Anche i CSV grezzi dei rollout e i checkpoint non sono inclusi.

Inoltre le curve storiche dimostrano che sono stati prodotti grafici e analisi, ma da sole non provano risultati statistici o l'assenza di errori nei singoli esperimenti. Per questo nella documentazione tecnica ho separato le intenzioni di ogni famiglia di algoritmi dalle proprietà di cui il codice fornisce effettivamente evidenza.

Questa impostazione è utile anche al colloquio: puoi spiegare **ciò che hai progettato, le difficoltà affrontate e quali controlli tecnici aggiungeresti**, senza rivendicare capacità non dimostrate.

## 11. La frase che riassume il progetto

> **Ho usato la simulazione del traffico per studiare un problema di controllo multi-agente, confrontando diverse strategie di apprendimento e approfondendo il compromesso tra decisioni locali, coordinamento e complessità degli scenari.**
