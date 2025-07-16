# Docking score prediction con RNN

## Pipeline di lavoro

### 1. **Caricamento e pulizia dati**
- Dataset originale in CSV con SMILES e docking scores separati da `;`
- Conversione dei docking scores da stringhe a `float`
- Rimozione di duplicati su base `SMILES` 


### 2. **Data augmentation SELFIES**
- Per ogni molecola rappresentata in SMILE otterremo:
  + 1 SELFIES equivalente allo SMILE del dataset  
  + 2 SELFIES da SMILES randomizzati con `doRandom=True`
per un totale di 3 rappresentazioni SELFIES per la stessa molecola con  stesso score 


### 3. **Tokenizzazione e codifica**
- Estrazione dei token da tutti i SELFIES
- Costruzione di un vocabolario `{token: index}` con indice 0 riservato al padding
- Codifica di ogni sequenza SELFIES in indici numerici

### 4. **Split del dataset**
- 80% training, 10% validation, 10% test
- Split eseguito dopo data augmentation 

---

## Architettura del modello RNN

```python
class DockingRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        ...
```

- Layer di embedding per trasformare i token in vettori
- LSTM multi-layer (2 livelli) con dropout interno
- Considerato dropout aggiuntivo tra LSTM e layer fully-connected
- Regressore finale `fc` che restituisce uno score scalare

---

## Allenamento del modello

- Ottimizzazione con Adam (`lr=1e-3`, `weight_decay=1e-5`)
- MSE loss
- Early stopping attivo con `patience=3`
- Tracciamento metriche: `train/val loss`, `MAE`, `RMSE`, `R2`
- Salvataggio del miglior modello con `torch.save(model.state_dict())`

---

## Valutazione sul test set

- Calcolo delle metriche:
  - `RMSE`, `MAE`, `R²`
- Visualizzazione Predetto vs Reale con Matplotlib
- Identificazione dei peggiori outlier (molecole con errore più alto)

---

## Strategie anti-overfitting adottate

- Dropout (0.3–0.4)
- Early stopping
- Regularizzazione L2 (weight decay)
- Data deduplication
- Data augmentation su SELFIES


---

## Persistenza e salvataggio

- Salvataggio del modello completo con `torch.save(model)`
- Download locale via `files.download()`
- Alternativamente: salvataggio su Google Drive

---

## Stato finale

- Ultimo test: `RMSE ≈ 0.7541`, `MAE ≈ 0.5804`, `R² ≈ 0.4236`
- Addestramento fermato alla 18ª epoca su 30 grazie a early stopping
- Modello più profondo e regolarizzato rispetto alla versione iniziale

---
## Approfondimenti e idee future

- Visualizzazione degli embedding con UMAP/t-SNE (è presente un notebook con un esempio applicato ai dati del dataset di test)
- Riutilizzo del modello RNN come encoder in un VAE
- Aggiunta di feature RDKit (LogP, MW, TPSA...)
- Attenzione bidirezionale e meccanismi di attention
- Uso di SELFIES random per aumentare la robustezza

**Prossimi step suggeriti:**
- Visualizzazione embedding
- Estensione VAE
- Feature engineering chimico
- Valutazione su dataset esterno o struttura target-specifica

