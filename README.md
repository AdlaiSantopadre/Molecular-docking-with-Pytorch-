# Descrizione progetto 
Il progetto di studio si articola in due parti:
1. Costruzione di un modello Ligand-only per la predizione regressiva della affinità di legame, cui fa riferimento il notebook Model DockingRNN.ipynb
2. Propositi di evoluzione in un modello generativo in prima istanza di tipo VAE 


Nel primo notebook utilizzeremo il dataset fornito per addestrare, validare e testare  un modello istanziato dalla classe DockingRNN progettata tramite una rete RNN in PyTorch. Adotteremo tecniche di data augmentation, regolarizzazione e controllo dell'overfitting.

Per la trattazione del punto 1  si faccia riferimento alle celle di testo del notebook citato. 
Un riassunto è proposto anche nel due file:

## 1.Descrizione_Modello_RNN_Docking 

Il punto 2 prevede più direttrici di studio ed implementazione:
# Evoluzione del modello RNN LSTM ligand-only


  
  - **Estensione a VAE molecolare**
    - 1. Riutilizzo dell’encoder LSTM
    - 2. Sampling da spazio latente (reparametrization trick)
    - 3. Decoder: RNN o Transformer
    - 4. Soluzioni al KL collapse (β-VAE, cyclical annealing)
    - 5. Generazione molecolare: validità, novità, similarità

  - **Meccanismi di attenzione**
    - 1. Attenzione soft globale su token SELFIES
    - 2. Attenzione come feature extractor per il regressore
    - 3. Auto-attention vs cross-attention (protein-ligand)

  - **Estensione a protein-ligand VAE**
    - 1. Rappresentazione della proteina come grafo (GCN, GAT)
    - 2. Concatenazione o attenzione incrociata ligand-protein
    

  - **Alternative architetturali: Transformers**
    - 1. Sostituzione dell’LSTM con Transformer encoder
    - 2. Decoder Transformer autoregressivo nel VAE
    - 3. Self-attention per generazione molecolare controllata

  - **Librerie, dataset e metriche**
    - 1. SELFIES, RDKit, DeepChem
    - 2. Dataset: ZINC, ChEMBL, BindingDB
    - 3. Metriche: RMSE, MAE, R², validità molecolare
    - 4. Visualizzazione embedding e molecole: UMAP, MolView, PyMol

  
