"""
Script di addestramento K-Fold Cross-Validation per xBD damage assessment.

SUPPORTO BACKBONE:
  CNN: ResNet50, EfficientNet (B0/B3), ConvNeXt (Tiny/Small)
  Transformers: Swin Transformer (Tiny/Small)

PERSONALIZZAZIONE:
- Modifica `core/config.py` per scegliere il backbone e i parametri di addestramento.

NOTE OPERATIVE
Provato con: 32Gb RAM, GPU NVIDIA GeForce RTX 4070 8GB, SSD NVMe 1Tb
Impraticabile per il modello Swin Small a causa della memoria GPU limitata.
se mixed precision false, impraticabile anche per Swin Tiny e ConvNeXt Small
"""

import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHYTON_DIR = os.path.join(SCRIPT_DIR, '..')
if PHYTON_DIR not in sys.path:
    sys.path.insert(0, PHYTON_DIR)

from core.config import TrainingConfig
from core.dataset import (create_training_dataset, create_validation_dataset)
from core.models import SiameseNetwork  
from core.losses import create_loss_function
from core.utils import Logger, setup_directories, setup_fold_directories



def collate_fn_skip_corrupted(batch):
    """
    intercetta il batch prima che venga assemblato e scarta 
    qualsiasi campione che abbia causato un errore durante il caricamento    
    """
    original_size = len(batch)
    # Filtra la lista 'batch', tenendo solo gli elementi che non sono None
    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) < original_size:
        print(f"Rimosse {original_size - len(batch)} immagini corrotte in questo batch.")
        
    return torch.utils.data.dataloader.default_collate(batch)

def freeze_backbone(model):
    """
    Congela tutti i parametri del backbone (feature extractor).
    
    Durante Feature Extraction, solo la testa (classifier_head) viene addestrata,
    mentre il backbone pre-addestrato rimane fisso.
    
    Args:
        model: SiameseNetwork con backbone e classifier_head
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    print(" Backbone CONGELATO - Solo classifier_head sarà addestrato")


def unfreeze_backbone(model):
    """
    Scongela tutti i parametri del backbone per il fine-tuning.
    """
    for param in model.backbone.parameters():
        param.requires_grad = True
    print(" Backbone scongelato - Tutto il modello sarà addestrato")


def create_data_loaders(train_df, val_df, config):
    """
    DataLoader organizza i campioni in batch e li fornisce iterativamente al modello.
    Usa Subset per filtrare i campioni del fold corrente.
    
    ARCHITETTURA:
    - HDF5 contiene tutti i campioni 
    - train_df/val_df contengono solo gli indici del fold corrente 
    - Subset filtra l'HDF5 per accedere solo ai campioni del fold (evita duplicazione dati)  
      
    Args:
        train_df (DataFrame): Indici campioni di training per il fold corrente
        val_df (DataFrame): Indici campioni di validation per il fold corrente
        config (TrainingConfig): Configurazione con parametri augmentation e batch size
        
    Returns:
        tuple: (train_loader, val_loader, train_subset, val_subset)
    """
    if config.enable_data_augmentation:
        print("Dataset training con Data Augmentation")
        train_dataset = create_training_dataset(
            hdf5_path=config.hdf5_dataset_path,
            enable_augmentation=True,
            config=config  
        )
    else:
        print("Dataset training senza Data Augmentation")
        train_dataset = create_training_dataset(config.hdf5_dataset_path, enable_augmentation=False)
    
    print("Dataset validation (no augmentation)")
    val_dataset = create_validation_dataset(config.hdf5_dataset_path)
    
    # Estrae indici del fold corrente e crea Subset per filtrare HDF5
    train_indices = train_df.index.tolist()
    val_indices = val_df.index.tolist()
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # DataLoader con shuffle solo per training 
    train_loader = DataLoader(dataset=train_subset, 
                          batch_size=config.batch_size,
                          shuffle=True,  
                          num_workers=config.num_workers,  
                          pin_memory=True,  # Velocizza trasferimento CPU->GPU
                          collate_fn=collate_fn_skip_corrupted)

    val_loader = DataLoader(dataset=val_subset,
                        batch_size=config.batch_size,
                        shuffle=False,  # Mantiene ordine costante per riproducibilità
                        num_workers=config.num_workers,
                        pin_memory=True,
                        collate_fn=collate_fn_skip_corrupted)  
    
    return train_loader, val_loader, train_subset, val_subset

def train_one_epoch(model, train_loader, optimizer, criterion, config, fold_k, epoch, scaler=None):
    """
    Esegue una singola epoca di addestramento con forward/backward pass.
    
    Returns:
        float: Loss media dell'epoca
    """
    model.train()  # Attiva Dropout, BatchNorm in modalità training
    train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Fold {fold_k} Epoch {epoch+1}/{config.num_epochs} [Training]")
    
    for batch_idx, (pre_images, post_images, labels) in enumerate(train_loop):
        # Trasferisce dati su GPU 
        pre_images = pre_images.to(config.device, non_blocking=True)
        post_images = post_images.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)
        
        optimizer.zero_grad()  # Resetta gradienti accumulati
        
        # Mixed precision: calcoli in fp16, pesi in fp32
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(pre_images, post_images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()  # Scala loss prima del backward
            scaler.step(optimizer)  # Aggiorna pesi con scaling
            scaler.update()  # Aggiorna fattore di scaling dinamico
        else:
            # Training standard (fp32)
            outputs = model(pre_images, post_images)
            loss = criterion(outputs, labels)
            loss.backward()  # Calcola gradienti
            optimizer.step()  # Aggiorna pesi
        
        train_loss += loss.item()
        # Aggiorna progress bar ogni 10 batch 
        if batch_idx % 10 == 0:
            train_loop.set_postfix(loss=loss.item())
    
    return train_loss / len(train_loader)

def validate_model(model, val_loader, criterion, config, fold_k, epoch, val_df):
    """
    Valuta il modello sul validation set e raccoglie metriche dettagliate.
    
    Returns:
        tuple: (avg_val_loss, val_f1, all_labels, all_preds, detailed_results_df)
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    detailed_results = []
    val_loop = tqdm(val_loader, desc=f"Fold {fold_k} Epoch {epoch+1}/{config.num_epochs} [Validation]")
    
    val_indices = val_df.index.tolist()
    
    with torch.no_grad():
        batch_idx = 0
        for pre_images, post_images, labels in val_loop:
            pre_images, post_images, labels = pre_images.to(config.device, non_blocking=True), post_images.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True)
            
            # Usa mixed precision anche per la validazione 
            if config.mixed_precision and config.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(pre_images, post_images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(pre_images, post_images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Sposta tutto su CPU in una volta sola per ridurre le trasferenze
            batch_preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            batch_probs = probabilities.cpu().numpy()
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            
            for i in range(len(batch_labels)):
                loader_idx = batch_idx * val_loader.batch_size + i
                
                # Assicura di non andare fuori dai limiti
                if loader_idx < len(val_indices):
                    real_df_idx = val_indices[loader_idx]  # Indice reale nel DataFrame originale
                    
                    detailed_results.append({
                        'sample_index': loader_idx,
                        'true_label': batch_labels[i].item(),
                        'predicted_label': batch_preds[i].item(),
                        'prob_no_damage': batch_probs[i][0].item(),
                        'prob_minor_damage': batch_probs[i][1].item(),
                        'prob_major_damage': batch_probs[i][2].item(),
                        'prob_destroyed': batch_probs[i][3].item(),
                        'disaster_type': val_df.loc[real_df_idx, 'disaster_type']
                    })
            batch_idx += 1
    
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_val_loss = val_loss / len(val_loader)
    
    detailed_results_df = pd.DataFrame(detailed_results)
    
    return avg_val_loss, val_f1, all_labels, all_preds, detailed_results_df

def save_loss_plot(train_losses, val_losses, plots_dir, fold_k, config):
    """
    Genera e salva grafico loss curves per monitorare overfitting.
    
    Args:
        train_losses: Loss per ogni epoca (training)
        val_losses: Loss per ogni epoca (validation)
        plots_dir: Directory output
        fold_k: Numero fold corrente
        config: Configurazione (per num_epochs)
    """
    plt.figure(figsize=(10, 6))  
    
    plt.plot(range(1, config.num_epochs + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(range(1, config.num_epochs + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - Fold {fold_k}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(plots_dir, f'loss_curves_fold_{fold_k}.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Libera memoria (evita accumulo figure)
    print(f"-> Grafico loss salvato: {loss_plot_path}")

def save_confusion_matrix(model, val_loader, plots_dir, fold_k, config):
    """
    Genera confusion matrix per analizzare errori di classificazione del modello.

    Args:
        model: Rete neurale addestrata
        val_loader: DataLoader validation
        plots_dir: Directory output
        fold_k: Numero fold corrente
        config: Configurazione (per target_names)
    """
    model.eval()
    
    final_preds = []
    final_labels = []
    with torch.no_grad():
        for pre_images, post_images, labels in val_loader:
            pre_images, post_images, labels = pre_images.to(config.device, non_blocking=True), post_images.to(config.device, non_blocking=True), labels.to(config.device, non_blocking=True)
            
            # Mixed precision per inferenza (se disponibile)
            if hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast('cuda'):
                    outputs = model(pre_images, post_images)
            else:
                outputs = model(pre_images, post_images)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(final_labels, final_preds)
    
    # Visualizza con heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, 
                annot=True,  # Mostra numeri nelle celle
                fmt='d',
                cmap='Blues',
                xticklabels=config.target_names,
                yticklabels=config.target_names)
    
    plt.title(f'Confusion Matrix - Fold {fold_k}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_plot_path = os.path.join(plots_dir, f'confusion_matrix_fold_{fold_k}.png')
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Confusion Matrix salvata: {cm_plot_path}")


def save_disaster_confusion_matrices(detailed_df, plots_dir, fold_k, config):
    """
    Crea e salva una confusion matrix separata per ogni tipo di disastro.
    
    Args:
        detailed_df (DataFrame): DataFrame con risultati dettagliati (true, predicted, disaster_type).
        plots_dir (str): Directory dove salvare i grafici.
        fold_k (int): Numero del fold corrente.
        config (TrainingConfig): Oggetto configurazione.
    """
    # Ottiene la lista unica dei disastri presenti nel set di validazione
    disaster_types = detailed_df['disaster_type'].unique()
    print(f"-> Generazione Confusion Matrix per tipo di disastro: {list(disaster_types)}")

    for disaster in disaster_types:
        # Filtra il DataFrame per il disastro corrente
        disaster_df = detailed_df[detailed_df['disaster_type'] == disaster]
        
        # Estrae etichette reali e predizioni per questo sottoinsieme
        true_labels = disaster_df['true_label']
        predicted_labels = disaster_df['predicted_label']
        
        # Calcola la confusion matrix specifica
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Crea la heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Greens', # Usiamo un colore diverso per distinguerle
                    xticklabels=config.target_names,
                    yticklabels=config.target_names)
        
        plt.title(f'Confusion Matrix - Fold {fold_k} - Disaster: {disaster}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Pulisce il nome del disastro per creare un nome di file valido
        safe_disaster_name = disaster.replace(" ", "_").lower()
        cm_plot_path = os.path.join(plots_dir, f'confusion_matrix_fold_{fold_k}_{safe_disaster_name}.png')
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Confusion Matrix per '{disaster}' salvata: {cm_plot_path}")

def save_disaster_reports_csv(detailed_df, reports_dir, fold_k):
    """
    Salva un file CSV di report dettagliato per ogni tipo di disastro.
    
    Args:
        detailed_df (DataFrame): DataFrame completo con i risultati.
        reports_dir (str): Directory dove salvare i report CSV.
        fold_k (int): Numero del fold corrente.
    """
    disaster_types = detailed_df['disaster_type'].unique()
    print(f"-> Generazione Report CSV per tipo di disastro: {list(disaster_types)}")

    for disaster in disaster_types:
        # Filtra il DataFrame per il disastro corrente
        disaster_df = detailed_df[detailed_df['disaster_type'] == disaster]
        
        safe_disaster_name = disaster.replace(" ", "_").lower()
        report_path = os.path.join(reports_dir, f'validation_report_fold_{fold_k}_{safe_disaster_name}.csv')
        disaster_df.to_csv(report_path, index=False)
        print(f"  - Report CSV per '{disaster}' salvato: {report_path}")


def save_summary_report(fold_results, fold_detailed_info, config):
    """
    Genera report testuale completo della cross-validation con statistiche aggregate.
    
    CONTENUTO REPORT:
    - Configurazione esperimento (modello, hyperparameters, device)
    - Statistiche aggregate (media F1, std, best/worst fold)
    - Risultati dettagliati per fold (samples, F1, classification report)
    
    Args:
        fold_results: Lista F1-score di ogni fold
        fold_detailed_info: Lista dizionari con info dettagliate per fold
        config: Configurazione training
        
    Returns:
        tuple: (mean_f1, std_f1)
    """
    mean_f1 = np.mean(fold_results)
    std_f1 = np.std(fold_results)
    best_f1 = np.max(fold_results)
    worst_f1 = np.min(fold_results)
    
    summary_path = os.path.join(config.results_dir, 'summary', 'cross_validation_summary.txt')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Header principale
        f.write("="*60 + "\n")
        f.write("RIEPILOGO COMPLETO DELLA CROSS-VALIDATION\n")
        f.write("="*60 + "\n\n")
        
        # Configurazione esperimento
        f.write("--- CONFIGURAZIONE ESPERIMENTO ---\n")
        f.write(f"Modello:              {config.backbone_name} (pretrained={config.pretrained})\n")
        f.write(f"Numero di fold (k):   {config.n_splits}\n")
        f.write(f"Epoche per fold:      {config.num_epochs}\n")
        f.write(f"Batch size:           {config.batch_size}\n")
        f.write(f"Learning rate:        {config.learning_rate}\n")
        f.write(f"Dropout rate:         {config.dropout_rate}\n")
        f.write(f"Loss function:        {config.loss_function}\n")
        f.write(f"Class weights:        {config.class_weights}\n")
        f.write(f"Device:               {config.device}\n\n")
        
        # Statistiche aggregate
        f.write("--- STATISTICHE AGGREGATE (WEIGHTED F1-SCORE) ---\n")
        f.write(f"Media F1-Score:       {mean_f1:.4f}\n")
        f.write(f"Deviazione Standard:  {std_f1:.4f}\n")
        f.write(f"Miglior F1-Score:     {best_f1:.4f}\n")
        f.write(f"Peggior F1-Score:     {worst_f1:.4f}\n\n")
        
        # Risultati dettagliati per fold
        f.write("="*60 + "\n")
        f.write("RISULTATI DETTAGLIATI PER FOLD\n")
        f.write("="*60 + "\n\n")
        
        for fold_info in fold_detailed_info:
            f.write(f"--- FOLD {fold_info['fold']} ---\n")
            f.write(f"Campioni:             {fold_info['train_samples']:,} training | {fold_info['val_samples']:,} validazione\n")
            f.write(f"Miglior F1-Score:     {fold_info['f1_score']:.4f}\n\n")
            f.write("Classification Report (sull'epoca migliore):\n")
            f.write(fold_info['classification_report'])
            f.write("\n\n")
    
    print(f"\n Report completo salvato in: {summary_path}")
    return mean_f1, std_f1

def train_single_fold(fold_k, train_df, val_df, config):
    """
    Addestra e valida il modello per un singolo fold della cross-validation.
    
    FLUSSO COMPLETO:
    1. Setup: DataLoader, modello, optimizer, loss function
    2. Training loop: Gradual unfreezing o full training
    3. Validation: Raccolta metriche e salvataggio best model
    4. Output: Loss curves, confusion matrices, report CSV
    
    Args:
        fold_k: Numero del fold corrente (0 to n_splits-1)
        train_df: DataFrame con campioni training
        val_df: DataFrame con campioni validation
        config: Configurazione training
        
    Returns:
        tuple: (best_f1, detailed_results_df, train_size, val_size, classification_report)
    """
    # Setup logging per questo fold
    models_dir, plots_dir, reports_dir = setup_fold_directories(config.results_dir, fold_k)

    log_path = os.path.join(reports_dir, f'training_log_fold_{fold_k}.txt')
    original_stdout = sys.stdout
    logger = Logger(log_path)
    sys.stdout = logger  # Reindirizza output su file

    try:
        print(f"\n{'='*25}")
        print(f" INIZIO ADDESTRAMENTO PER FOLD {fold_k} ")
        print(f"{'='*25}")

        train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
            train_df, val_df, config)
        
        print(f"Fold {fold_k}: {len(train_dataset)} patch di training, {len(val_dataset)} patch di validazione.")

        model = SiameseNetwork(config)
        print(f"Using device: {config.device}")
        print(f"Modello: {config.backbone_name} | Classi: {config.num_classes} | Hidden: {config.hidden_size}")
        model.to(config.device)

        # Compila modello per prestazioni migliori 
        if hasattr(config, 'compile_model') and config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                print(" Modello compilato con torch.compile")
            except Exception as e:
                print(f" torch.compile non disponibile: {e}")
        
    

        if config.use_gradual_unfreezing:
            print("\n" + "="*60)
            print(" STRATEGIA: GRADUAL UNFREEZING")
            print("="*60)
            print(f"Fase 1 (Epoche 1-{config.frozen_epochs}): Feature Extraction")
            print(f"  • Backbone FROZEN (pesi ImageNet fissi)")
            print(f"  • Solo classifier_head addestrato")
            print(f"  • Learning rate: {config.learning_rate}")
            print(f"\nFase 2 (Epoche {config.frozen_epochs+1}-{config.num_epochs}): Fine-Tuning")
            print(f"  • Backbone UNFROZEN (adattamento dominio)")
            print(f"  • Tutto il modello addestrato")
            print(f"  • Learning rate RIDOTTO: {config.finetune_learning_rate}")
            print("="*60 + "\n")
            
            freeze_backbone(model)
        else:
            print("\n" + "="*60)
            print(" STRATEGIA: FULL TRAINING")
            print("="*60)
            print(f"Tutto il modello addestrato da subito")
            print(f"Learning rate: {config.learning_rate}")
            print("="*60 + "\n")
        
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        criterion = create_loss_function(config, train_df)
        
        # Mixed precision per GPU (fp16/fp32 automatico)
        scaler = torch.amp.GradScaler('cuda') if config.mixed_precision and config.device.type == 'cuda' else None
        if scaler:
            print(" Mixed precision training abilitato")
        
        print("Modello e optimizer inizializzati.")
        
        best_val_f1 = 0.0
        best_detailed_df = pd.DataFrame()
        best_classification_report = ""
        train_losses = []
        val_losses = []

        for epoch in range(config.num_epochs):
            
            # TRANSIZIONE: Feature Extraction → Fine-Tuning
            if config.use_gradual_unfreezing and epoch == config.frozen_epochs:
                print("\n" + ""*30)
                print(" TRANSIZIONE: Feature Extraction → Fine-Tuning")
                print(""*30)
                
                unfreeze_backbone(model)
                
                # Ricrea optimizer con LR ridotto (necessario per cambiare LR)
                optimizer = optim.AdamW(model.parameters(), lr=config.finetune_learning_rate)
                print(f" Optimizer ricreato con LR ridotto: {config.finetune_learning_rate}")
                
                # Ricrea scaler (necessario dopo cambio optimizer)
                if scaler:
                    scaler = torch.amp.GradScaler('cuda')
                    print("✓ GradScaler ricreato")
                
                print(""*30 + "\n")
            
            avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config, fold_k, epoch, scaler)
            
            avg_val_loss, val_f1, all_labels, all_preds, detailed_results_df = validate_model(
                model, val_loader, criterion, config, fold_k, epoch, val_df)
            
            report = classification_report(all_labels, all_preds, target_names=config.target_names, digits=4)
            print(report)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1-Score: {val_f1:.4f}")

            # Salva il modello se migliora l'F1-score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model_path = os.path.join(models_dir, f'best_model_fold_{fold_k}.pth')
                
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f" Best model salvato: {model_path}")
                
                best_detailed_df = detailed_results_df.copy()
                best_classification_report = report
                report_path = os.path.join(reports_dir, f'validation_report_fold_{fold_k}.csv')
                
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                best_detailed_df.to_csv(report_path, index=False)
                print(f" Report dettagliato salvato: {report_path}")

        # FINALIZZAZIONE FOLD
        save_loss_plot(train_losses, val_losses, plots_dir, fold_k, config)
        
        model_path = os.path.join(models_dir, f'best_model_fold_{fold_k}.pth')
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f" Best model caricato per grafici finali")
        else:
            print(f" File modello non trovato, uso stato corrente")
        
        save_confusion_matrix(model, val_loader, plots_dir, fold_k, config)

        if not best_detailed_df.empty:
            save_disaster_confusion_matrices(best_detailed_df, plots_dir, fold_k, config)
            save_disaster_reports_csv(best_detailed_df, reports_dir, fold_k)
        else:
            print(" Nessun risultato dettagliato per report disaster-specific")

        print(f"\n{'='*60}")
        print(f" FOLD {fold_k} COMPLETATO - Best F1: {best_val_f1:.4f}")
        print(f"{'='*60}\n")

    finally:
        logger.close()
        sys.stdout = original_stdout
        print(f" Log salvato: {log_path}")
        
    return best_val_f1, best_detailed_df, len(train_dataset), len(val_dataset), best_classification_report

def train_model():
    """
    Funzione principale che coordina la K-Fold Cross-Validation completa.
    
    PIPELINE COMPLETA:
    1. Carica configurazione (hyperparameters, device, paths)
    2. Carica catalogo CSV e pulisce dati (rimuove 'un-classified')
    3. Loop K-Fold: addestra modello per ogni fold indipendentemente
    4. Aggrega risultati: statistiche cross-validation, performance per disaster
    5. Genera report finali: summary TXT/CSV, disaster-specific analysis
    
    OUTPUT GENERATI:
    - results_dir/fold_X/: modelli, grafici, report per fold
    - results_dir/summary/: report aggregati cross-validation e disaster performance
    
    Returns:
        None: Funzione di orchestrazione che stampa risultati finali
    """
    config = TrainingConfig()
    
    print("INIZIO CROSS-VALIDATION")
    print(config)  # Stampa la configurazione per trasparenza
    
    # Mostra configurazione data augmentation
    if config.enable_data_augmentation:
        print("\n" + "="*60)
        print(" DATA AUGMENTATION: ABILITATO")
        print("="*60)
        print(f"Parametri:")
        print(f"  • Horizontal Flip: p={config.augment_horizontal_flip_prob}")
        print(f"  • Vertical Flip: p={config.augment_vertical_flip_prob}")
        print(f"  • Rotation (90°/180°/270°): p={config.augment_rotation_prob}")
        print(f"  • Color Jitter: p={config.augment_color_jitter_prob}")
        print(f"    - Brightness: ±{int(config.augment_brightness*100)}%")
        print(f"    - Contrast: ±{int(config.augment_contrast*100)}%")
        print(f"    - Saturation: ±{int(config.augment_saturation*100)}%")
        print(f"    - Hue: ±{int(config.augment_hue*100)}%")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(" DATA AUGMENTATION: DISABILITATO")
        print("="*60)
    
    setup_directories(config.results_dir)
    
    # Carica catalogo CSV
    print(f"\n Caricamento catalogo: {config.labels_csv_path}")
    all_labels_df = pd.read_csv(config.labels_csv_path)
    print(f" Catalogo caricato: {len(all_labels_df)} patch totali")

    # Rimuove campioni 'un-classified' 
    all_labels_df = all_labels_df[all_labels_df['damage'] != 'un-classified'].reset_index(drop=True)

    # K-Fold Cross-Validation Loop
    fold_results = []
    all_fold_detailed_results = []
    fold_detailed_info = []

    # Riprende da fold specifico (in caso di interruzione) 0 fa l'intero ciclo
    start_fold = 0  

    for k in range(start_fold, config.n_splits):
        # Split train/val per fold corrente (mantieni indici originali)
        val_df = all_labels_df[all_labels_df['fold'] == k]
        train_df = all_labels_df[all_labels_df['fold'] != k]
        
        # Addestra fold
        best_f1, detailed_results_df, train_size, val_size, class_report = train_single_fold(k, train_df, val_df, config)
        fold_results.append(best_f1)
        
        # Raccogli risultati dettagliati
        if not detailed_results_df.empty:
            all_fold_detailed_results.append(detailed_results_df)
        
        fold_detailed_info.append({
            'fold': k,
            'f1_score': best_f1,
            'train_samples': train_size,
            'val_samples': val_size,
            'classification_report': class_report
        })

    # AGGREGAZIONE RISULTATI E REPORT FINALI
    print(f"\n{'='*60}")
    print(f" CROSS-VALIDATION COMPLETATA")
    print(f"{'='*60}")
    print(f"F1-Score per fold: {fold_results}")
    
    mean_f1, std_f1 = save_summary_report(fold_results, fold_detailed_info, config)
    print(f"\n Performance media: F1-Score = {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == '__main__':
    train_model()