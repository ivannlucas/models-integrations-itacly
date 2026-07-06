import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import joblib
import time
from src.training.model import CNN_Pasteurizer, PhysicsGuidedLoss
from src.utils.logging import get_logger

logger = get_logger(__name__)

def train_model(train_loader, val_loader, test_loader, scaler, feature_cols, ts1_mean_train, config: dict, save_metrics=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Entorno GPU detectado correctamente. Inicializando entrenamiento en CUDA...")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Entorno Apple Silicon GPU detectado. Inicializando entrenamiento en MPS...")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ No se ha detectado GPU. Entrenamiento ejecutándose en CPU. Revisa tu instalación de PyTorch si dispones de hardware acelerador.")

    # Parámetros desde config
    learning_rate = config['training']['learning_rate']
    dropout_rate = config['training']['dropout_rate']
    max_lambda = config['training']['max_lambda']
    epochs = config['training']['epochs']
    warmup_epochs = config['training']['warmup_epochs']
    ramp_up_epochs = config['training']['ramp_up_epochs']
    patience = config['training']['patience']

    n_canales = len(feature_cols)
    n_classes = config['training'].get('n_classes', 3)

    # 1. Instanciar Modelo y Optimizador
    model = CNN_Pasteurizer(n_sensors=n_canales, n_classes=n_classes, dropout_prob=dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 2. Instanciar Loss
    criterion_dns = PhysicsGuidedLoss(feature_cols=feature_cols, scaler=scaler, lambda_phys=1.0).to(device)

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    logger.info(f"Entrenando: Fase 1 (0-{warmup_epochs}) solo datos, Fase 2 ({warmup_epochs}-{warmup_epochs+ramp_up_epochs}) datos + ponderación lineal física, Fase 3 (>{warmup_epochs+ramp_up_epochs}) datos + física final")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_time = time.time() 

    for epoch in range(epochs):
        train_correct, train_total = 0, 0
        # Curriculum: Gestión de Lambda
        if epoch < warmup_epochs:
            current_lambda = 0.0
        elif epoch < (warmup_epochs + ramp_up_epochs):
            progress = (epoch - warmup_epochs) / ramp_up_epochs
            current_lambda = progress * max_lambda
        else:
            current_lambda = max_lambda
            
        criterion_dns.lambda_phys = current_lambda
        
        # TRAIN Phase
        model.train()
        run_total, run_sup, run_phys = 0.0, 0.0, 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            preds = model(inputs)
            p_foul, p_valv, p_bomb, p_acum = preds
            loss, l_sup, l_pump, l_cool = criterion_dns(preds, labels, inputs)
            
            loss.backward()
            optimizer.step()
            
            run_total += loss.item()
            run_sup += l_sup
            run_phys += (l_pump + l_cool)

            # Accuracy on Train
            _, pred_f = torch.max(p_foul, 1)
            _, pred_v = torch.max(p_valv, 1)
            _, pred_b = torch.max(p_bomb, 1)
            _, pred_a = torch.max(p_acum, 1)
            rows_match = (pred_f == labels[:, 0]) & (pred_v == labels[:, 1]) & \
                         (pred_b == labels[:, 2]) & (pred_a == labels[:, 3])
            train_correct += rows_match.sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = run_total / len(train_loader) 
        train_acc = train_correct / train_total
        
        # VALID Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        criterion_val = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                p_foul, p_valv, p_bomb, p_acum = model(inputs)
                
                l1 = criterion_val(p_foul, labels[:, 0])
                l2 = criterion_val(p_valv, labels[:, 1])
                l3 = criterion_val(p_bomb, labels[:, 2])
                l4 = criterion_val(p_acum, labels[:, 3])
                val_loss += (l1+l2+l3+l4).item()
                
                _, pred_f = torch.max(p_foul, 1)
                _, pred_v = torch.max(p_valv, 1)
                _, pred_b = torch.max(p_bomb, 1)
                _, pred_a = torch.max(p_acum, 1)
                
                rows_match = (pred_f == labels[:, 0]) & (pred_v == labels[:, 1]) & \
                             (pred_b == labels[:, 2]) & (pred_a == labels[:, 3])
                correct += rows_match.sum().item()
                total += labels.size(0)
                
        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Early Stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            
        if (epoch+1) % 5 == 0 or epoch == 0:
            logger.info(f"Ep {epoch+1:03d}/{epochs} | Lam: {current_lambda:.2f} | "
                        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
        if counter >= patience and epoch > warmup_epochs:
            logger.info(f"Early Stopping activado en época {epoch+1}")
            break

    elapsed_time = time.time() - start_time
    mins, secs = int(elapsed_time // 60), int(elapsed_time % 60)
    logger.info(f"Entrenamiento finalizado en {mins}m {secs}s. Restableciendo mejores pesos...")
    
    model.load_state_dict(best_model_wts)
    
    # Save artifacts
    ruta_modelos = config['paths']['models_dir']
    os.makedirs(ruta_modelos, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(ruta_modelos, "neurosymbolic_cnn.pth"))
    joblib.dump(scaler, os.path.join(ruta_modelos, "scaler_cnn_dns.pkl"))
    joblib.dump(feature_cols, os.path.join(ruta_modelos, "feature_columns.pkl"))
    joblib.dump(ts1_mean_train, os.path.join(ruta_modelos, "ts1_mean_train.pkl"))
    
    logger.info(f"Artefactos de entrenamiento guardados en {ruta_modelos}/")
    
    # Guardar Curvas de Aprendizaje
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.title('Pérdida/Loss', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='#2ca02c', linewidth=2)
    plt.plot(history['val_acc'], label='Val Acc', color='#d62728', linewidth=2)
    plt.title('Precisión/Accuracy', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    ruta_metricas = config['paths'].get('metrics_dir', 'models/metrics')
    os.makedirs(ruta_metricas, exist_ok=True)
    curvas_path = os.path.join(ruta_metricas, "learning_curves.png")
    plt.savefig(curvas_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfica de Curvas de Aprendizaje guardada en {curvas_path}")
    
    # ---------------------------------------------------------
    # EVALUACIÓN EN TEST Y GUARDADO DE MÉTRICAS
    # ---------------------------------------------------------
    if save_metrics and test_loader is not None:
        logger.info("Generando métricas sobre el conjunto de test...")
        import numpy as np
        import pandas as pd
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
        
        y_true_list = []
        y_pred_list = []
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.cpu().numpy()

                p_foul, p_valv, p_bomb, p_acum = model(inputs)
                preds_batch = torch.stack([
                    p_foul.argmax(1), p_valv.argmax(1), p_bomb.argmax(1), p_acum.argmax(1)
                ], dim=1).cpu().numpy()

                y_true_list.append(labels)
                y_pred_list.append(preds_batch)

        y_true = np.vstack(y_true_list)
        y_pred = np.vstack(y_pred_list)
        
        target_names = ['Fouling', 'Válvula', 'Bomba', 'Acumulador']
        class_labels = ['Sano', 'Warning', 'Crítico']
        
        # Calcular Exact Match Ratio global (todas las etiquetas correctas a la vez)
        exact_matches = np.all(y_pred == y_true, axis=1)
        exact_match_ratio = np.mean(exact_matches)
        logger.info(f"Exact Match Ratio Global en Test: {exact_match_ratio:.2%}")
        
        metrics_data = []
        for i, component in enumerate(target_names):
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true[:, i], y_pred[:, i], labels=[0, 1, 2], zero_division=0
            )
            # Extraer matriz de confusión en array plano (9 valores)
            cm_matrix = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1, 2])
            cm = cm_matrix.flatten().tolist()
            
            # Generar Gráfica de la Matriz de Confusión
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_labels, yticklabels=class_labels,
                        annot_kws={"size": 22, "weight": "bold"}, cbar_kws={'shrink': 0.8})
            plt.title(f'Matriz de Confusión: {component}', fontsize=16, fontweight='bold', pad=15)
            plt.ylabel('Clase Real', fontsize=14, fontweight='bold')
            plt.xlabel('Predicción del Modelo', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12, rotation=0)
            
            cm_path = os.path.join(ruta_metricas, f"confusion_matrix_{component}.png")
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Matriz de Confusión visual guardada en {cm_path}")
            
            for cls_idx, cls_name in enumerate(class_labels):
                metrics_data.append({
                    'Componente': component,
                    'Clase': cls_name,
                    'Accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
                    'Precision': precision[cls_idx],
                    'Recall': recall[cls_idx],
                    'F1-Score': f1[cls_idx],
                    'Support': support[cls_idx],
                    'Confusion_Matrix_Flat': cm,  # Matriz aplanada completa como ref
                    'Exact_Match_Global': exact_match_ratio
                })

        df_metrics = pd.DataFrame(metrics_data)
        
        csv_metricas = os.path.join(ruta_metricas, "test_metrics_confusion.csv")
        df_metrics.to_csv(csv_metricas, index=False)
        logger.info(f"Métricas y Matriz de confusión guardadas en {csv_metricas}")
        
    return model
