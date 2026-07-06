import torch
import torch.nn as nn

class CNN_Pasteurizer(nn.Module):
    def __init__(self, n_sensors, n_classes=3, dropout_prob=0.5):
        super(CNN_Pasteurizer, self).__init__()
        
        # Backbone (Extractor de características espaciotemporales)
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv1d(n_sensors, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            
            # Bloque 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            
            # Bloque 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            
            # Bloque 4 (Consolidación)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            
            # Global Pooling: Colapsa el eje temporal a 1 único vector descriptivo
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.dropout_final = nn.Dropout(dropout_prob)
        
        # 4 Cabezas de Clasificación Independientes (One-vs-Rest interno)
        self.head_fouling = nn.Linear(256, n_classes)
        self.head_valvula = nn.Linear(256, n_classes)
        self.head_bomba = nn.Linear(256, n_classes)
        self.head_acumulador = nn.Linear(256, n_classes)

    def forward(self, x):
        # x shape: (Batch, Canales, Pasos_Temporales)
        x = self.features(x)
        x = x.squeeze(-1) # Quitar dimensión tiempo: (Batch, 256, 1) -> (Batch, 256)
        x = self.dropout_final(x)
        
        # Retorna 4 tensores de logits
        return self.head_fouling(x), self.head_valvula(x), self.head_bomba(x), self.head_acumulador(x)

class PhysicsGuidedLoss(nn.Module):
    def __init__(self, feature_cols, scaler, lambda_phys=1.0):
        super(PhysicsGuidedLoss, self).__init__()
        self.feature_cols = feature_cols
        self.lambda_phys = lambda_phys
        
        # Buffers para desnormalizar en GPU
        self.register_buffer('means', torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer('scales', torch.tensor(scaler.scale_, dtype=torch.float32))
        
        # Índices
        try:
            self.idx_ps1 = feature_cols.index('PS1_rmean') 
            self.idx_eps1 = feature_cols.index('EPS1_rmean')
            self.idx_fs1 = feature_cols.index('FS1_rmean') 
            self.idx_ts1 = feature_cols.index('TS1_rmean') 
            self.idx_ts2 = feature_cols.index('TS2_rmean') 
        except ValueError:
            self.idx_ps1 = feature_cols.index('PS1')
            self.idx_eps1 = feature_cols.index('EPS1')
            self.idx_fs1 = feature_cols.index('FS1')
            self.idx_ts1 = feature_cols.index('TS1')
            self.idx_ts2 = feature_cols.index('TS2')

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets, inputs_tensor):
        # 1. Recuperar Valores Reales
        inputs_mean = inputs_tensor.mean(dim=2) 
        X_real = inputs_mean * self.scales + self.means
        
        P_in_bar = X_real[:, self.idx_ps1]
        Q_lmin = X_real[:, self.idx_fs1]
        Power_W = X_real[:, self.idx_eps1]
        T_in = X_real[:, self.idx_ts1]  
        T_out = X_real[:, self.idx_ts2] 

        # REGLA 1: BOMBA (Hidráulica)
        P_hyd_est = P_in_bar * Q_lmin * 1.6666
        efficiency = P_hyd_est / (torch.abs(Power_W) + 1.0)
        
        # Si Eff < 0.75, penalizamos predecir "Sano" (clase 0)
        prob_sano_bomba = torch.softmax(preds[2], dim=1)[:, 0]
        violation_pump = torch.relu(0.75 - efficiency) ** 2 
        loss_physics_pump = torch.mean(violation_pump * prob_sano_bomba) * 100.0

        # REGLA 2: PASTEURIZADOR (Termodinámica)
        m_dot = Q_lmin * 0.017
        delta_T = T_out - T_in 
        Heat_Transfer_kW = m_dot * 3.9 * delta_T
        Q_expected = m_dot * 3.9 * 2.0 
        
        thermal_efficiency = Heat_Transfer_kW / (Q_expected + 0.1)
        
        # Si rinde menos del 80% de lo esperado -> CASTIGO a "Sano"
        prob_sano_cooler = torch.softmax(preds[0], dim=1)[:, 0]
        violation_cooler = torch.relu(0.80 - thermal_efficiency) ** 2
        loss_physics_cooler = torch.mean(violation_cooler * prob_sano_cooler) * 100.0

        # LOSS TOTAL
        supervised_loss = self.ce_loss(preds[0], targets[:, 0]) + \
                          self.ce_loss(preds[1], targets[:, 1]) + \
                          self.ce_loss(preds[2], targets[:, 2]) + \
                          self.ce_loss(preds[3], targets[:, 3])

        total_loss = supervised_loss + (self.lambda_phys * (loss_physics_pump + loss_physics_cooler))
        
        return total_loss, supervised_loss.item(), loss_physics_pump.item(), loss_physics_cooler.item()
