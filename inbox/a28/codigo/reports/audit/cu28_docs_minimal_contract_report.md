# Auditoria documental minima CU28

- Fecha de ejecucion: `2026-06-26`
- Scope: `mixed_context`
- Checks superados: `9/9`
- Estado: **PASS**

## Checks

| Check | Estado | Detalle |
|---|---|---|
| docs contiene exactamente la whitelist | PASS | whitelist exacta |
| docs/audit no existe | PASS | ausente |
| docs sin nombres legacy/deprecated/old/draft/backup | PASS | sin nombres obsoletos |
| declaraciones contractuales obligatorias | PASS | presentes |
| docs sin resultados o metricas obsoletas hardcodeadas | PASS | sin valores manuales |
| docs sin contradicciones de ruta oficial | PASS | mixed_context coherente |
| reports contiene evidencia oficial y auditorias vigentes | PASS | official=3; audit=6 |
| configuracion del data blob usa el snapshot documental minimo | PASS | snapshot exacto |
| data blob incluye snapshot documental actualizado | PASS | rutas y hashes vigentes |

## Whitelist

- `docs/README.md`
- `docs/reproducibility.md`
- `docs/repository_outputs.md`
- `docs/data_lineage.md`
- `docs/data_sources_registry.md`
- `docs/data_blob_inventory.md`
- `docs/etl_pipeline.md`
- `docs/feature_engineering.md`
- `docs/input_contract.md`
- `docs/output_contract.md`
- `docs/leakage_policy.md`
- `docs/model_card_cu28.md`
- `docs/platform_usage.md`
- `docs/simulation_assumptions.md`
- `docs/simulation_data_basis.md`

`docs/` conserva contrato tecnico estable. `reports/` conserva resultados, auditorias y evidencias generadas.
