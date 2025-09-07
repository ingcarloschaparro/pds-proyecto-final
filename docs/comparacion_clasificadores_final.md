
#  Comparación Final de Clasificadores PLS

##  Métricas de Rendimiento

| Modelo | F1 Macro | Accuracy | Precision PLS | Recall PLS |
|--------|----------|----------|---------------|------------|
| **TF-IDF + LogReg** | 0.8679 | 0.8732 | 0.7971 | 0.8910 |
| **DistilBERT** | 0.8702 | 0.8775 | N/A | N/A |

## ⏱ Tiempos de Respuesta

| Modelo | Tiempo (seg) | Ventaja |
|--------|--------------|---------|
| **TF-IDF + LogReg** | 0.0020 |  50x más rápido |
| **DistilBERT** | 0.1000 |  Más lento |

##  Requerimientos de Recursos

| Modelo | Tamaño | GPU Requerida | Memoria |
|--------|--------|---------------|---------|
| **TF-IDF + LogReg** | ~50MB | No | 50-100MB |
| **DistilBERT** | ~268MB | Recomendado | 1-2GB |

##  Análisis Comparativo

| Aspecto | TF-IDF + LogReg | DistilBERT |
|---------|-----------------|------------|
| Rendimiento |  (0.8679 F1) |  (0.8702 F1) |
| Velocidad |  (0.002s) |  (0.100s) |
| Recursos |  (50MB) |  (268MB) |
| Simplicidad |  (Simple) |  (Complejo) |
| Interpretabilidad |  (Alta) |  (Media) |
| Escalabilidad |  (Excelente) |  (Buena) |

##  Recomendación Final

###  RECOMENDADO: TF-IDF + LOGISTIC REGRESSION

**Justificación basada en métricas objetivas:**

####  Rendimiento Suficiente:
- **F1 Macro**: 0.8679 (supera objetivo ≥0.80)
- **Accuracy**: 87.32%
- **Recall PLS**: 89.10% (muy bueno para detectar PLS)

####  Ventajas Prácticas:
- **50x más rápido** en inferencia (0.002s vs 0.100s)
- **Modelo 5x más pequeño** (50MB vs 268MB)
- **No requiere GPU**
- **Más fácil de desplegar** y mantener
- **Perfecto para experimentación** con MLflow

####  Suficiente para la Tarea:
- Supera ampliamente los criterios de éxito
- Rendimiento comparable a DistilBERT
- Ideal para clasificación de textos biomédicos

###  Conclusión:
**TF-IDF + Logistic Regression es la mejor opción para la Fase 2** por su combinación óptima de rendimiento, velocidad y simplicidad operacional.

** LISTO PARA:** Experimentación con MLflow, Dashboard y Producción

---
*Reporte generado automáticamente - Fecha: $(date)*
