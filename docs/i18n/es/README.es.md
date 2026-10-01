# AstroML — Español

<!-- Traducción parcial (partial translation). Solo se han traducido las
secciones iniciales; el resto queda en inglés como referencia. Conserva
idénticos los bloques de código, enlaces e identificadores respecto a
README.md. Ver docs/i18n/README.md para el flujo completo. -->

[![CI](https://github.com/Traqora/astroml/actions/workflows/ci.yml/badge.svg)](https://github.com/Traqora/astroml/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Traqora/astroml/branch/main/graph/badge.svg)](https://codecov.io/gh/Traqora/astroml)
[![Code Complexity](https://img.shields.io/badge/complexity-A-brightgreen)](https://github.com/mombu/xenon)

## Marco de aprendizaje automático sobre grafos dinámicos para la red Stellar

**AstroML** es un framework de Python orientado a la investigación para
construir **modelos de aprendizaje automático sobre grafos dinámicos** en la
blockchain Stellar de la Stellar Development Foundation.

Trata los datos de la blockchain como un **grafo multi-activo y en continua
evolución temporal**, lo que habilita investigación avanzada en ML sobre redes
de transacciones: detección de fraude, detección de anomalías y modelado del
comportamiento.

---

## ✨ Características

AstroML ofrece herramientas de extremo a extremo para:

- Ingesta y normalización del libro mayor (ledger)
- Construcción de grafos de transacciones dinámicos
- Ingeniería de características para cuentas de blockchain
- Redes neuronales de grafos (GNN)
- Embeddings de nodos auto-supervisados
- Detección de anomalías
- Modelado temporal
- Experimentación de ML reproducible
- Registro de modelos con versionado y seguimiento de métricas

---

## 🧠 Idea central

Las redes de blockchain son, por naturaleza, **sistemas con estructura de
grafo**:

| Concepto de blockchain | Representación en el grafo  |
| ---------------------- | --------------------------- |
| Cuentas                | Nodos                       |
| Transacciones          | Aristas dirigidas           |
| Activos                | Tipos de arista             |
| Tiempo                 | Dimensión dinámica          |

La mayoría de las herramientas de análisis dependen de heurísticas estáticas o
consultas SQL.

**AstroML, en cambio, habilita:**

- Aprendizaje dinámico sobre grafos
- GNN temporales
- Aprendizaje de representaciones
- Experimentación de nivel de investigación

---

## 📌 Por traducir (TODO)

Las secciones siguientes aún no están traducidas; toma como referencia el
[`README.md`](../../../README.md) en inglés:

- `## 📦 Model Registry`
- `## 🎯 Target Users`
- `## 🚀 Quick Start`
- `## 📖 Documentation`
- el resto del documento
