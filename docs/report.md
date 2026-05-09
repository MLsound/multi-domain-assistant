!["Facultad de Ingeniería de la UBA"](LogoFIUBA.jpg)

# Sistema RAG Agéntico Multi-Agente (Knowledge Assistant)
## Procesamiento del Lenguaje Natural III _(NLP3)_
### Carrera de Maestría en Inteligencia Artificial (UBA) - _Cohorte 2_

Este repositorio contiene el informe del proyecto final desarrollado para la materia ***Procesamiento del Lenguaje Natural III (PLN3)***. El trabajo se centra en el diseño e implementación de **Knowledge Assistant**, un sistema de **RAG Agéntico** (Retrieval-Augmented Generation) que trasciende las arquitecturas tradicionales mediante el uso de orquestación basada en grafos de estados (**LangGraph**).

---

### Información
- Autores (Grupo 1):
  - Alejandro Lloveras _(a1716)_ — alejandro.lloveras@gmail.com
  - Fabian Sarmiento _(2672002)_ — fsarmiento1805@gmail.com
  - Jorge Cuenca _(a0805)_ — jorge.cuenca@unillanos.edu.co
- Docentes: 
  - Oksana Bokhonok — bokhonokok@gmail.com
  - Abraham Rodriguez — abraham.rodz17@gmail.com

---

## Índice de Contenidos
1. **[Resumen Ejecutivo](#1.-resumen-ejecutivo)**
2. **[Introducción](#2-introduccion)**
  * 2.1 [Planteamiento del Problema](21-planteamiento-del-problema)
  * 2.2 [Objetivo](#22-objetivo)

3. **[Arquitectura del Sistema](3-arquitectura-del-sistema)**
  * 3.1 [Orquestación de Estados (LangGraph)](#31-orquestacion-de-estados-langgraph)
  * 3.2 [Desglose de Agentes y Componentes](#32-desglose-de-agentes-y-componentes)
    * *Enrutador Basado en MLP*
    * *Retriever de Dos Etapas*
    * *Agente Crítico (Faithfulness Loop)*
    * *Agente de Síntesis*
    * *Agentes de Seguridad (Guard)*
    * *Agente de Caché Semántico*
    * *Agente de Acción*

4. **[Implementación Técnica](#4-implementacion-tecnica)**
  * 4.1 [Dominio de Aplicación (Energía Sostenible y Smart Buildings)](#41-dominio-de-aplicacion)
  * 4.2 [Interfaz y Despliegue (FastAPI y Modelos Cuantizados)](#42-interfaz-y-despliegue)

5. **[Evaluación y Resultados](#5-evaluacion-y-resultados)**
  * *Métricas RAGAS (Fidelidad, Relevancia, Precisión, Recall)*

6. **[Conclusión](#6-conclusion)**

## Índice de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Introducción](#2-introducción)
   - [Planteamiento del Problema](#21-planteamiento-del-problema)
   - [Objetivo](#22-objetivo)

3. [Arquitectura del Sistema](#3-arquitectura-del-sistema)
   - [Orquestación de Estados (LangGraph)](#31-orquestación-de-estados-langgraph)
   - [Desglose de Agentes y Componentes](#32-desglose-de-agentes-y-componentes)

4. [Implementación Técnica](#4-implementación-técnica)
   - [Dominio de Aplicación](#41-dominio-de-aplicación)
   - [Interfaz y Despliegue](#42-interfaz-y-despliegue)

5. [Evaluación y Resultados](#5-evaluación-y-resultados)

6. [Conclusión](#6-conclusión)

---

## 1. Resumen Ejecutivo

Este informe detalla el diseño e implementación de **Knowledge Assistant**, un framework de Generación Aumentada por Recuperación (RAG) agéntico y agnóstico al dominio. Desarrollado por el **Grupo 1**, el sistema utiliza una capa de orquestación de grafo de estados (**LangGraph**) y se expone a través de una interfaz REST con **FastAPI**. El sistema aborda las limitaciones inherentes del RAG tradicional —como las alucinaciones y la baja precisión en la recuperación— mediante una arquitectura transparente y modular que aplica procesos de crítica y reintento automatizados.

## 2. Introducción

### 2.1 Planteamiento del Problema

El desarrollo de sistemas RAG fiables enfrenta desafíos críticos en la precisión del enrutamiento de consultas, la veracidad de las respuestas y la seguridad del pipeline de datos. Actualmente, existe una tendencia a utilizar frameworks de alto nivel que operan como "cajas negras", lo que dificulta el control granular sobre el flujo de decisión. Esta opacidad incrementa el riesgo de alucinaciones e ineficiencias en la búsqueda dentro de corpus heterogéneos.

### 2.2 Objetivo

El objetivo principal es construir una arquitectura propia que garantice:

1. **Fidelidad del contenido:** Mediante un bucle de validación y crítica.
2. **Transparencia:** Evitando flujos preconfigurados de terceros.
3. **Eficiencia:** Optimizando la latencia y los costos de cómputo mediante enrutamiento especializado y caché semántico.

## 3. Arquitectura del Sistema

El sistema se basa en una arquitectura de **7 agentes** desacoplados, donde cada uno cumple un rol específico dentro de un ciclo de vida de consulta orquestado explícitamente.

### 3.1 Orquestación de Estados (LangGraph)

A diferencia de los pipelines lineales, el uso de un grafo de estados permite:

* **Aristas Condicionales:** Transiciones lógicas basadas en la salida de cada agente.
* **Ciclos de Retroalimentación:** Permite que el flujo regrese a etapas anteriores si la calidad de la información es insuficiente.
* **Observabilidad:** Control total sobre cada paso del proceso de razonamiento.

### 3.2 Desglose de Agentes y Componentes

* **Enrutador Basado en MLP:** En lugar de utilizar llamadas costosas a un LLM para la clasificación de intención, se implementa un clasificador de red neuronal (**MLP**) propio entrenado en **PyTorch**. Este dirige las consultas hacia el dominio de conocimiento más relevante basándose en probabilidades.
* **Retriever de Dos Etapas:** Combina una búsqueda vectorial ponderada en **Qdrant** con un *re-ranker* de tipo **cross-encoder** para asegurar que solo los fragmentos más pertinentes lleguen al modelo de generación.
* **Agente Crítico (Faithfulness Loop):** Un nodo especializado que evalúa la fidelidad de la respuesta generada respecto al contexto recuperado. Si detecta inconsistencias, activa un bucle de re-recuperación con una consulta refinada.
* **Agente de Síntesis:** Genera la respuesta final utilizando técnicas de **In-Context Learning (ICL)** y razonamiento **Chain-of-Thought (CoT)**.
* **Agentes de Seguridad (Guard):** Capas de validación de entrada y salida que detectan patrones de inyección de prompts o fugas de información sensible.
* **Agente de Caché Semántico:** Una base de datos vectorial que almacena consultas recurrentes. Si una nueva consulta presenta una similitud de coseno superior al umbral definido, el sistema responde instantáneamente evitando el flujo completo.
* **Agente de Acción:** Encargado de ejecutar funciones externas o llamadas a APIs cuando la consulta lo requiere.

## 4. Implementación Técnica

### 4.1 Dominio de Aplicación

Aunque el framework es agnóstico, se implementa para el dominio de **energía sostenible y edificios inteligentes** como demostración. Esto incluye la integración de pipelines de datos meteorológicos (como el sistema **Intel** para datos **ERA5-Land**) para proporcionar contexto ambiental en tiempo real.

### 4.2 Interfaz y Despliegue

* **API REST:** Implementada en **FastAPI**, proporcionando un punto de acceso profesional y robusto.
* **Monitoreo:** Incluye métricas de rendimiento y registros de auditoría automáticos para cada paso del grafo.
* **Optimización Local:** Soporte para modelos cuantizados (GGUF/AWQ), permitiendo la ejecución eficiente en hardware local.

## 5. Evaluación y Resultados

El sistema se evalúa bajo el framework **RAGAS**, centrándose en cuatro pilares:

1. **Fidelidad (Faithfulness):** Verificación de que la respuesta provenga estrictamente del contexto.
2. **Relevancia de la Respuesta:** Qué tan bien satisface la intención del usuario.
3. **Precisión del Contexto:** Relación señal-ruido en los fragmentos recuperados.
4. **Exhaustividad del Contexto (Recall):** Asegurar que toda la información necesaria fue recuperada.

## 6. Conclusión

El proyecto **Knowledge Assistant** del Grupo 1 demuestra que una aproximación agéntica basada en grafos mejora significativamente la fiabilidad de los LLMs en entornos profesionales. Al implementar cada componente desde cero y evitar herramientas de "caja negra", se logra un sistema altamente auditable, seguro y eficiente, alineado con las exigencias académicas de la materia **PLN3**.