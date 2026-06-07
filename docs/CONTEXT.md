# Application Context: Sustainable Energy & Smart Building Systems

This document explains the specific application context used to validate the **Knowledge Assistant** framework. While the system is designed to be domain-agnostic, this test deployment focuses on the intersection of **Sustainable Energy**, **Photovoltaic (PV) Physics**, and **Home Energy Management Systems (HEMS)**.

## Knowledge Domains in this Test

The assistant is currently indexed with a curated corpus spanning three distinct but related categories:

### 1. Scientific Research (Science Domain)
Focuses on the fundamental physics and reliability of renewable energy systems.
*   **Key Topics**: Photovoltaic yield physics, MPPT (Maximum Power Point Tracking) efficiency, thermal degradation of crystalline silicon, and building thermodynamics (infiltration, SHGC, latent heat).
*   **Role in RAG**: Provides the "ground truth" for physical laws and environmental correlations.

### 2. Software Documentation (Software Domain)
Covers the technical implementation, schemas, and control logic for smart energy devices.
*   **Key Topics**: EV charger control JSON schemas, HEMS (Home Energy Management System) operating modes (ZENKI/AI), Inverter telemetry protocols, and smart thermostat command structures.
*   **Role in RAG**: Provides the technical specifications for developers and system architects.

### 3. User & Safety Manuals (User Domain)
Contains procedural instructions and safety protocols for end-users and technicians.
*   **Key Topics**: NFPA 855 safety standards for Battery Energy Storage Systems (BESS), factory reset procedures, seasonal efficiency checklists, and appliance scheduling best practices.
*   **Role in RAG**: Provides actionable, safety-critical information and configuration steps.

## Cross-Domain Enrichment (Environmental Data)

A unique feature of this test is the **Environmental Metadata Injection**. When the system detects a query related to the **Science** domain, it automatically triggers a mock MCP (Model Context Protocol) tool that provides real-time, dynamic (randomized within logical ranges) environmental variables:
*   **Temperature (2m)**
*   **Humidity**
*   **Total Precipitation**
*   **Wind Speed & Direction**
*   **Air Pressure**
*   **Soil Temperature & Moisture**

This data is injected into the LLM's context alongside the retrieved document chunks, allowing for grounded responses that combine static research data with dynamic environmental conditions.

## Why this Context?
This context was chosen because it represents a "high-stakes" technical environment where:
1.  **Hallucinations are dangerous** (e.g., battery safety or electrical control logic).
2.  **Precision is required** (e.g., calculating efficiency based on specific physical formulas).
3.  **Cross-referencing is essential** (e.g., understanding how environmental physics affects specific software control strategies).
