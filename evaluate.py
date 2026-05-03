import os
import warnings
import time
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall
from src.agents.graph import RAGGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from ragas.run_config import RunConfig
from langchain_core.outputs import LLMResult, Generation
from dotenv import load_dotenv

# 1. Suppress noisy warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# Global lock to ensure ONLY ONE Gemini call happens at a time across the whole process
gemini_lock = asyncio.Lock()

# 2. Custom Wrapper for LLM with strictly sequential locking
class KnowledgeRagasLLM(BaseRagasLLM):
    def __init__(self, llm):
        self.langchain_llm = llm
        
    def get_temperature(self, temperature=None):
        return 0

    def generate_text(self, prompt, n=1, temperature=1e-8, callbacks=None, **kwargs):
        return asyncio.run(self.agenerate_text(prompt, n, temperature, callbacks, **kwargs))

    async def generate(self, prompts, n=1, temperature=1e-8, callbacks=None, **kwargs):
        generations = []
        for prompt in prompts:
            text = await self.agenerate_text(prompt, n, temperature, callbacks, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    async def agenerate_text(self, prompt, n=1, temperature=1e-8, callbacks=None, **kwargs):
        async with gemini_lock:
            await asyncio.sleep(15) # Maximum safety delay for free tier
            
            # Robust prompt extraction
            if hasattr(prompt, "to_string"):
                prompt_text = prompt.to_string()
            elif hasattr(prompt, "text"):
                prompt_text = prompt.text
            elif isinstance(prompt, str):
                prompt_text = prompt
            else:
                prompt_text = str(prompt)

            # Debug logging
            with open("eval_log.txt", "a") as f:
                f.write(f"\n--- RAW PROMPT OBJECT TYPE ---\n{type(prompt)}\n")
                f.write(f"\n--- EXTRACTED PROMPT TEXT ---\n{prompt_text}\n")
            
            # We use the extracted text to create a clean message for Gemini
            res = await self.langchain_llm.ainvoke(prompt_text, stop=kwargs.get("stop"))
            
            with open("eval_log.txt", "a") as f:
                f.write(f"\n--- RESPONSE ---\n{res.content}\n")
                
            return res.content

# 3. Custom Wrapper for Embeddings with strictly sequential locking
class KnowledgeRagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_query(self, text):
        return asyncio.run(self.embed_texts([text], is_query=True))[0]

    def embed_documents(self, texts):
        return asyncio.run(self.embed_texts(texts, is_query=False))

    async def embed_texts(self, texts, is_query=False):
        results = []
        for text in texts:
            async with gemini_lock:
                await asyncio.sleep(15) # Wait between every single embedding call
                if is_query:
                    res = await self.embeddings.aembed_query(text)
                else:
                    res = await self.embeddings.aembed_documents([text])
                    res = res[0]
                results.append(res)
        return results

def run_evaluation():
    load_dotenv()
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    # 4. Configure Gemini with safe sequential wrappers
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    evaluator_llm = KnowledgeRagasLLM(llm)
    evaluator_embeddings = KnowledgeRagasEmbeddings(emb)

    print("--- Initializing Knowledge RAG System ---")
    mlp_model_path = "models/router_mlp.pth"
    rag_system = RAGGraph(mlp_model_path)

    # 5. Define Test Suite (Stronger, 10-Question Set)
    test_questions = [
        # --- Science Domain ---
        {
            "question": "What mathematical distribution is used to model photovoltaic module lifespan?",
            "is_help": False,
            "ground_truth": "The two-parameter Weibull distribution is used to model photovoltaic module lifespan based on thermal and humidity stress."
        },
        {
            "question": "Why is MPPT more efficient than PWM controllers in cold weather?",
            "is_help": False,
            "ground_truth": "MPPT is more efficient because it tracks the maximum power point, delivering 15-30% more power, especially in cold weather when panel voltage significantly exceeds battery voltage."
        },
        {
            "question": "Define the Sensible Heat Ratio (SHR) and its importance.",
            "is_help": False,
            "ground_truth": "SHR is the ratio of room sensible heat to room total heat. It determines if a cooling coil can meet both temperature and dehumidification requirements."
        },
        # --- Software Domain ---
        {
            "question": "What is the peak shaving logic for the Smart Wallbox?",
            "is_help": False,
            "ground_truth": "The logic is to decrease the charge current if the total household load exceeds 7kW."
        },
        {
            "question": "How does the HEMS middleware identify individual appliance energy signatures?",
            "is_help": False,
            "ground_truth": "It uses smart outlets utilizing the WI-SUN Home Area Network (HAN) specification for intrusive load monitoring."
        },
        {
            "question": "What is the specific API command string to set a Nest thermostat mode?",
            "is_help": False,
            "ground_truth": "The command string is 'sdm.devices.commands.ThermostatMode.SetMode'."
        },
        # --- User Domain ---
        {
            "question": "What are the NFPA 855 clearance requirements for BESS?",
            "is_help": False,
            "ground_truth": "A minimum 3-foot (36-inch) clearance from combustibles must be maintained."
        },
        {
            "question": "How should a user respond to a lithium-ion battery fire?",
            "is_help": False,
            "ground_truth": "Use water for lithium-ion fires because lithium salts are non-reactive with water."
        },
        {
            "question": "What is the recommended relative humidity range for user comfort?",
            "is_help": False,
            "ground_truth": "The recommended 'Comfort Zone' is between 40% and 60% relative humidity."
        },
        # --- Cross-Domain / Ambiguous ---
        {
            "question": "How do environmental variables affect photovoltaic efficiency according to the research data?",
            "is_help": True, # Forcing help section override logic check
            "ground_truth": "Efficiency degrades by 0.4% to 0.5% for every degree Celsius above 25°C, and high humidity (85% RH at 85°C) can cause circuitry corrosion."
        }
    ]

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print(f"--- Running Inference on {len(test_questions)} questions (with Rate Limit Safety) ---")
    for i, q in enumerate(test_questions):
        print(f"[{i+1}/{len(test_questions)}] Querying: {q['question']}")
        
        inputs = {"query": q["question"], "history": [], "is_help_section": q.get("is_help", False)}
        result = rag_system.app.invoke(inputs)
        
        data["question"].append(q["question"])
        data["answer"].append(result["response"])
        data["contexts"].append([c["content"] for c in result["retrieved_chunks"]])
        data["ground_truth"].append(q.get("ground_truth", ""))
        
        if i < len(test_questions) - 1:
            print("Waiting 15s for rate limit safety...")
            time.sleep(15)

    dataset = Dataset.from_dict(data)
    
    print("--- Starting Ragas Evaluation (Strict Sequential Mode) ---")
    # Timeout increased to accommodate the slow-walk processing
    run_config = RunConfig(max_workers=1, timeout=900)
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )
    
    print("\n--- Evaluation Results ---")
    print(result)
    return result

if __name__ == "__main__":
    run_evaluation()
