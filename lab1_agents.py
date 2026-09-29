import os
import sys
from dotenv import load_dotenv

load_dotenv()

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace
from deepeval.contextvars import get_current_golden

# Importamos tu agente instrumentado
from agent_instrumented import support_agent as _support_agent

# Wrapper para inyectar expectativas (Golden) en la traza activa de DeepEval
@observe(name="support_agent_evaluation")
def test_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)

    return _support_agent(user_input)

# 1. Instanciamos las métricas agénticas
# TaskCompletion evalúa con GPT-4o si se resolvió la intención
task_completion = TaskCompletionMetric(threshold=0.7, model="gpt-4o")

# ToolCorrectness audita si llamó a la herramienta esperada con los argumentos correctos
tool_correctness = ToolCorrectnessMetric()

# 2. Diseñamos el dataset de prueba (Goldens)
dataset = EvaluationDataset(goldens=[
    Golden(
        input="Where is my order ORD-1042?",
        expected_tools=[ToolCall(name="get_order_status")]
    ),
    Golden(
        input="Can I return food items?",
        expected_tools=[ToolCall(name="get_refund_policy")]
    )
])

# 3. Ejecutamos la evaluación iterativa
if __name__ == "__main__":
    print("\n🚀 Corriendo Laboratorio 1: Evaluación de Agente (Herramientas + Resolución)...\n")
    for golden in dataset.evals_iterator(metrics=[task_completion, tool_correctness]):
        test_agent(golden.input)
    print("\n✅ Evaluación completada. Revisa tu consola y tu panel de Confident AI.")