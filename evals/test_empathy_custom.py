import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams
from deepeval.tracing import observe
from agent_instrumented import support_agent as _support_agent


@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)


# 1. Definición de la métrica personalizada con G-Eval
empathy_and_tone = GEval(
    name="Empatía y Tono Profesional",
    criteria=(
        "Evalúa si la respuesta del asistente muestra un tono cálido, empático y servicial. "
        "Debe validar educadamente la situación del usuario, brindar la información de forma clara "
        "y evitar sonar frío, tajante o puramente robótico. Penaliza si la respuesta es cortante "
        "o no ofrece disposición para seguir ayudando."
    ),
    evaluation_steps=[
        "Verificar si el asistente inicia con un saludo o tono cordial.",
        "Comprobar si la información solicitada se entrega de forma comprensible.",
        "Verificar si el mensaje incluye una frase de cierre cordial o disposición de ayuda adicional.",
        "Penalizar si el tono es excesivamente plano, insensible o burocrático."
    ],
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT
    ],
    model="gpt-4o-mini",
    threshold=0.75,
)

# 2. Casos de prueba: Uno neutro y uno de cliente frustrado
dataset = EvaluationDataset(goldens=[
    Golden(
        input="I am very upset, my order ORD-1042 was supposed to arrive earlier. Where is it?"
    ),
    Golden(
        input="What is the refund policy for electronics? I hope I don't lose my money."
    )
])

# 3. Ejecución de la evaluación
for golden in dataset.evals_iterator(metrics=[empathy_and_tone]):
    support_agent(golden.input)