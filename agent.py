"""Agente ReAct cíclico con LangGraph.

Grafo:

    START -> modelo --(tools_condition)--> herramientas -> modelo -> ... -> END
                 \\__ sin tool_calls ______________________________________/

- `modelo`: el LLM (con las herramientas enlazadas vía `bind_tools`) decide solo
  si responde o si llama a una herramienta. No hay if/else manual por pregunta.
- `herramientas`: `ToolNode` ejecuta las llamadas y devuelve los resultados como
  `ToolMessage`. Después el flujo vuelve a `modelo` (el ciclo): si el resultado es
  un error o está incompleto, el modelo puede reintentar o pedir aclaraciones.
- El checkpointer guarda el estado por `thread_id`, así el agente recuerda la sesión.
"""

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS

DEFAULT_MODEL = "gpt-4o-mini"

# Techo de pasos del grafo por invocación (evita bucles infinitos y costos
# inesperados). Un caso de 2 herramientas usa ~5 pasos: modelo, tools, modelo,
# tools, modelo.
RECURSION_LIMIT = 10

# Máximo de mensajes que se le envían al modelo (el historial completo sigue en el
# checkpoint). Evita que el contexto crezca sin límite en sesiones largas.
MAX_MENSAJES_CONTEXTO = 20

SYSTEM_PROMPT = (
    "Sos un asistente de soporte que responde consultas sobre pedidos de clientes. "
    "Usá las herramientas disponibles para obtener datos reales: nunca inventes "
    "cifras ni pedidos. Si una pregunta necesita varios datos, llamá a las "
    "herramientas que hagan falta, una tras otra, usando el resultado de la "
    "anterior. Si una herramienta devuelve un error, leelo: corregí el argumento "
    "y reintentá si podés, o pedile al usuario la aclaración que falta. "
    "Respondé en español, de forma breve, con los montos en pesos (ej. $14.500)."
)


class AgentState(MessagesState):
    """Estado del grafo. Hereda `messages` de MessagesState, cuyo reducer
    (`add_messages`) AGREGA los mensajes nuevos en vez de reemplazar la lista."""


def create_llm() -> BaseChatModel:
    """LLM de OpenAI. Lee OPENAI_API_KEY del entorno (.env)."""
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), temperature=0)


def build_graph(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    """Arma y compila el grafo. `llm` es un chat model sin enlazar: acá se le
    hace `bind_tools`. Se recibe como parámetro para poder inyectar otro modelo
    (por ejemplo, uno falso en los tests)."""
    llm_con_herramientas = llm.bind_tools(TOOLS)

    async def modelo(state: AgentState) -> dict[str, Any]:
        # Recorta SOLO lo que ve el modelo; el estado guardado queda completo.
        recientes = trim_messages(
            state["messages"],
            max_tokens=MAX_MENSAJES_CONTEXTO,
            token_counter=len,  # cuenta mensajes, no tokens
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        respuesta = await llm_con_herramientas.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *recientes]
        )
        return {"messages": [respuesta]}

    graph = StateGraph(AgentState)
    graph.add_node("modelo", modelo)
    graph.add_node("herramientas", ToolNode(TOOLS))
    graph.add_edge(START, "modelo")
    # tools_condition -> "tools" si el último mensaje trae tool_calls, si no END.
    graph.add_conditional_edges(
        "modelo", tools_condition, {"tools": "herramientas", END: END}
    )
    graph.add_edge("herramientas", "modelo")  # el ciclo de retorno
    return graph.compile(checkpointer=checkpointer)
