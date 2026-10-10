"""Demo del agente: razonamiento multi-paso, memoria por thread_id y traza.

Uso:
    python main.py                                  # demo completa + traza en traza_ejecucion.json/.log
    python main.py --chat --thread-id ana           # chat interactivo; la sesión persiste en checkpoints.db
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agent import RECURSION_LIMIT, build_graph, create_llm

DB_PATH = "checkpoints.db"
TRACE_JSON = Path("traza_ejecucion.json")
TRACE_LOG = Path("traza_ejecucion.log")

# (thread_id, pregunta). Los dos primeros turnos comparten thread: el segundo
# ("¿y el último?") solo se entiende si el agente recuerda el primero.
ESCENARIO = [
    ("demo-102", "¿Cuántos pedidos tuvo el cliente 102, cuál fue el total y qué productos incluía el último pedido?"),
    ("demo-102", "¿y el anterior a ese?"),
    ("demo-999", "¿Cuántos pedidos tuvo el cliente 999?"),
    ("demo-nuevo", "¿y el último?"),  # thread distinto: no hay contexto previo
]


def _describir(mensaje: BaseMessage) -> dict[str, Any] | None:
    """Convierte un mensaje a un paso legible de la traza ReAct."""
    if isinstance(mensaje, AIMessage):
        if mensaje.tool_calls:
            return {
                "paso": "accion",
                "descripcion": "El agente decide llamar herramientas",
                "tool_calls": [
                    {"herramienta": c["name"], "argumentos": c["args"]}
                    for c in mensaje.tool_calls
                ],
            }
        return {"paso": "respuesta", "contenido": mensaje.content}
    if isinstance(mensaje, ToolMessage):
        return {
            "paso": "observacion",
            "herramienta": mensaje.name,
            "resultado": mensaje.content,
        }
    return None


async def ejecutar_turno(graph, thread_id: str, pregunta: str) -> dict[str, Any]:
    """Corre un turno en streaming y devuelve la traza de ese turno."""
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    pasos: list[dict[str, Any]] = []
    llamadas_a_herramientas = 0
    respuesta = ""
    error = None
    try:
        async for update in graph.astream(
            {"messages": [("user", pregunta)]}, config, stream_mode="updates"
        ):
            for _nodo, salida in update.items():
                for mensaje in salida.get("messages", []):
                    paso = _describir(mensaje)
                    if paso is None:
                        continue
                    pasos.append(paso)
                    if paso["paso"] == "accion":
                        llamadas_a_herramientas += len(paso["tool_calls"])
                    if paso["paso"] == "respuesta":
                        respuesta = paso["contenido"]
    except GraphRecursionError:
        error = (
            f"Se alcanzó el límite de {RECURSION_LIMIT} pasos (recursion_limit) "
            "sin llegar a una respuesta."
        )
    return {
        "thread_id": thread_id,
        "usuario": pregunta,
        "pasos": pasos,
        "llamadas_a_herramientas": llamadas_a_herramientas,
        "respuesta": respuesta,
        "error": error,
    }


def formatear_log(turnos: list[dict[str, Any]]) -> str:
    lineas: list[str] = []
    for t in turnos:
        lineas += ["=" * 70, f"[thread_id={t['thread_id']}]", f"Usuario: {t['usuario']}"]
        for p in t["pasos"]:
            if p["paso"] == "accion":
                for c in p["tool_calls"]:
                    args = ", ".join(f"{k}={v!r}" for k, v in c["argumentos"].items())
                    lineas.append(f"  -> Acción: {c['herramienta']}({args})")
            elif p["paso"] == "observacion":
                lineas.append(f"  <- Observación ({p['herramienta']}): {p['resultado']}")
        if t["error"]:
            lineas.append(f"  !! {t['error']}")
        else:
            lineas.append(f"Respuesta: {t['respuesta']}")
        lineas.append(f"(herramientas invocadas en este turno: {t['llamadas_a_herramientas']})")
    return "\n".join(lineas) + "\n"


async def demo() -> None:
    # Arrancamos con una base limpia para que la traza sea reproducible.
    for sufijo in ("", "-wal", "-shm"):
        Path(DB_PATH + sufijo).unlink(missing_ok=True)

    turnos: list[dict[str, Any]] = []
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph(create_llm(), checkpointer)
        for thread_id, pregunta in ESCENARIO:
            turno = await ejecutar_turno(graph, thread_id, pregunta)
            turnos.append(turno)
            print(formatear_log([turno]))

    TRACE_JSON.write_text(
        json.dumps(
            {
                "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "recursion_limit": RECURSION_LIMIT,
                "turnos": turnos,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    TRACE_LOG.write_text(formatear_log(turnos), encoding="utf-8")
    print(f"Traza guardada en {TRACE_JSON} y {TRACE_LOG}")


async def chat(thread_id: str) -> None:
    print(f"Chat con thread_id='{thread_id}' (se guarda en {DB_PATH}). Vacío para salir.")
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph(create_llm(), checkpointer)
        while True:
            pregunta = (await asyncio.to_thread(input, "\nVos: ")).strip()
            if not pregunta:
                break
            turno = await ejecutar_turno(graph, thread_id, pregunta)
            print(turno["error"] or f"Agente: {turno['respuesta']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente ReAct con LangGraph.")
    parser.add_argument("--chat", action="store_true", help="Chat interactivo.")
    parser.add_argument("--thread-id", default="chat-1", help="Id de la sesión (con --chat).")
    args = parser.parse_args()

    load_dotenv()
    try:
        asyncio.run(chat(args.thread_id) if args.chat else demo())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as error:  # clave faltante, red, etc.: mensaje corto, sin traceback
        raise SystemExit(f"Error: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
