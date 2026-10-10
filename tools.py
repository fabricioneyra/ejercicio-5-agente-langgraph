"""Herramientas del agente: simulan una base de datos de pedidos.

El LLM decide qué herramienta usar leyendo ÚNICAMENTE el nombre, los tipos y el
docstring de cada función. Por eso los docstrings dicen qué hace la herramienta,
qué argumentos espera, qué devuelve y cuándo conviene (o no) usarla.
"""

from typing import Any

from langchain_core.tools import tool

# "Base de datos" simulada: cliente_id -> lista de pedidos (del más viejo al más nuevo).
PEDIDOS_POR_CLIENTE: dict[int, list[dict[str, Any]]] = {
    101: [
        {"pedido_id": 5001, "fecha": "2026-08-03", "total": 8200, "estado": "entregado"},
    ],
    102: [
        {"pedido_id": 5010, "fecha": "2026-07-12", "total": 4500, "estado": "entregado"},
        {"pedido_id": 5018, "fecha": "2026-08-20", "total": 6000, "estado": "entregado"},
        {"pedido_id": 5027, "fecha": "2026-09-29", "total": 4000, "estado": "en camino"},
    ],
    103: [],
}

DETALLE_PEDIDOS: dict[int, dict[str, Any]] = {
    5001: {"items": [{"producto": "Teclado mecánico", "cantidad": 1, "precio": 8200}]},
    5010: {"items": [{"producto": "Mouse inalámbrico", "cantidad": 3, "precio": 1500}]},
    5018: {
        "items": [
            {"producto": "Monitor 24 pulgadas", "cantidad": 1, "precio": 5000},
            {"producto": "Cable HDMI", "cantidad": 1, "precio": 1000},
        ]
    },
    5027: {"items": [{"producto": "Webcam HD", "cantidad": 2, "precio": 2000}]},
}


@tool
def buscar_pedidos(cliente_id: int) -> dict[str, Any]:
    """Busca en la base de datos todos los pedidos de UN cliente.

    Usala cuando el usuario pregunte cuántos pedidos tuvo un cliente, cuánto
    gastó en total, o necesites conocer los números de pedido (pedido_id) de ese
    cliente. Requiere el número de cliente (entero, ej. 102).

    Devuelve un diccionario con: cliente_id, cantidad de pedidos, total gastado
    (suma de los pedidos, en pesos argentinos) y la lista de pedidos ordenados del
    más viejo al más nuevo (cada uno con pedido_id, fecha, total y estado). El
    ÚLTIMO pedido es el último de la lista.

    NO devuelve los productos de cada pedido: para eso usá `detalle_pedido` con un
    pedido_id. Si el cliente no existe, devuelve un campo "error" explicando el
    problema.
    """
    if cliente_id not in PEDIDOS_POR_CLIENTE:
        return {
            "error": f"No existe el cliente {cliente_id}.",
            "clientes_validos": sorted(PEDIDOS_POR_CLIENTE),
        }
    pedidos = PEDIDOS_POR_CLIENTE[cliente_id]
    return {
        "cliente_id": cliente_id,
        "cantidad_pedidos": len(pedidos),
        "total_gastado": sum(p["total"] for p in pedidos),
        "pedidos": pedidos,
    }


@tool
def detalle_pedido(pedido_id: int) -> dict[str, Any]:
    """Devuelve el detalle de UN pedido puntual: los productos que incluye.

    Usala cuando el usuario pregunte qué compró un cliente, qué contenía un pedido
    o cuántas unidades se pidieron. Requiere el número de pedido (pedido_id, ej.
    5027), que NO es lo mismo que el número de cliente: si solo tenés el cliente,
    primero llamá a `buscar_pedidos` para obtener sus pedido_id.

    Devuelve pedido_id y una lista de items (producto, cantidad y precio unitario
    en pesos). Si el pedido no existe, devuelve un campo "error".
    """
    detalle = DETALLE_PEDIDOS.get(pedido_id)
    if detalle is None:
        return {
            "error": f"No existe el pedido {pedido_id}. Verificá el número, o "
            "usá buscar_pedidos para listar los pedidos de un cliente."
        }
    return {"pedido_id": pedido_id, **detalle}


TOOLS = [buscar_pedidos, detalle_pedido]
