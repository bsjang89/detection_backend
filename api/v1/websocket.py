from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional
import json
import asyncio

router = APIRouter()


class ConnectionManager:
    """
    Manages WebSocket connections for training session monitoring.
    """

    def __init__(self):
        # Map: training_session_id -> set of WebSocket connections
        self.active_connections: Dict[int, Set[WebSocket]] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, websocket: WebSocket, training_session_id: int):
        """Accept and register a WebSocket connection"""
        await websocket.accept()
        self.loop = asyncio.get_running_loop()

        if training_session_id not in self.active_connections:
            self.active_connections[training_session_id] = set()

        self.active_connections[training_session_id].add(websocket)
        print(f"Client connected to training session {training_session_id}")

    def disconnect(self, websocket: WebSocket, training_session_id: int):
        """Remove a WebSocket connection"""
        if training_session_id in self.active_connections:
            self.active_connections[training_session_id].discard(websocket)

            # Clean up empty sets
            if not self.active_connections[training_session_id]:
                del self.active_connections[training_session_id]

        print(f"Client disconnected from training session {training_session_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to a specific client"""
        await websocket.send_text(json.dumps(message))

    async def broadcast(self, message: dict, training_session_id: int):
        """Broadcast message to all clients watching a training session"""
        if training_session_id not in self.active_connections:
            return

        # Create a copy to avoid modification during iteration
        connections = list(self.active_connections[training_session_id])

        for connection in connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                # Remove dead connection
                self.disconnect(connection, training_session_id)

    def broadcast_from_thread(self, message: dict, training_session_id: int):
        """
        Thread-safe broadcast helper for non-async callers (e.g. training thread).
        """
        if self.loop is None or self.loop.is_closed():
            return

        future = asyncio.run_coroutine_threadsafe(
            self.broadcast(message, training_session_id),
            self.loop
        )

        def _log_error(fut):
            try:
                fut.result()
            except Exception as e:
                print(f"Thread-safe broadcast failed: {e}")

        future.add_done_callback(_log_error)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/training/{training_session_id}")
async def training_websocket(
    websocket: WebSocket,
    training_session_id: int
):
    """
    WebSocket endpoint for real-time training monitoring.

    Clients connect to receive live updates during training:
    - Epoch progress
    - Loss metrics
    - Validation metrics
    - Training status changes

    Message format:
    {
        "type": "epoch_update" | "status_update" | "error",
        "data": {...}
    }
    """
    await manager.connect(websocket, training_session_id)

    try:
        # Send initial connection confirmation
        await manager.send_personal_message({
            "type": "connected",
            "training_session_id": training_session_id,
            "message": "Connected to training session"
        }, websocket)

        # Keep connection alive and handle incoming messages
        while True:
            # Wait for messages from client (e.g., ping/pong)
            data = await websocket.receive_text()

            # Echo back (for keep-alive)
            if data == "ping":
                await manager.send_personal_message({
                    "type": "pong"
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket, training_session_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, training_session_id)


# Helper function to broadcast from training service
async def broadcast_training_update(
    training_session_id: int,
    epoch: int,
    metrics: dict
):
    """
    Broadcast training update to all connected clients.
    Called by TrainingService during training.

    Args:
        training_session_id: Training session ID
        epoch: Current epoch number
        metrics: Training metrics dictionary
    """
    message = {
        "type": "epoch_update",
        "training_session_id": training_session_id,
        "epoch": epoch,
        "metrics": metrics
    }

    await manager.broadcast(message, training_session_id)


def broadcast_training_update_sync(
    training_session_id: int,
    epoch: int,
    metrics: dict
):
    """Thread-safe wrapper for broadcasting training updates from sync contexts."""
    message = {
        "type": "epoch_update",
        "training_session_id": training_session_id,
        "epoch": epoch,
        "metrics": metrics
    }
    manager.broadcast_from_thread(message, training_session_id)


async def broadcast_status_update(
    training_session_id: int,
    status: str,
    message: str = None
):
    """
    Broadcast training status update.

    Args:
        training_session_id: Training session ID
        status: New status (running, completed, failed, stopped)
        message: Optional status message
    """
    msg = {
        "type": "status_update",
        "training_session_id": training_session_id,
        "status": status,
        "message": message
    }

    await manager.broadcast(msg, training_session_id)


def broadcast_status_update_sync(
    training_session_id: int,
    status: str,
    message: str = None
):
    """Thread-safe wrapper for broadcasting status updates from sync contexts."""
    msg = {
        "type": "status_update",
        "training_session_id": training_session_id,
        "status": status,
        "message": message
    }
    manager.broadcast_from_thread(msg, training_session_id)
