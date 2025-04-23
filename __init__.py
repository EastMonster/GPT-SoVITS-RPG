"""The tauri-app."""
import sys

from anyio.from_thread import start_blocking_portal
from pydantic import BaseModel
from pytauri import (
    BuilderArgs,
    Commands,
    builder_factory,
    context_factory,
)

commands: Commands = Commands()

class Info(BaseModel):
    """Info model."""

    name: str

counter = 0

@commands.command()
async def greet() -> Info:
    global counter
    counter += 1
    return Info(name=f"Hello, World! counter = {counter}")

def main() -> None:
    """Run the tauri-app."""
    with start_blocking_portal("asyncio") as portal:
        app = builder_factory().build(
            BuilderArgs(
                context=context_factory(),
                invoke_handler=commands.generate_handler(portal)
            )
        )
        app.run()