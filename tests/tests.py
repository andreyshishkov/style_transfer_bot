import pytest
from unittest.mock import AsyncMock
from app import start_command, change_style, transfer_style
from settings.messages import START_MESSAGE


@pytest.mark.asyncio
async def test_start_handler():
    message = AsyncMock()
    await start_command(message)

    message.answer.assert_called_with(START_MESSAGE)

