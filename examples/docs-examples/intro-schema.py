from airtrain.core.schemas import InputSchema, OutputSchema
from pydantic import BaseModel, Field


class TicketInputSchema(InputSchema):
    customer_id: str
    issue_description: str
    priority: int = Field(default=3, ge=1, le=5)


class TicketOutputSchema(OutputSchema):
    ticket_id: str
    assigned_agent: str
    estimated_time: str
    next_steps: list[str]
