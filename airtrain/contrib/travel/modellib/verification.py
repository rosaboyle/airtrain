from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import date


class TravelCompanion(BaseModel):
    type: str = Field(..., description="Type of companion (kid/pet/adult)")
    count: int = Field(..., description="Number of companions of this type")
    details: Optional[Dict[str, str]] = Field(
        default=None, description="Additional details like ages, special needs"
    )


class HealthCondition(BaseModel):
    condition: str = Field(..., description="Name of health condition")
    severity: str = Field(..., description="Severity level (mild/moderate/severe)")
    requirements: List[str] = Field(
        ..., description="Special requirements or precautions"
    )


class UserTravelInfo(BaseModel):
    origin: str = Field(..., description="Starting location")
    destination: str = Field(..., description="Travel destination")
    start_date: date = Field(..., description="Travel start date")
    end_date: date = Field(..., description="Travel end date")
    companions: List[TravelCompanion] = Field(default_factory=list)
    outdoor_activities: List[str] = Field(default_factory=list)
    health_conditions: List[HealthCondition] = Field(default_factory=list)
    complete: bool = Field(
        default=False, description="Whether all required info is collected"
    )
