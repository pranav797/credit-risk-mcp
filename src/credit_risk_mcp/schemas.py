"""User-facing input schema for a borrower.

Exposes ~13 human-understandable fields instead of the model's 235 columns.
Everything is optional EXCEPT the two loan amounts (a loan needs a size, and the
applicant needs an income for the risk to mean anything). Unsupplied fields fall
back to training medians in model_service.build_feature_vector.

Pydantic validates input for free (ranges, allowed categories) and the MCP SDK
turns this model straight into the JSON schema Claude reads.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Literal aliases so only valid category values are accepted (these mirror the
# one-hot categories, including each group's reference level).
Gender = Literal["M", "F"]
YesNo = Literal["Y", "N"]
IncomeType = Literal[
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Student",
    "Unemployed",
    "Businessman",
    "Maternity leave",
]
EducationType = Literal[
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
FamilyStatus = Literal[
    "Married",
    "Single / not married",
    "Civil marriage",
    "Separated",
    "Widow",
    "Unknown",
]
HousingType = Literal[
    "House / apartment",
    "With parents",
    "Municipal apartment",
    "Rented apartment",
    "Office apartment",
    "Co-op apartment",
]


class BorrowerProfile(BaseModel):
    """A loan applicant, described in plain terms.

    Only `annual_income` and `credit_amount` are required; all else defaults to
    the population median when omitted. The field descriptions matter: they are
    what Claude reads to fill the fields sensibly (e.g. income is ANNUAL).
    """

    model_config = {"extra": "forbid"}
    
    annual_income: float = Field(
        ..., gt=0, description="Total yearly income in the loan's currency units."
    )
    credit_amount: float = Field(
        ..., gt=0, description="Total loan amount requested."
    )
    annuity: Optional[float] = Field(
        None, gt=0, description="Loan annuity: the amount paid each period."
    )
    age: Optional[float] = Field(
        None, ge=18, le=100, description="Applicant age in years."
    )
    years_employed: Optional[float] = Field(
        None, ge=0, le=60, description="Years at current employment."
    )
    gender: Optional[Gender] = Field(None, description="M or F.")
    owns_car: Optional[YesNo] = Field(None, description="Y if the applicant owns a car.")
    owns_realty: Optional[YesNo] = Field(
        None, description="Y if the applicant owns real estate."
    )
    num_children: Optional[int] = Field(
        None, ge=0, le=20, description="Number of children."
    )
    income_type: Optional[IncomeType] = Field(None, description="Category of income.")
    education: Optional[EducationType] = Field(
        None, description="Highest education level."
    )
    family_status: Optional[FamilyStatus] = Field(None, description="Marital status.")
    housing_type: Optional[HousingType] = Field(
        None, description="Housing situation."
    )
    ext_source_1: Optional[float] = Field(
        None, ge=0, le=1, description="Normalised external credit-bureau score #1 (0-1)."
    )
    ext_source_2: Optional[float] = Field(
        None, ge=0, le=1, description="Normalised external credit-bureau score #2 (0-1)."
    )
    ext_source_3: Optional[float] = Field(
        None, ge=0, le=1, description="Normalised external credit-bureau score #3 (0-1)."
    )
