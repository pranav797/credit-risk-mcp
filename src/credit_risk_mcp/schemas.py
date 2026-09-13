"""User-facing input schema for a borrower.

>>> SCAFFOLD — fill this in. Reference solution in ../reference/schemas.py.

The idea: expose ~13 human-understandable fields instead of the model's 235
columns. Everything is optional EXCEPT the two loan amounts (a loan needs a size,
and the applicant needs an income for the risk to mean anything). Unsupplied
fields fall back to training medians later, in model_service.build_feature_vector.

Why pydantic? It validates input for free (ranges, allowed categories) and the
MCP SDK can turn this model straight into the JSON schema Claude reads. Get the
types right here and later phases get easier.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# TODO: define Literal aliases for the categorical fields so only valid values
# are accepted. The allowed values are the one-hot categories from the guide
# (include the reference category too, e.g. "Businessman"). Example:
#   Gender = Literal["M", "F"]
#   IncomeType = Literal["Working", "Commercial associate", "Pensioner", ...]
#   EducationType = ...
#   FamilyStatus = ...
#   HousingType = ...
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
    the population median when omitted.

    TODO — declare the fields. Suggested set (names matter: model_service maps
    these exact attribute names):
      required: annual_income (>0), credit_amount (>0)
      optional numeric: annuity (>0), age (18-100), years_employed (0-60),
                        num_children (0-20), ext_source_1/2/3 (0-1)
      optional categorical: gender, owns_car ("Y"/"N"), owns_realty ("Y"/"N"),
                        income_type, education, family_status, housing_type

    Use Field(..., gt=0, description=...) for required, Field(None, ge=, le=,
    description=...) for optional. Good descriptions help Claude fill fields
    sensibly later (e.g. note that income is ANNUAL).
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
    # TODO: add the remaining ~13 optional fields described above.
