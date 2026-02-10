from __future__ import annotations

from typing import Any, List, Optional, Dict
from pydantic import BaseModel, ConfigDict, Field


class ReferenceNumber(BaseModel):
    model_config = ConfigDict(extra="allow")
    ReferenceNumber: str
    Type: str
    IsPrimary: bool = False


class ServiceFlag(BaseModel):
    model_config = ConfigDict(extra="allow")
    ServiceCode: str
    IsSelected: bool = False


class Dates(BaseModel):
    model_config = ConfigDict(extra="allow")
    EarliestPickupDate: Optional[str] = None
    LatestPickupDate: Optional[str] = None
    EarliestDropDate: Optional[str] = None
    LatestDropDate: Optional[str] = None


class Contact(BaseModel):
    model_config = ConfigDict(extra="allow")
    Name: Optional[str] = None
    Email: Optional[str] = None
    Phone: Optional[str] = None


class Party(BaseModel):
    model_config = ConfigDict(extra="allow")
    Name: Optional[str] = None
    AddressLine1: Optional[str] = None
    AddressLine2: Optional[str] = None
    City: Optional[str] = None
    StateProvince: Optional[str] = None
    PostalCode: Optional[str] = None
    CountryCode: Optional[str] = None
    IsResidential: Optional[bool] = None
    Comments: Optional[str] = None
    Contact: Optional[Contact] = None


class Dimensions(BaseModel):
    model_config = ConfigDict(extra="allow")
    Length: Optional[float] = None
    Width: Optional[float] = None
    Height: Optional[float] = None
    Uom: Optional[str] = None


class FreightClasses(BaseModel):
    model_config = ConfigDict(extra="allow")
    FreightClass: Optional[float] = None
    Type: Optional[str] = None


class Weights(BaseModel):
    model_config = ConfigDict(extra="allow")
    Actual: Optional[float] = None
    Uom: Optional[str] = None


class Quantities(BaseModel):
    model_config = ConfigDict(extra="allow")
    Actual: Optional[float] = None
    Uom: Optional[str] = None


class Item(BaseModel):
    model_config = ConfigDict(extra="allow")
    Id: Optional[str] = None
    Description: Optional[str] = None
    Dimensions: Optional[Dimensions] = None
    FreightClasses: Optional[FreightClasses] = None
    NmfcCode: Optional[str] = None
    HazardousMaterial: Optional[bool] = None
    Weights: Optional[Weights] = None
    Quantities: Optional[Quantities] = None


class PaymentAddress(BaseModel):
    model_config = ConfigDict(extra="allow")
    Name: Optional[str] = None
    AddressLine1: Optional[str] = None
    AddressLine2: Optional[str] = None
    City: Optional[str] = None
    StateProvince: Optional[str] = None
    PostalCode: Optional[str] = None
    CountryCode: Optional[str] = None
    IsResidential: Optional[bool] = None
    Comments: Optional[str] = None
    Contact: Optional[Contact] = None


class Payment(BaseModel):
    model_config = ConfigDict(extra="allow")
    Address: Optional[PaymentAddress] = None


class Shipment(BaseModel):
    """
    Lenient model:
    - We strongly type what we render
    - We allow extra keys everywhere so Sheets can evolve payload
    """
    model_config = ConfigDict(extra="allow")

    Status: Optional[str] = None
    ReferenceNumbers: List[ReferenceNumber] = Field(default_factory=list)
    ServiceFlags: List[ServiceFlag] = Field(default_factory=list)
    Dates: Optional[Dates] = None
    Shipper: Optional[Party] = None
    Consignee: Optional[Party] = None
    Items: List[Item] = Field(default_factory=list)
    Payment: Optional[Payment] = None

    def primary_reference(self) -> Optional[str]:
        for r in self.ReferenceNumbers or []:
            if getattr(r, "IsPrimary", False):
                return r.ReferenceNumber
        # fallback: first ref number if exists
        if self.ReferenceNumbers:
            return self.ReferenceNumbers[0].ReferenceNumber
        return None

    def selected_service_codes(self) -> List[str]:
        codes: List[str] = []
        for s in self.ServiceFlags or []:
            if getattr(s, "IsSelected", False) and s.ServiceCode:
                codes.append(s.ServiceCode)
        return codes
