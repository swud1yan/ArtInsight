"""Parsing helpers for image preprocessing operation strings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationSpec:
    """Parsed operation name and optional argument."""

    name: str
    argument: str


@dataclass(frozen=True)
class ResizeSize:
    """Parsed resize target dimensions."""

    width: int
    height: int


@dataclass(frozen=True)
class CropBox:
    """Parsed crop rectangle using exclusive right/bottom coordinates."""

    left: int
    top: int
    right: int
    bottom: int


class OperationParser:
    """Parse preprocessing operation strings into structured values."""

    def split_operation(self, operation: str) -> OperationSpec:
        """Split an operation into lowercase name and optional argument."""
        if not operation or not operation.strip():
            msg = "Operation must be a non-empty string."
            raise ValueError(msg)
        name, separator, argument = operation.strip().partition(":")
        return OperationSpec(name=name.lower(), argument=argument if separator else "")

    def parse_size(self, argument: str) -> ResizeSize:
        """Parse a size argument formatted as WIDTHxHEIGHT."""
        if "x" not in argument.lower():
            msg = "Size must use WIDTHxHEIGHT format."
            raise ValueError(msg)
        width_text, height_text = argument.lower().split("x", maxsplit=1)
        width = self.parse_positive_int(width_text, "width")
        height = self.parse_positive_int(height_text, "height")
        return ResizeSize(width=width, height=height)

    def parse_crop_box(self, argument: str) -> CropBox:
        """Parse crop coordinates formatted as left,top,right,bottom."""
        parts = [part.strip() for part in argument.split(",")]
        if len(parts) != 4:
            msg = "Crop coordinates must use left,top,right,bottom format."
            raise ValueError(msg)
        try:
            left, top, right, bottom = (int(part) for part in parts)
        except ValueError as exc:
            msg = "Crop coordinates must be integers."
            raise ValueError(msg) from exc
        return CropBox(left=left, top=top, right=right, bottom=bottom)

    def parse_positive_int(self, argument: str, field_name: str) -> int:
        """Parse a positive integer argument."""
        try:
            value = int(argument)
        except ValueError as exc:
            msg = f"{field_name} must be an integer."
            raise ValueError(msg) from exc
        if value <= 0:
            msg = f"{field_name} must be positive."
            raise ValueError(msg)
        return value