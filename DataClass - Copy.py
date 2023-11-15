from dataclasses import dataclass

@dataclass
class ProcessedData:
    ImageName: str
    BottomHeightY: int
    TopHeightY: int
    ContaminationHeight: int