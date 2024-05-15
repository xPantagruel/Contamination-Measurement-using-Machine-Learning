# @file DataClass.py
# @brief definition of data class for more clear code that represents measured value for each contamination layer
# @author Matěj Macek (xmacek27@stud.fit.vutbr.cz)
# @date 4.5.2024

from dataclasses import dataclass

@dataclass
class ProcessedData:
    ImageName: str
    BottomHeightY: int
    TopHeightY: int
    ContaminationHeight: int