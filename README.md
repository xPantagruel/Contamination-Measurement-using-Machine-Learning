# ContaminationMeasurement

## Research sources:

Thesis:
Ice on lines measuring: https://home.cc.umanitoba.ca/~petersii/wren/images/ci_dissertations/Borkowski_MSc2002.pdf
Canny edge detector : http://bigwww.epfl.ch/demo/ip/demos/edgeDetector/
Edge Detection thesis: https://core.ac.uk/download/pdf/62729625.pdf
blurring methods: https://dspace.library.uu.nl/handle/1874/320867
thresholding and working with noisy images(from Manuel): https://is.muni.cz/th/igph2/bachelor_thesis.pdf
MEASURING THE THICKNESS OF MATERIAL LAYERS REMOVED FROM A SAMPLE IN AN ELECTRON MICROSCOPE : https://www.vut.cz/www_base/zav_prace_soubor_verejne.php?file_id=253832
Evaluation of chamber contamination in a scanning electron microscope:https://pubs.aip.org/avs/jvb/article/27/6/2711/591098/Evaluation-of-chamber-contamination-in-a-scanning
GOOD FOR RECOURCES SEM -
Studium kovových materiálů pomocí nízkonapěťové elektronové mikroskopie: https://www.vut.cz/www_base/zav_prace_soubor_verejne.php?file_id=54463
SCANNING ELECTRON MICROSCOPY (SEM): A REVIEW - https://fluidas.ro/hervex/proceedings2018/77-85.pdf
Scanning Electron Microscopy - https://www.intechopen.com/books/1505
An Approach to the Reduction of Hydrocarbon Contamination in the ScanningElectron Microscope - https://onlinelibrary.wiley.com/doi/epdf/10.1002/sca.1996.4950180402
Electron Beam-Induced Sample Contamination in the SEM A. E. Vladár and M. T. Postek - https://www.cambridge.org/core/services/aop-cambridge-core/content/view/74D0DE119CB72062BB022E0146B9F3F6/S1431927605507785a.pdf/electron-beam-induced-sample-contamination-in-the-sem.pdf { https://www.cambridge.org/core/journals/microscopy-and-microanalysis/article/electron-beaminduced-sample-contamination-in-the-sem/74D0DE119CB72062BB022E0146B9F3F6#article}
Electron-Beam-Induced Carbon Contamination in STEM-in-SEM: Quantification and Mitigation Milena Hugenschmidt, Katharina Adrion, Aaron Marx, Erich Müller, Dagmar Gerthsen https://academic.oup.com/mam/article/29/1/219/6927139
contamination:https://academic.oup.com/mam/article/29/1/219/6927139
U-net cutting edge " https://www.sciencedirect.com/science/article/pii/S2351978920315869
segmentation :https://www.sciencedirect.com/science/article/pii/S0263224122001038

## TODO:

    create unit tests
    get the start and end of the bottom of the contamination
    check if bottom is higher than top of tin ball sides (if not then it is not contamination)

## What I do durring week:

    29/09/2023
        - working with adaptive thresholding
            results: block size 31 and c = -10 shows best thresholding from image when after is used normal thresholding
        - find out use thresholding then adaptive and after again thresholding gets from image the area with contamination pretty well on most of the images

# IDEAS 
    I should Create algorithm that will get from image the height using :
        Normal Old school Aproach
        Segmentational Aproach
    I can as a experiment for threshold create app that will let me set the best threshold for each image and than it will save it to file and I will make some statistics from it