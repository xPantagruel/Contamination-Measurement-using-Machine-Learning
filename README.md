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
## TODO:
    Get someone to show me how the contamination measurement is working

## What I do durring week:
    29/09/2023 
        - working with adaptive thresholding
            results: block size 31 and c = -10 shows best thresholding from image when after is used normal thresholding 
        - find out use thresholding then adaptive and after again thresholding gets from image the area with contamination pretty well on most of the images  