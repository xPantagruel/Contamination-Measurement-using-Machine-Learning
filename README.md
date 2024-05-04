# How to run the program

- run the 'main.py' file using ' Python3 main.py '

## Requirements

- Python 3.8
- libraries from each file

## How to run

- run the 'main.py' file, it will start the program and you will be able to see the figures
- important files:
  - main.py = loading images and iterate through them calling on each image the functions from the ContaminationMeasurementClass.py you can change variable use_multiprocessing = True/False to use multiprocessing or not
  - ContaminationMeasurementClass.py = contains the whole proccess of the workflow for calling image processing functions from image_processing.py
  - image_processing.py = contains all the functions for image processing and measuring functions, also contains the functions for saving the images. In the file are also not needed functions that are not used in the workflow, but were used for testing different approaches.
  - tests.py = contains the file for testing results of the functions from contamination_measurement.py
  - DataClass.py = contains the dataclass ProcessedData
  - AcquireDataset.py = contains the functions for acquiring the dataset from the images, filtering the dataset and saving the dataset, for use it is needed to change the path to the images in the bottom of the file

---

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
object density segmentation : https://www.sciencedirect.com/science/article/pii/S016926070900159X

State of the art:
coating layer https://www.sciencedirect.com/science/article/pii/S0263224122001038
Film thickness https://www.sciencedirect.com/science/article/pii/S0894177715001405
for edge detection otsu and canny https://link.springer.com/article/10.1007/s40815-020-01030-5
for deep learning CNN https://iopscience.iop.org/article/10.35848/1347-4065/ac923d/meta?casa_token=lH_9A5N1rW0AAAAA:52E5x1lxtaVnnG8NRQJr9oCs-P1o60xIRWLOSZ2zQVWuBl_ZZktkEqzkbWaRnkif0PxACTuH5hdESf4Xvz4KG4q9iGg#jjapac923dt4

# TODO

    Add the measuring of the height of the image from the thesis of Kutalik

# IDEAS

    I should Create algorithm that will get from image the height using :
        Normal Old school Aproach
        Segmentational Aproach

# State of art

https://www.sciencedirect.com/science/article/pii/S0894177715001405?casa_token=MJF4LeBNVI4AAAAA:lEQjyi2veWInS2Bw-B1g4ltQtrAiJ3Hp2u4SrP4RPAlvMbk2poskBcu3tQM6qY-i7ZbOg4nOCg
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8300062/

# FIX :

# DATA AUGMENTATION 
https://www.mdpi.com/2076-3417/11/15/6721


PLan:
- WHat to do next: 
  measure error for both datasetds
    it will be measured on uniq dataset where I will choose only the images that the methods is working on and exclude the images without contamination
    I want to measure difference for bottom and for top and than for the height
  validate the error and write about it in the thesis 
  I can try the error measurement on folder WholeDataset and compare results that should be for the model much better



  CODE : 
    error measurement
    refactor whole model training and verify that it is still training right 
    refactor whole first method
    train augmentated model
    I can make comparision of models using the ground truth masks and compare tham to measured by me
  WRITE: 
    handle citations
    implementation second method
    write about training the model on more than 100 epochs 
    write about choosing right model and verifying the model according to results of f1 score and so on 
    compare methods to each other and say which one will be use and is better
    compare scharr and canny
  DEADLINES : 
    21.4. POSTER for Excel at fit
    9.5. BC thesis 


  RESULTS: (error measurements on uniq images)
    First method : 569 succesed, 87 failed 
      height MAE: 5.2555555555555555
      height MSE: 45.522222222222226
      height RMSE: 6.747015801242963
      top MAE: 3.4851851851851854
      top MSE: 21.596296296296295
      top RMSE: 4.647181543290115
      bottom MAE: 6.333333333333333
      bottom MSE: 59.36296296296296
      bottom RMSE: 7.7047363980192705
    Second method : 516 succesed, 142 failed (model)
      height MAE: 6.188679245283019
      height MSE: 66.98113207547169
      height RMSE: 8.184200148791065
      top MAE: 4.509433962264151
      top MSE: 31.11320754716981
      top RMSE: 5.577921436087982
      bottom MAE: 8.20754716981132
      bottom MSE: 102.32075471698113
      bottom RMSE: 10.115372198638127

    Benchmarking the Robustness of Semantic Segmentation Models: https://openaccess.thecvf.com/content_CVPR_2020/papers/Kamann_Benchmarking_the_Robustness_of_Semantic_Segmentation_Models_CVPR_2020_paper.pdf
    DeepLabv3-based brain tumor segmentation: https://www.sciencedirect.com/science/article/pii/S1110016823005379#b0160


state of the art : 
Development of Tool for Contamination Layer Thickness Measurement Using High Power Extreme Ultraviolet Light and in Situ Ellipsometer
:https://sci-hub.se/10.1116/1.3244628

Electron-Beam-Induced Carbon Contamination in STEM-in-SEM: Quantification and Mitigation: https://watermark.silverchair.com/ozac003.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAABcUwggXBBgkqhkiG9w0BBwagggWyMIIFrgIBADCCBacGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMDt5r5V7MzmAQureAAgEQgIIFeA8j1Zq9SE03dBSu2Rp4pU1NugXgLofigt1JHtDc5RtUK0pdsu9BOUEWrdxkSKqt14myKnazuuuh07W05wxvTEFtp6SFIun_fMlFKjUwGOMUpTpsBNteI4Lhs6jHExVRRBdVe7QOPUcE3_N9IxYxZ9A-rnhERxesC-yJ5HOhjNpSg7RFBy-exfWvkUzy7yFYtebiHp0NWtvI0V6rvmdXQh-fUeo8ykn5G2pXp-HgPL2eCy8nyIF_6FUsJTa4DEhzV4tVnDOHnshVrIY9JqJaRdMw_-WqFzXaknuxQaIVarsv0roczrT3Rf2q8IeZxmfV7XwdfO5xMd0IKP5ehmnxn_yXLsZBzG59npAYZZx9x8wwK4lM-t9Di1t2ZaoPVj8kfSzCjJvHMSh8Vq6I9KizYw_uqHFAqrPt3zf0cu1zl7VLc4epY-4w3yt3mVnCvlEqXVhNip4nBH7SonNwOEkfDmg1iCMEPuT5SFVHnHqJsEzB_IXBBXDQxTATRKl1Vz_XER5fH_6OOVMyaUZsM-8S9q-b3Ut3GJ39AWZBU4OKnst0ZT2zm-2PfMNF-bRPaXe7BmB4Yoommg3CLsMslor-Dp0LdiIcs4CFGXy13RENgVtz79UvUpqmVm7PkmKuWx77gQkUikJfnm3BD6H3mfhbBXkp8Aeca84UBxUUoAc_QsYjrMMwadWO8xICT3vvSUQuLyinJ8gwCwqoNZaEl6F1h781dc5O_iv15jyA_T7CrJd14zSf8SBeizymp1jEdhrv3im6DAisQRYf0-WWw426L00emav8dZiWgyG1AjkXEEGXkf-by491GrVpGkBuvKwwFDcuC-sg9fKmSK2Sv1A_azczA9EstIFCbr-GkttAkaDTyKZHoMd4D4Z4_QXWwqdhC-XYMBNC6wfSbg5cyWgG_PExT2y6OYv9kpU1-obc3zIOJD_2OFymqH9oy0eaoC2oCNePTXry8AQ0bew1kdBsbSukPXNNJo9u9OXS3fJCn9SVPcOO4ZiJbDGiPd3su-XtujDXwLd8n0sYHRgj8TZO_pWT1Nn4_XmPxVEXSkrAQxK_p1D087beW6gt9QKnQKv3omp51dmMf0eMGFMWg86pSOhPjmSp36f1jF9jkaLisoKWx4It2lJOmeIXUzUd8lmgwhOJvT-IH9lL7DFQKbFtRkpiHn5UV6WA_b4t5SeXEAWZSnBbVyhKqDuEZbYMg0z5-EDFKd4foYtA8laa9lTT4piKHPJ4FUNvLZiSDhsPghcw6o9IPP7U1DsqeIRAl0xcsb4ZM5oNY3_wn2JcziBra1fHOKac6RXC1GOcXcBJrAlNeWzT8DlNjzXa7hZCxJ9k8Ei3pZdRhrMB4g-T_XI8_OuBW1PDOuX_c-LNzB6fdO4CyvIBV89YRFydX8ouGx5-FziFMsUU_OeT7Jcv3vQbKSkl2n0988BPgmpgXFd9efojch9T8giSWLHH68SGV1_QMOmE4r6nkaw8mVz8X_Xu1b8cYsHd_ufgMVmvMlbimp-DgCKIszYFZ5LSh4aWmydk-glhinCMPabUs4HD_Is7WPhyPY3UcIobg-lwD51Dg3zQhsIWWuUuFBfwDiY_erxQ-JY8ku6MqwZjBwizTo1_DuC5CJAP5UrYzviLhqJviBMlx_b3GuhZD76Lx0Uz2A7gdw9UwcyNPGzCFFcdWVEvsNHXe0AQLsUgZdyyCu4HFC4IAdFNcZk2RPB6auad0xAHv37pq4hhCCvDn4HmUqtlIGErZrL0jHGsse7Zf9YJFbVoiwPSHmV8NgtQP8JEC8Fx340yMI3bKOCDRNIKFgHCQRLU02jYpXkyAFz5kkNhy_vq-LLOFHik3tx75PtdUNIyb-kGFMPRP_JQ

Semantic segmentation mudrock SEM : https://arxiv.org/pdf/2102.03393
deep learning clasification sem : https://direct.mit.edu/dint/article/2/4/513/94893/Deep-Learning-Feature-Learning-and-Clustering
ROI in SEM https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04334-x
Depth estimation from SEM images using deep learning and angular data diversity https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12496/124961E/Depth-estimation-from-SEM-images-using-deep-learning-and-angular/10.1117/12.2658094.full
scharr, edge detection  https://www.mdpi.com/2079-6412/14/3/313