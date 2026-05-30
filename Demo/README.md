# Age Estimation CNN (Demo)

Matlab Demo source code for the paper:

	Age estimation based on face images and pre-trained Convolutional Neural Networks, 
	2017 IEEE Symp. on Computational Intelligence for Security and Defense Applications (CISDA 2017),
	Honolulu, HI, USA, November 27–30, 2017
	
Project page:

http://iebil.di.unimi.it/projects/softbio

Source code:

https://github.com/AngeloUNIMI/AgeEstimationCNN

Citation:

    @INPROCEEDINGS{8285381,
        author={A. {Anand} and R. {Donida Labati} and A. {Genovese} and E. {Muñoz} and V. {Piuri} and F. {Scotti}},
        booktitle={2017 IEEE Symposium Series on Computational Intelligence (SSCI)},
        title={Age estimation based on face images and pre-trained convolutional neural networks},
        year={2017},
        pages={1-7},
        doi={10.1109/SSCI.2017.8285381},
        month={Nov},}

Main files:

    - launch_ageEstimation_demo: main file
    
Required files

    - \results\AgeDB\pca.mat
    if not downloaded by Github LFS, download at:
    https://drive.google.com/open?id=14eiuSH0n8oXlv-muHeA19RkxykPuQzVu

    - \results\AgeDB\PCA30DB_Results_CONCAT.mat
    if not downloaded by Github LFS, download at:
    https://drive.google.com/open?id=1lPglz7ilE2DNIel_ISSeinPDnygreJA-

Part of the code uses the Matlab source code of matconvnet:

http://www.vlfeat.org/matconvnet/
    
    @inproceedings{vedaldi15matconvnet,
      author    = {A. Vedaldi and K. Lenc},
      title     = {MatConvNet -- Convolutional Neural Networks for MATLAB},
      booktitle = {Proceeding of the {ACM} Int. Conf. on Multimedia},
      year      = {2015},
    }
	
