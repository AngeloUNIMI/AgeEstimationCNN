# Age Estimation CNN

Matlab source code for the paper:

	Age estimation based on face images and pre-trained Convolutional Neural Networks, 
	2017 IEEE Symp. on Computational Intelligence for Security and Defense Applications (CISDA 2017),
	Honolulu, HI, USA, November 27–30, 2017
	
Project page:

http://iebil.di.unimi.it/projects/softbio

Demo:

https://github.com/AngeloUNIMI/Demo_AgeEstimationCNN

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

    - launch_all.m: main file

Required files:

    - ./AgeDB: 
    Database of images downloaded from: 
    https://ibug.doc.ic.ac.uk/resources/agedb/
    The structure of the folders must be:
    "./AgeDB/0_MariaCallas_35_f.jpg"
    etc.

Part of the code uses the Matlab source code of matconvnet:

http://www.vlfeat.org/matconvnet/
    
    @inproceedings{vedaldi15matconvnet,
      author    = {A. Vedaldi and K. Lenc},
      title     = {MatConvNet -- Convolutional Neural Networks for MATLAB},
      booktitle = {Proceeding of the {ACM} Int. Conf. on Multimedia},
      year      = {2015},
    }
	
