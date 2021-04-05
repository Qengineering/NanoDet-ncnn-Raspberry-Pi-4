# NanoDet on Raspberry Pi 4
![output image]( https://qengineering.eu/images/JamesNanoDet_RPi.jpg )
## NanoDet with the ncnn framework. <br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)<br/><br/>
Paper: https://arxiv.org/pdf/1912.02424.pdf<br/><br/>
Special made for a bare Raspberry Pi 4 see [Q-engineering deep learning examples](https://qengineering.eu/deep-learning-examples-on-raspberry-32-64-os.html)

------------

## Benchmark.
| Model  | size | objects | mAP | Jetson Nano 2015 MHz | RPi 4 64-OS 1950 MHz |
| ------------- | :-----:  | :-----:  | :-----:  | :-------------:  | :-------------: |
| NanoDet | 320x320 | 80 | 20.6  |  21.1 FPS | **13.0 FPS** |
| YoloV2 | 416x416  | 20 | 19.2 |  10.1 FPS | 3.0 FPS |
| YoloV3 | 352x352 tiny | 20 | 16.6 | 17.7 FPS | 4.4 FPS |
| YoloV4 | 416x416 tiny | 80 | 21.7 | 11.2 FPS | 3.4 FPS |
| YoloV4 | 608x608 full | 80 | 45.3 | 0.7 FPS | 0.2 FPS |
| YoloV5 | 640x640 small| 80 | 22.5 | 4.0 FPS | 1.6 FPS |

------------

## Dependencies.<br/>
### Due to a not yet solved compiler issue [#2812](https://github.com/Tencent/ncnn/issues/2812), use only ncnn version 20200916 or earlier!<br/>
```git clone -b 20200916 --depth 1 https://github.com/Tencent/ncnn.git```<br/><br/>
To run the application, you have to:
- A raspberry Pi 4 with a 32 or 64-bit operating system. It can be the Raspberry 64-bit OS, or Ubuntu 18.04 / 20.04. [Install 64-bit OS](https://qengineering.eu/install-raspberry-64-os.html) <br/>
- The Tencent ncnn framework installed. [Install ncnn](https://qengineering.eu/install-ncnn-on-raspberry-pi-4.html) <br/>
- OpenCV 64 bit installed. [Install OpenCV 4.5](https://qengineering.eu/install-opencv-4.5-on-raspberry-64-os.html) <br/>
- Code::Blocks installed. (```$ sudo apt-get install codeblocks```)

------------

## Installing the app.
To extract and run the network in Code::Blocks <br/>
$ mkdir *MyDir* <br/>
$ cd *MyDir* <br/>
$ wget https://github.com/Qengineering/YoloV5-ncnn-Raspberry-Pi-4/archive/refs/heads/master.zip <br/>
$ unzip -j master.zip <br/>
Remove master.zip, LICENSE and README.md as they are no longer needed. <br/> 
$ rm master.zip <br/>
$ rm LICENSE <br/>
$ rm README.md <br/> <br/>
Your *MyDir* folder must now look like this: <br/> 
James.mp4 <br/>
parking.jpg <br/>
busstop.jpg <br/>
NanoDet.cpb <br/>
NanoDet.cpp <br/>
nanodet_m.bin <br/>
nanodet_m.param <br/>

------------

## Running the app.
To run the application load the project file NanoDet.cbp in Code::Blocks.<br/> 
Next, follow the instructions at [Hands-On](https://qengineering.eu/deep-learning-examples-on-raspberry-32-64-os.html#HandsOn).<br/><br/>
Many thanks to [nihui](https://github.com/nihui/) again!<br/>
![output image]( https://qengineering.eu/images/test_busNanoDet.jpg )

