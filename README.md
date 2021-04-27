# Vehicle Color Recognition using CNN 
Vehicle information recognition is a key component of Intelligent Transportation Systems. Color plays an important role in vehicle identification. As a vehicle has its inner structure, the main challenge of vehicle color recognition is to select the region of interest (ROI) for recognizing its dominant color.

## Training 	
> Note : For inference tasks only, the weights can be [downloaded](https://drive.google.com/drive/folders/1iBAn9IwWXY8Ur4JA89ZkIOP4MSjtDea0?usp=sharing) and training is not required.   

In case you want to train the model (preferably, with addition data), edit the	`Train.ipynb` notebook in `Training` directory.

## Running 
Download weights from [here](https://drive.google.com/drive/folders/1iBAn9IwWXY8Ur4JA89ZkIOP4MSjtDea0?usp=sharing)  
Edit `config.py` file and modify `MODEL_PATH` variable to the full path of the downloaded weights. 

Run the server using `python3 api_server.py --port 1234`.

> Note : The server assumes that the images will be sent as a multipart file using 'image' as the `key`. 

## References

Link to Original Project - [Link](http://cloud.eic.hust.edu.cn:8071/~pchen/project.html)

References
P. Chen, X. Bai and W. Liu, "Vehicle Color Recognition on Urban Road by Feature Context," in IEEE Transactions on Intelligent Transportation Systems, vol. 15, no. 5, pp. 2340-2346, Oct. 2014, doi: 10.1109/TITS.2014.2308897.

"Vehicle Color Recognition using Convolutional Neural Network",<br />
Reza Fuad Rachmadi and I Ketut Eddy Purnama<br />
https://arxiv.org/abs/1510.07391
