/*******************************************************************************************************************
@Copyright 2018 Inspur Co., Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files(the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of the Software, and 
to permit persons to whom the Software is furnished to do so, subject to the following conditions : 

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

@Filename:  master.cpp
@Author:    Michael.Chen
@Version:   5.0
@Date:      31st/Jul/2018
*******************************************************************************************************************/

#include "setting.hpp"
#include "DetectProcess.hpp" 
#include <thread> 
#include <unistd.h>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/************************************************* 
    Function:       main
    Description:    function entrance
    Calls:          
                    DetectProcess::ImageProducer()
                    DetectProcess::ImageConsumer() 
    Input:          None 
    Output:         None 
    Return:         return 0
    *************************************************/
int main(int argc, char * argv[])
{
    char *config_file_name = "../param/param_config.xml";
    FileStorage fs(config_file_name, FileStorage::READ);    //initialization
    if (!fs.isOpened())     //open xml config file
    {
        std::cout << "Could not open the configuration file: param_config.xml " << std::endl;
        return -1;
    }
    Settings setting(config_file_name);

    DetectProcess image_cons_prod(&setting);
    std::thread task0(&DetectProcess::ImageReader, image_cons_prod);  // add image reading thread
    std::thread task1(&DetectProcess::ImageProcesser, image_cons_prod);  // add image processing thread

    task0.join();
    task1.join();
    
	return EXIT_SUCCESS;
}
 
