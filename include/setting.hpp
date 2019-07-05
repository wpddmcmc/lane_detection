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

@Filename:  setting.hpp
@Author:    Michael.Chen
@Version:   5.0
@Date:      31st/Jul/2018
*******************************************************************************************************************/

#pragma once
#include "opencv2/core/core.hpp"
#include <string>

using namespace cv;

class Settings{
public:
    Settings(const std::string & filename)
	{
        FileStorage setting_fs(filename, FileStorage::READ);
        read(setting_fs);
        setting_fs.release();
    }
    void read(const FileStorage& fs) {
        fs["debug_mode"] >> debug_mode;
        fs["video_name"] >> video_name;
    }
public:
    int debug_mode;
    std::string video_name;
};