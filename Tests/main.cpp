#include <iostream>
#include<doctest.hpp>
#include<opencv2/opencv.hpp>
#include<torch/torch.h>

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.

TEST_CASE("TestingLibtorchOpenCVLinking") {
    // TIP Press <shortcut actionId="RenameElement"/> when your caret is at the <b>lang</b> variable name to see how CLion can help you rename it.

    torch::Tensor tensor = torch::tensor({4,3,4},torch::kCUDA);
    torch::Tensor tensor2 = tensor.clone();

    std::cout<<tensor<<"\n";
    cv::Mat image=  cv::Mat(10,10,CV_8U);
    std::cout<<image.size()<<"\n";
    std::cout<<image<<"\n";


    const auto lang = "C++";
    std::cout << "Hello and welcome to " << lang << "!\n";

    for (int i = 1; i <= 5; i++)
    {
        // TIP Press <shortcut actionId="Debug"/> to start debugging your code. We have set one <icon src="AllIcons.Debugger.Db_set_breakpoint"/> breakpoint for you, but you can always add more by pressing <shortcut actionId="ToggleLineBreakpoint"/>.
        std::cout << "i = " << i << std::endl;
    }


    // TIP See CLion help at <a href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>. Also, you can try interactive lessons for CLion by selecting 'Help | Learn IDE Features' from the main menu.
}