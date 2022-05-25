//
// Created by lazo2 on 28.04.2022.
//

#ifndef TIM_TIMER_H
#define TIM_TIMER_H
#include <chrono>

class Timer {
public:
    Timer(){
        start = std::chrono::high_resolution_clock::now();
    }
    ~Timer(){
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout<<"DURATION "<<duration.count()<<" s\n\n";
    }
private:
    std::chrono::time_point<std::chrono::steady_clock> start, end;
};


#endif //TIM_TIMER_H
