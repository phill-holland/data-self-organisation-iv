#include "genetic/insert.h"
#include <iostream>
#include <sstream>
#include <functional>

std::mt19937_64 organisation::genetic::inserts::insert::generator(std::random_device{}());

void organisation::genetic::inserts::insert::generate(data &source)
{
    clear();

    int length = (std::uniform_int_distribution<int>{1, LENGTH})(generator);

    std::unordered_map<int, point> duplicates;

    for(int i = 0; i < length; ++i)
    {
        value temp;

        temp.delay = (std::uniform_int_distribution<int>{MIN, MAX})(generator);
        temp.movementPatternIdx = (std::uniform_int_distribution<int>{0, _max_movement_patterns - 1})(generator);
        temp.starting.generate(_width,_height,_depth);

        int index = ((_width * _height) * temp.starting.z) + ((temp.starting.y * _width) + temp.starting.x);
        if(duplicates.find(index) == duplicates.end())
        {
            values.push_back(temp);
        }
    }
}

bool organisation::genetic::inserts::insert::mutate(data &source)
{    
    if(values.empty()) return false;

    const int COUNTER = 15;

    int offset = 0;
    value val, old;
    int counter = 0;
    
    do
    {
        offset = (std::uniform_int_distribution<int>{0, (int)(values.size() - 1)})(generator);        
        int mode = (std::uniform_int_distribution<int>{0, 2})(generator);

        if(mode == 0)
        {
            val.delay = (std::uniform_int_distribution<int>{MIN, MAX})(generator);
        }
        else if(mode == 1)
        {
            val.movementPatternIdx = (std::uniform_int_distribution<int>{0, _max_movement_patterns - 1})(generator);
        }
        else if(mode == 2)
        {          
            std::unordered_map<int, point> duplicates;

            for(auto &it: values)
            {
                int index = ((_width * _height) * it.starting.z) + ((it.starting.y * _width) + it.starting.x);    
                duplicates[index] = it.starting;
            }  

            int index = 0;
            int duplicates_counter = 0;
            do 
            {
                val.starting.generate(_width,_height,_depth);         
                index = ((_width * _height) * val.starting.z) + ((val.starting.y * _width) + val.starting.x);    
            }while((duplicates.find(index) != duplicates.end())&&(duplicates_counter++<COUNTER));

            if(duplicates_counter >= COUNTER) return false;
        }
            
        old = values[offset];        
        values[offset] = val;

    }while((old==val)&&(counter++<COUNTER));

    if(old==val) return false;

    return true;   
}

void organisation::genetic::inserts::insert::append(genetic *source, int src_start, int src_end)
{
    insert *s = dynamic_cast<insert*>(source);

    int length = src_end - src_start;  

    for(int i = 0; i < length; ++i)
    {
        values.push_back(s->values[src_start + i]);
    }   
}

std::string organisation::genetic::inserts::insert::serialise()
{
    std::string result;

    for(auto &it: values)
    {
        result += "I " + std::to_string(it.delay);
        result += " " + it.starting.serialise();
        result += " " + std::to_string(it.movementPatternIdx);
        result += "\n";
    }

    return result;
}

void organisation::genetic::inserts::insert::deserialise(std::string source)
{
    std::stringstream ss(source);
    std::string str;

    value value;
    int index = 0;

    while(std::getline(ss,str,' '))
    {
        
        if(index == 0)
        {
            if(str.compare("I")!=0) return;    
            value.clear();
        }
        else if(index == 1)
        {
            value.delay = std::atoi(str.c_str());         
        }
        else if(index == 2)
        {
            value.starting.deserialise(str);
        }
        else if(index == 3)
        {
            value.movementPatternIdx = std::atoi(str.c_str());
            values.push_back(value);            
        }

        ++index;
    };
}

bool organisation::genetic::inserts::insert::validate(data &source)
{
    if(values.empty()) { std::cout << "insert::validate(false): values is empty\r\n"; return false; }

    std::unordered_map<int, point> duplicates;

    for(auto &it: values)
    {
        if((it.delay < MIN)||(it.delay > MAX))
        { 
            std::cout << "insert::validate(false): delay out of bounds\r\n"; 
            return false; 
        }

        if(!it.starting.inside(_width,_height,_depth))
        {
            std::cout << "insert::validate(false): invalid starting position\r\n"; 
            return false; 
        }

        int index = ((_width * _height) * it.starting.z) + ((it.starting.y * _width) + it.starting.x);
        if(duplicates.find(index) == duplicates.end())
            duplicates[index] = it.starting;
        else
        {
            std::cout << "insert::validate(false): duplicate starting position\r\n"; 
            return false;  
        }

        if((it.movementPatternIdx < 0)||(it.movementPatternIdx >= _max_movement_patterns))
        {
            std::cout << "insert::validate(false): movement out of bounds\r\n"; 
            return false; 
        }
    }

    return true;
}

void organisation::genetic::inserts::insert::copy(const insert &source)
{
    values.assign(source.values.begin(), source.values.end());
}

bool organisation::genetic::inserts::insert::equals(const insert &source)
{
    if(values.size() != source.values.size()) 
        return false;

    for(int i = 0; i < values.size(); ++i)
    {
        if(values[i] != source.values[i]) 
            return false;
    }

    return true;
}