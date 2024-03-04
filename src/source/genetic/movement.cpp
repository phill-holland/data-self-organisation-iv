#include "genetic/movement.h"
#include <sstream>
#include <functional>
#include <iostream>

std::mt19937_64 organisation::genetic::movements::movement::generator(std::random_device{}());

void organisation::genetic::movements::movement::set(int pattern, std::vector<organisation::vector> &source)
{
    for(auto &it: source)
    {
        directions.push_back(std::tuple<int,organisation::vector>(pattern,it));
    }
}

std::string organisation::genetic::movements::movement::serialise()
{
    std::string result;

    for(auto &it: directions)
    {
        int index = std::get<0>(it);
        organisation::vector direction = std::get<1>(it);
    
        result += "M " + direction.serialise();
        result += " " + std::to_string(index);
        result += "\n";
    }

    return result;
}

void organisation::genetic::movements::movement::deserialise(std::string source)
{
    std::stringstream ss(source);
    std::string str;

    int value = 0;
    int index = 0;

    organisation::vector temp;

    while(std::getline(ss,str,' '))
    {
        if(index == 0)
        {
            if(str.compare("M")!=0) return;
        }
        else if(index == 1)
        {
            temp.clear();
            temp.deserialise(str);
        }
        else if(index == 2)
        {
            int pattern_index = std::atoi(str.c_str());
            directions.push_back(std::tuple<int,vector>(pattern_index,temp));
        }

        ++index;
    };
}

bool organisation::genetic::movements::movement::validate(data &source)
{
    if(directions.empty()) 
    { 
        std::cout << "movement::validate(false): directions is empty\r\n"; 
        return false; 
    }

    int previous = 0;
    for(auto &it: directions)
    {
        int index = std::get<0>(it);
        if((index != previous)&&(previous + 1 != index))
        {
            std::cout << "movement::validate(false): indices not sequential (" << index << "," << previous << ")\r\n"; 
            return false;
        }

        if(index >= _max_movement_patterns)
        {
            std::cout << "movement::validate(false): index exceeds max allowed movement patterns (" << index << "," << _max_movement_patterns << ")\r\n"; 
            return false;
        }

        vector direction = std::get<1>(it);

        if((direction.x < -1)||(direction.x > 1)
            ||(direction.y < -1)||(direction.y > 1)
            ||(direction.z < -1)||(direction.z > 1))
        {
            std::cout << "movement::validate(false): direction out of bounds (" << direction.x << "," << direction.y << "," << direction.z << ")\r\n"; 
            return false;
        }
    }

    return true;
}

void organisation::genetic::movements::movement::generate(data &source)
{
    int total_patterns = (std::uniform_int_distribution<int>{1, _max_movement_patterns - 1})(generator);
    for(int j = 0; j < total_patterns; ++j)
    {
        int n = (std::uniform_int_distribution<int>{_min_movements, _max_movements - 1})(generator);

        for(int i = 0; i < n; ++i)
        {
            int value = (std::uniform_int_distribution<int>{0, 26})(generator);
            vector v1;
            if(v1.decode(value))
            {
                directions.push_back(std::tuple<int,organisation::vector>(j,v1));
            }
        }
    }
}

bool organisation::genetic::movements::movement::mutate(data &source)
{
    if(directions.empty()) return false;

    const int COUNTER = 15;

    int n = 0; 
    int value = 0, old = 0;
    int counter = 0;

    do
    {        
        n = (std::uniform_int_distribution<int>{0, (int)(directions.size() - 1)})(generator);
        value = (std::uniform_int_distribution<int>{0, 26})(generator);

        old = std::get<1>(directions[n]).encode();
        vector v1;
        v1.decode(value);
        std::get<1>(directions[n]) = v1;   
    } while((old == value)&&(counter++<COUNTER));    
    
    if(old==value) return false;

    return true;
}

void organisation::genetic::movements::movement::append(genetic *source, int src_start, int src_end)
{
    movement *s = dynamic_cast<movement*>(source);
    int length = src_end - src_start;
    
    int new_index = 0;
    if(directions.size() > 0) new_index = std::get<0>(directions[directions.size() - 1]) + 1;

    int previous_index = 0;
    for(int i = 0; i < length; ++i)
    {
        int current_index = std::get<0>(s->directions[src_start + i]);
        vector current_direction = std::get<1>(s->directions[src_start + i]);

        if((i != 0)&&(previous_index != current_index))
            ++new_index;

        directions.push_back(std::tuple<int,vector>(new_index, current_direction));

        previous_index = current_index;
    }   
}

void organisation::genetic::movements::movement::copy(const movement &source)
{
    directions.assign(source.directions.begin(), source.directions.end());
}

bool organisation::genetic::movements::movement::equals(const movement &source)
{
    if(directions.size() != source.directions.size()) 
        return false;

    for(int i = 0; i < directions.size(); ++i)
    {
        if(directions[i] != source.directions[i]) 
            return false;
    }

    return true;
}