#include <gtest/gtest.h>
#include "parallel/inserts.hpp"
#include "parallel/program.hpp"
#include "parallel/map/configuration.hpp"
#include "general.h"
#include "data.h"
#include "schema.h"
#include "kdpoint.h"
#include <unordered_map>

TEST(BasicProgramInsertParallel, BasicAssertions)
{    
    //GTEST_SKIP();

    const int width = 10, height = 10, depth = 10;

    std::string input1("daisy daisy give me your answer do .");
    std::string input2("monkey monkey eat my face .");

    std::vector<std::string> strings = organisation::split(input1 + " " + input2);
    organisation::data mappings(strings);

	::parallel::device device(0);
	::parallel::queue queue(device);

    organisation::parameters parameters(width, height, depth);
    parameters.mappings = mappings;
    
    parameters.dim_clients = organisation::point(1,1,1);

    organisation::inputs::epoch epoch1(input1);
    organisation::inputs::epoch epoch2(input2);

    parameters.input.push_back(epoch1);
    parameters.input.push_back(epoch2);

    organisation::parallel::inserts inserts(device, &queue, parameters);

    organisation::schema s1(parameters);

    organisation::genetic::inserts::insert insert(parameters);

    organisation::point starting(width/2,height/2,depth/2);

    organisation::genetic::inserts::value a(1, organisation::point(starting.x,starting.y,starting.z), 0);
    organisation::genetic::inserts::value b(2, organisation::point(starting.x,starting.y+1,starting.z), 0);
    organisation::genetic::inserts::value c(3, organisation::point(starting.x,starting.y+3,starting.z), 0);

    //insert.values = { 1,2,3 };    
    insert.values = { a, b, c };
    s1.prog.set(insert);

    std::vector<organisation::schema*> source = { &s1 };
    
    inserts.copy(source.data(), source.size());
    inserts.set(mappings, parameters.input);
    
    std::vector<int> expected_insert_counts1 = { 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    std::vector<int> expected_insert_value1 = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 0, 4, 0, 5, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0 };

    for(int i = 0; i < 30; ++i)    
    {
        int count = inserts.insert(0);
        auto data = inserts.get();

        EXPECT_EQ(expected_insert_counts1[i], count);
        EXPECT_EQ(data.size(), count);

std::cout << "i: " << i << " count:" << count << "\r\n";
        if(count > 0) 
        { 
std::cout << "x,y,z " << data[0].position.serialise() << "\r\n";         
            EXPECT_EQ(expected_insert_value1[i], data[0].data.x);
            EXPECT_EQ(organisation::point(width / 2, height / 2, depth / 2),data[0].position);
            EXPECT_EQ(0, data[0].client);
        }        
    }
    
    inserts.clear();

    std::vector<int> expected_insert_counts2 = { 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    std::vector<int> expected_insert_value2 =  { 0, 7, 0, 0, 7, 0, 0, 0, 8, 0, 9, 0, 0,10, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    for(int i = 0; i < 30; ++i)    
    {
        int count = inserts.insert(1);
        auto data = inserts.get();

        EXPECT_EQ(expected_insert_counts2[i], count);
        EXPECT_EQ(data.size(), count);

        if(count > 0) 
        { 
            EXPECT_EQ(expected_insert_value2[i], data[0].data.x);
            EXPECT_EQ(organisation::point(width / 2, height / 2, depth / 2),data[0].position);
            EXPECT_EQ(0, data[0].client);
        }        
    }
}

