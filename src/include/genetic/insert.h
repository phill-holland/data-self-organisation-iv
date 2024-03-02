#include "genetic/templates/genetic.h"
#include "genetic/templates/serialiser.h"
#include "data.h"
#include "point.h"
#include "parameters.h"
#include <vector>
#include <random>
#include <string>

#ifndef _ORGANISATION_GENETIC_INSERT
#define _ORGANISATION_GENETIC_INSERT

namespace organisation
{
    namespace genetic
    {
        namespace inserts
        {
            class value
            {
            public:
                int delay;
                point starting;
                int movementPatternIdx;

            public:
                value(int _delay = 0, point _starting = point(0,0,0), int _movementPatternIdx = 0)
                {
                    delay = _delay;
                    starting = _starting;
                    movementPatternIdx = _movementPatternIdx;
                }

                void clear()
                {
                    delay = 0;
                    starting = point(0,0,0);
                    movementPatternIdx = 0;
                }

            public:
                bool operator==(const value &src) const
                {
                    return delay == src.delay &&
                           starting == src.starting &&
                           movementPatternIdx == src.movementPatternIdx;
                }

                bool operator!=(const value &src) const
                {
                    return delay != src.delay ||
                           starting != src.starting ||
                           movementPatternIdx != src.movementPatternIdx;
                }
            };

            class insert : public templates::genetic, public templates::serialiser
            {
                static std::mt19937_64 generator;

                const static int LENGTH = 15;         
                const static int MIN = 1;            
                const static int MAX = 5;
                
                int _width, _height, _depth;                
                int _max_movement_patterns;

            public:
                std::vector<value> values;

            public:
                insert(parameters &settings) 
                { 
                    _width = settings.width;
                    _height = settings.height;
                    _depth = settings.depth;

                    _max_movement_patterns = settings.max_movement_patterns;
                }

            public:
                size_t size() { return values.size(); }

                void clear() 
                {
                    values.clear();
                }

                bool empty() { return values.empty(); }
                
                void generate(data &source);
                bool mutate(data &source);
                void append(genetic *source, int src_start, int src_end);

                std::string serialise();
                void deserialise(std::string source);

                bool validate(data &source);
                
            public:
                void copy(const insert &source);
                bool equals(const insert &source);
            };
        };
    };
};

#endif